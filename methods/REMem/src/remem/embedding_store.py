import logging
import os
import pickle
from copy import deepcopy
from typing import Dict, List

import numpy as np
import pandas as pd

from remem.utils.misc_utils import compute_mdhash_id

logger = logging.getLogger(__name__)


class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace, from_scratch=False):
        """
        :param embedding_model: An instance that provides a batch_encode() method.
        :param db_filename: Path to the SQLite database file.
        :param columns: Optional list of extra columns to store (as TEXT). Must include "content".
                        For example: ["content", "head", "relation", "tail", "type"]
                        If not provided, the default table schema is used.
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename = os.path.join(db_filename, f"vdb_{self.namespace}.pkl")
        self._load_data(from_scratch)

    def insert_strings(self, texts: List[str], embed=True):
        nodes_dict = {}

        for text in texts:
            if not isinstance(text, str):
                continue
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {"content": text}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  # Nothing to insert.

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        logger.info(
            f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist."
        )

        if not missing_ids:
            return {}  # All records already exist.

        # Prepare the texts to encode from the "content" field.
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        if embed:
            missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)
        else:
            missing_embeddings = [None] * len(missing_ids)

        self._upsert(missing_ids, texts_to_encode, missing_embeddings)

    def insert_chunk_dicts(self, chunk_meta: List[Dict], extract_method=None, embed=True, remove_qualifiers=True):
        nodes_dict = {}

        from remem.utils.chunk_utils import make_chunk_content

        texts = [make_chunk_content(extract_method, chunk) for chunk in chunk_meta]
        for text_idx, text in enumerate(texts):
            assert isinstance(text, str)
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {
                "content": text,
                "metadata": chunk_meta[text_idx],
            }

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return nodes_dict  # Nothing to insert.

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        logger.info(
            f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist."
        )

        if not missing_ids:
            return nodes_dict  # All records already exist.

        # Prepare the texts to encode from the "content" field.
        contents = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        texts_to_encode = contents
        if extract_method == "temporal" and remove_qualifiers and texts_to_encode[0].startswith("("):
            texts_to_encode = contents.copy()
            texts_to_encode = [text.split(', {"start_time')[0].strip() for text in texts_to_encode]
            texts_to_encode = [text.replace(", ", " ")[1:-1] for text in texts_to_encode]
        chunks_metadata_to_encode = [nodes_dict[hash_id]["metadata"] for hash_id in missing_ids]

        if embed:
            missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)
        else:
            missing_embeddings = [None] * len(missing_ids)

        self._upsert(missing_ids, contents, missing_embeddings, chunks_metadata_to_encode)
        return nodes_dict

    def _validate_and_fix_embeddings(self, embeddings, context=""):
        """
        Validate embeddings and replace problematic ones with zero vectors.

        Args:
            embeddings: List of embeddings to validate
            context: Context string for logging (e.g., "during load", "during upsert")

        Returns:
            tuple: (processed_embeddings, none_count, corrupted_count, embedding_dim)
        """
        none_count = 0
        corrupted_count = 0
        embedding_dim = None

        # First pass: find embedding dimension from valid embeddings
        for emb in embeddings:
            if emb is not None and hasattr(emb, "__len__"):
                try:
                    emb_array = np.asarray(emb)
                    # Check if the embedding contains any None/NaN values
                    if not np.any(pd.isna(emb_array)) and emb_array.size > 0:
                        embedding_dim = emb_array.shape[0] if emb_array.ndim == 1 else emb_array.shape[-1]
                        break
                except:
                    continue

        # If no valid embedding found, try to get dimension from existing data or model
        if embedding_dim is None:
            if hasattr(self, "embeddings") and len(self.embeddings) > 0:
                # Try to get dimension from existing embeddings
                for existing_emb in self.embeddings:
                    if existing_emb is not None and hasattr(existing_emb, "__len__"):
                        try:
                            existing_array = np.asarray(existing_emb)
                            if not np.any(pd.isna(existing_array)) and existing_array.size > 0:
                                embedding_dim = (
                                    existing_array.shape[0] if existing_array.ndim == 1 else existing_array.shape[-1]
                                )
                                break
                        except:
                            continue

            # Fallback to model default
            if embedding_dim is None:
                # try to get self.embedding_model.embedding_size, if not available, use _get_embedding_dimension
                if hasattr(self.embedding_model, "embedding_size"):
                    embedding_dim = self.embedding_model.embedding_size
                else:
                    from remem.embedding_model.openai_embedding_client import _get_embedding_dimension

                    embedding_dim = _get_embedding_dimension(self.embedding_model.embedding_model_name)
                logger.error(f"Could not determine embedding dimension, using {embedding_dim}")

        # Second pass: replace problematic embeddings with zero vectors and count them
        processed_embeddings = []
        for emb in embeddings:
            needs_replacement = False

            if emb is None:
                needs_replacement = True
                none_count += 1
            else:
                try:
                    emb_array = np.asarray(emb)
                    # Check for various corruption patterns
                    if emb_array.size == 0:
                        needs_replacement = True
                        corrupted_count += 1
                    elif np.any(pd.isna(emb_array)):  # Check for None/NaN values
                        needs_replacement = True
                        corrupted_count += 1
                    elif not np.isfinite(emb_array).all():  # Check for inf values
                        needs_replacement = True
                        corrupted_count += 1
                except:
                    needs_replacement = True
                    corrupted_count += 1

            if needs_replacement:
                processed_embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
            else:
                processed_embeddings.append(emb)

        # Log results
        total_replaced = none_count + corrupted_count
        if total_replaced > 0:
            logger.info(f"Found and replaced {total_replaced} problematic embeddings {context}:")
            logger.info(f"  - {none_count} None embeddings")
            logger.info(f"  - {corrupted_count} corrupted embeddings (containing None/NaN/Inf values)")
            logger.info(f"  - Replacement dimension: {embedding_dim}")

        return processed_embeddings, none_count, corrupted_count, embedding_dim

    def _load_data(self, from_scratch):
        if os.path.exists(self.filename) and from_scratch is False:
            with open(self.filename, "rb") as f:
                data = pickle.load(f)

            self.hash_ids = data["hash_ids"]
            self.texts = data["texts"]
            self.embeddings = data["embeddings"]
            self.metadata = data.get("metadata", None)

            # Validate and fix embeddings
            self.embeddings, none_count, corrupted_count, embedding_dim = self._validate_and_fix_embeddings(
                self.embeddings, context="during load"
            )

            if self.metadata is not None:
                self.hash_id_to_row = {
                    h: {"hash_id": h, "content": t, "metadata": m}
                    for h, t, m in zip(self.hash_ids, self.texts, self.metadata)
                }
            else:
                self.hash_id_to_row = {h: {"hash_id": h, "content": t} for h, t in zip(self.hash_ids, self.texts)}

            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            self.hash_ids, self.texts, self.embeddings, self.metadata = [], [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}

    def _save_data(self):
        data_to_save = {"hash_ids": self.hash_ids, "texts": self.texts, "embeddings": self.embeddings}

        if self.metadata is None or len(self.metadata) == 0:
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t} for h, t, e in zip(self.hash_ids, self.texts, self.embeddings)
            }
        else:
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t, "metadata": m}
                for h, t, e, m in zip(self.hash_ids, self.texts, self.embeddings, self.metadata)
            }
            data_to_save["metadata"] = self.metadata

        with open(self.filename, "wb") as f:
            pickle.dump(data_to_save, f)

        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings, metadata=None):
        # Validate and fix embeddings
        processed_embeddings, none_count, corrupted_count, embedding_dim = self._validate_and_fix_embeddings(
            embeddings, context="during upsert"
        )

        self.embeddings.extend(processed_embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        if metadata is not None:
            self.metadata.extend(metadata)

        logger.info("Saving new records.")
        self._save_data()

    def get_row(self, hash_id):
        return self.hash_id_to_row[hash_id]

    def get_rows(self, hash_ids):
        if not hash_ids:
            return {}

        results = {id: self.hash_id_to_row[id] for id in hash_ids}

        return results

    def get_all_ids(self):
        return self.hash_ids

    def get_text_for_all_rows(self):
        return deepcopy(self.hash_id_to_row)

    def get_hash_id_to_row_readonly(self):
        """
        Get direct reference to hash_id_to_row for read-only access.
        WARNING: Do not modify the returned dictionary or its contents!
        Use get_text_for_all_rows() if you need to modify the data.
        """
        return self.hash_id_to_row

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)

    def get_embeddings(self, hash_ids, dtype=np.float32) -> list[np.ndarray]:
        if not hash_ids:
            return []

        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]

        return embeddings
