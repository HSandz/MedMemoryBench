import os
from abc import ABC, abstractmethod

import numpy as np
import torch


class TextRetriever(ABC):
    def __init__(self, corpus):
        self.corpus = corpus

    @abstractmethod
    def _preprocess(self):
        pass

    @abstractmethod
    def scores_on_corpus(self, query):
        pass

    def get_top_k_sentences(self, query, k=10, distinct=True):
        top_k_indices = self.get_top_k_indices(query, k, distinct)
        top_k_sentences = [self.corpus[i] for i in top_k_indices]
        return top_k_sentences

    def get_top_k_indices(self, query, k=10, distinct=True):
        scores = self.scores_on_corpus(query)

        # Get the top k indices with the highest scores
        if distinct is False:
            top_k_indices = np.argsort(scores)[::-1][:k]
            return top_k_indices
        else:
            top_k_indices = np.argsort(scores)[::-1][: 5 * k]

            # Remove duplicates by storing seen sentences in a set
            seen_sentences = set()
            unique_top_k_indices = []
            for i in top_k_indices:
                if self.corpus[i] not in seen_sentences:
                    unique_top_k_indices.append(int(i))
                    seen_sentences.add(self.corpus[i])

                    if len(unique_top_k_indices) == k:
                        break

            return unique_top_k_indices


# from warnings import deprecated
# @deprecated("BM25Retriever is deprecated, use BM25SparseRetriever instead")
# class BM25Retriever(TextRetriever):
#     def __init__(self, corpus, split=' '):
#         super().__init__(corpus)
#         self.bm25 = None
#         self.split_char = split
#         self._preprocess()
#
#     def _preprocess(self):
#         tokenized_corpus = [doc.split(self.split_char) for doc in self.corpus]
#         from rank_bm25 import BM25Okapi
#         self.bm25 = BM25Okapi(tokenized_corpus)
#
#     def scores_on_corpus(self, query):
#         tokenized_query = query.split(" ")
#         scores = self.bm25.get_scores(tokenized_query)
#         return scores


class BM25SparseRetriever(TextRetriever):
    """
    BM25 sparse retriever using the bm25s package (https://github.com/xhluca/bm25s) with automatic index caching.

    Automatically generates index cache paths within the provided working_dir, following
    ReMem's embedding store patterns. Cache is stored at: {working_dir}/bm25_cache/

    Dependencies:
    pip install bm25s

    # Install all extra dependencies
    pip install bm25s[full]

    # If you want to use stemming for better results, you can install a stemmer
    pip install PyStemmer

    # To speed up the top-k selection process, you can install `jax`
    pip install jax[cpu]
    pip install jax[cuda]

    Args:
        corpus: List of documents to index
        index_path: Optional path to save/load BM25 index. If None, auto-generates within working_dir
        stopwords: Language for stopwords removal (default: 'en')
        use_stemmer: Whether to use stemming (default: True)
        stemmer_lang: Language for stemming (default: 'english')
        working_dir: Working directory for cache storage, following ReMem patterns (optional)
        global_config: ReMem global config object for working_dir detection (optional)
    """

    def __init__(
        self,
        corpus,
        index_path=None,
        stopwords="en",
        use_stemmer=True,
        stemmer_lang="english",
        working_dir=None,
        global_config=None,
    ):
        super().__init__(corpus)

        self.stemmer = None
        self.stopwords = stopwords
        if use_stemmer:
            import Stemmer

            self.stemmer = Stemmer.Stemmer(stemmer_lang)
        self.corpus_tokens = None

        # Auto-generate index path if not provided, following ReMem working_dir patterns
        if index_path is None:
            import os

            # Use working_dir if provided, otherwise try to infer from global_config
            if working_dir is None and global_config is not None:
                # Try to construct working_dir similar to ReMem patterns
                if hasattr(global_config, "save_dir"):
                    working_dir = global_config.save_dir
                else:
                    working_dir = "outputs/default"
            elif working_dir is None:
                # Fallback to simple default
                working_dir = "outputs/default"

            # Create BM25 cache directory within working_dir, following the same pattern as embedding stores
            bm25_cache_dir = os.path.join(working_dir, "bm25_cache")

            # Create a unique index name based on corpus characteristics
            import hashlib

            corpus_size = len(corpus)

            # Create a simple hash of the corpus data for cache uniqueness
            # Use first 1000 chars of joined corpus to balance speed vs uniqueness
            corpus_sample = "".join(corpus[: min(10, len(corpus))])[:1000]
            corpus_hash = hashlib.md5(corpus_sample.encode("utf-8")).hexdigest()[:8]

            stemmer_suffix = "_stemmed" if use_stemmer else "_nostem"
            index_path = os.path.join(bm25_cache_dir, f"bm25_index_{corpus_size}_{corpus_hash}{stemmer_suffix}")

        self.index_path = index_path

        # Create directory if it doesn't exist
        os.makedirs(self.index_path, exist_ok=True)

        # Check if index already exists and can be loaded
        if os.path.exists(self.index_path) and len(os.listdir(self.index_path)) > 0:
            self.retriever = self.load(self.index_path)
            if self.retriever is None:
                print(f"Failed to load index from {self.index_path}, rerun the preprocessing")
                self._preprocess()
            else:
                print(f"Loaded existing BM25 index from {self.index_path}")
        else:
            print(f"Creating new BM25 index at {self.index_path}")
            self._preprocess()
        assert self.retriever is not None, "retriever is None"
        assert self.retriever.corpus is not None, "corpus is None"

    def _preprocess(self):
        # Tokenize the corpus and keep only ids (optimized for speed and memory)
        import bm25s

        self.retriever = bm25s.BM25(corpus=self.corpus)
        assert self.corpus is not None and len(self.corpus) > 0, "Corpus is empty"
        # Suppress progress bars by setting show_progress=False
        self.corpus_tokens = bm25s.tokenize(
            self.corpus, stopwords=self.stopwords, stemmer=self.stemmer, show_progress=False
        )
        self.retriever.index(self.corpus_tokens)
        # self.corpus = self.retriever.corpus
        self.save(self.index_path)

    def get_top_k_indices(self, query, k=10, distinct=True, return_scores=False):
        import bm25s

        try:
            query_tokens = bm25s.tokenize(query, stemmer=self.stemmer, show_progress=False)
            if len(query_tokens.ids[0]) == 0:
                if return_scores:
                    return [], []
                return []
            results, scores = self.retriever.retrieve(query_tokens, k=k)
        except Exception as e:
            print("get top-k indices exception", e)
            print("query:", query)
            print("#corpus:", len(self.corpus))
            exit(1)

        # for each result, get the index of the passage
        indices = []
        res_score = []
        for idx, result in enumerate(results[0]):
            try:
                corpus_idx = self.retriever.corpus.index(result)
            except Exception as e:
                print("BM25 sparse retriever: passage not found in corpus", e)
                exit(1)
            indices.append(corpus_idx)
            res_score.append(scores[0][idx])

        if return_scores:
            return indices, res_score
        return indices

    def get_top_k_sentences(self, query, k=10, distinct=True, return_scores=False):
        import bm25s

        try:
            query_tokens = bm25s.tokenize(query, stemmer=self.stemmer, show_progress=False)
            if len(query_tokens.ids[0]) == 0:
                if return_scores:
                    return [], []
                return []
            results, scores = self.retriever.retrieve(query_tokens, k=k)
            if return_scores:
                return results, scores
            return results
        except Exception as e:
            print("get top-k indices exception", e)
            print("query:", query)
            print("#corpus:", len(self.corpus))

    def scores_on_corpus(self, query):
        print("BM25SparseRetriever does not support scores_on_corpus method. Use get_top_k_indices instead.")
        pass

    def save(self, index_path, save_corpus=True):
        if index_path is not None:
            if save_corpus:
                assert self.corpus is not None, "Corpus is None"
            self.retriever.save(index_path, corpus=self.corpus if save_corpus else None)
            print(f"Index saved to {index_path}")

    @staticmethod
    def load(index_path, load_corpus=True):
        if not os.path.exists(index_path):
            print(f"Index path {index_path} does not exist")
            return None

        try:
            import bm25s

            retriever = bm25s.BM25.load(index_path, load_corpus=load_corpus)
            return retriever
        except Exception as e:
            print("Loading retriever exception", e)
            print("Index path:", index_path)
            return None


class TfidfRetriever(TextRetriever):
    def __init__(self, corpus):
        super().__init__(corpus)
        self.vectorizer = None
        self.tfidf_matrix = None
        self._preprocess()

    def _preprocess(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def scores_on_corpus(self, query):
        query_vec = self.vectorizer.transform([query])
        scores = np.dot(self.tfidf_matrix, query_vec.T).toarray().flatten()
        return scores


class SentenceTransformerRetriever(TextRetriever):
    def __init__(self, corpus, model_name=None, model=None):
        super().__init__(corpus)
        from sentence_transformers import SentenceTransformer

        assert model_name is not None or model is not None, "Either model_name or model should be provided"
        if model is None:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = model
        self.embeddings = None
        self._preprocess()

    def _preprocess(self):
        with torch.no_grad():
            self.embeddings = self.model.encode(self.corpus)

    def scores_on_corpus(self, query):
        from sentence_transformers import util

        with torch.no_grad():
            query_embedding = self.model.encode([query])[0]
            scores = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
            return scores.cpu().numpy()


class GritLMRetriever(TextRetriever):

    def __init__(self, corpus, model_name="GritLM/GritLM-7B", model=None, instruction=""):
        from gritlm import GritLM

        super().__init__(corpus)
        assert model_name is not None or model is not None, "Either model_name or model should be provided"
        if model is None:
            self.model = GritLM(model_name, torch_dtype="auto")
        else:
            self.model = model
        self.instruction = instruction
        self.embeddings = None
        self._preprocess()

    def gritlm_instruction(self, instruction):
        return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

    def _preprocess(self):
        with torch.no_grad():
            self.embeddings = self.model.encode(self.corpus, instruction=self.gritlm_instruction(""))

    def scores_on_corpus(self, query):
        from sentence_transformers import util

        if isinstance(query, str):
            query = [query]
        with torch.no_grad():
            query_embedding = self.model.encode(query, instruction=self.gritlm_instruction(self.instruction))[0]
            scores = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
            return scores.cpu().numpy()


class DPR(TextRetriever):
    def __init__(
        self,
        corpus,
        passage_encoder="facebook-dpr-ctx_encoder-single-nq-base",
        query_encoder="facebook-dpr-question_encoder-single-nq-base",
    ):
        super().__init__(corpus)
        # Initialize the encoders
        from sentence_transformers import SentenceTransformer

        self.passage_encoder = SentenceTransformer(passage_encoder)
        self.query_encoder = SentenceTransformer(query_encoder)
        self.passage_embeddings = self._preprocess()

    def _preprocess(self):
        res = self.passage_encoder.encode(self.corpus)
        print("Passage embedded")
        return res

    def scores_on_corpus(self, query):
        from sentence_transformers import util

        query_embedding = self.query_encoder.encode(query)
        scores = util.dot_score(query_embedding, self.passage_embeddings)[0]
        return scores.cpu().numpy()


class CrossEncoderRetrieval(TextRetriever):
    def __init__(self, corpus, model_name_or_path="cross-encoder/nli-deberta-v3-base"):
        super().__init__(corpus)
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name_or_path)

    def _preprocess(self):
        pass

    def scores_on_corpus(self, query):
        sentence_combinations = [[query, corpus_sentence] for corpus_sentence in self.corpus]
        scores = self.model.predict(sentence_combinations)[:, 1]
        return scores


class Colbertv2Retrieval(TextRetriever):
    def __init__(self, corpus: list, root: str, index_name: str):
        self.root = root
        self.index_name = index_name
        self.corpus = corpus
        self.checkpoint_path = "exp/colbertv2.0"

        from colbert import Searcher
        from colbert.infra import ColBERTConfig, Run, RunConfig

        self._preprocess()

        with Run().context(RunConfig(nranks=1, experiment="colbert", root=self.root)):
            config = ColBERTConfig(
                root=self.root.rstrip("/") + "/colbert",
            )
            self.searcher = Searcher(index=self.index_name, config=config)

    def _preprocess(self, overwrite="reuse"):
        """
        :param overwrite: one value from [True, 'reuse', 'resume', "force_silent_overwrite"]
        """
        from colbert.infra import ColBERTConfig, Run, RunConfig

        with Run().context(RunConfig(nranks=1, experiment="colbert", root=self.root)):
            config = ColBERTConfig(
                nbits=2,
                root=self.root,
            )
            from colbert import Indexer

            indexer = Indexer(checkpoint=self.checkpoint_path, config=config)
            indexer.index(name=self.index_name, collection=self.corpus, overwrite=overwrite)

    def scores_on_corpus(self, query):
        pass

    def get_top_k_sentences(self, query, k=100, distinct=True):
        from colbert.data import Queries

        query = Queries(path=None, data={0: query})
        ranking = self.searcher.search_all(query, k)
        res = []
        for item in list(ranking.data.values())[0]:
            res.append(self.corpus[item[0]])
        return res

    def get_top_k_indices(self, query, k=100, distinct=True):
        from colbert.data import Queries

        query = Queries(path=None, data={0: query})
        ranking = self.searcher.search_all(query, k)
        res = []
        for item in list(ranking.data.values())[0]:
            res.append(item[0])
        return res
