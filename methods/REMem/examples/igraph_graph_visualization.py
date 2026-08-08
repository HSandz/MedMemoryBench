import argparse
import pickle
import random

import igraph as ig
from igraph import plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="musique", help="Dataset name")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini", help="LLM name")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2", help="embedding model name")
    parser.add_argument(
        "--show_names",
        action="store_true",
        default=False,
        help="Whether to show the original names of the nodes and edges.",
    )
    args = parser.parse_args()

    llm_label = args.llm_name.replace("/", "_")
    embedding_label = args.embedding_name.replace("/", "_")

    # Load the GraphML file
    G = ig.Graph.Read_GraphML(
        f"outputs/{args.dataset}/{args.dataset}_{llm_label}_{embedding_label}/KnowledgeGraph/graph_elements/graph_graph.graphml"
    )

    # Assign random colors to nodes and optionally set titles
    node_colors = ["#{:06x}".format(random.randint(0, 0xFFFFFF)) for _ in range(G.vcount())]
    G.vs["color"] = node_colors

    if args.show_names:

        with open(
            f"outputs/{args.dataset}/{args.dataset}_{llm_label}_{embedding_label}/KnowledgeGraph/graph_elements/kv_store_nodes.pkl",
            "rb",
        ) as f:
            nodes_storage_data = pickle.load(f)

        with open(
            f"outputs/{args.dataset}/{args.dataset}_{llm_label}_{embedding_label}/KnowledgeGraph/graph_elements/kv_store_facts.pkl",
            "rb",
        ) as f:
            facts_storage_data = pickle.load(f)

        # Build map from node key to original name, and map from edge key to original name
        node_key2name = {node_key: node_entry["content"] for node_key, node_entry in nodes_storage_data.items()}
        head_tail_names2relation_name = {}
        for fact_key, fact_entry in facts_storage_data.items():
            head_name = fact_entry["head"]
            tail_name = fact_entry["tail"]
            if (head_name, tail_name) not in head_tail_names2relation_name:
                head_tail_names2relation_name[(head_name, tail_name)] = []
            head_tail_names2relation_name[(head_name, tail_name)].append(fact_entry["relation"])
        for head_name, tail_name in head_tail_names2relation_name:
            if len(head_tail_names2relation_name[(head_name, tail_name)]) == 1:
                head_tail_names2relation_name[(head_name, tail_name)] = head_tail_names2relation_name[
                    (head_name, tail_name)
                ][0]
            else:
                head_tail_names2relation_name[(head_name, tail_name)] = str(
                    list(set(head_tail_names2relation_name[(head_name, tail_name)]))
                )

    if "name" in G.es.attributes():
        G.es["name"] = G.es["name"]
    else:
        # G.es["name"] = ["Edge {}".format(i) for i in range(G.ecount())]
        if not args.show_names:
            G.es["name"] = ["Edge {}".format(i) for i in range(G.ecount())]
        else:
            for e in G.es:
                head_node_name = node_key2name[G.vs[e.source]["name"]]
                tail_node_name = node_key2name[G.vs[e.target]["name"]]
                if (head_node_name, tail_node_name) in head_tail_names2relation_name:
                    e["name"] = head_tail_names2relation_name[(head_node_name, tail_node_name)]
                else:
                    e["name"] = (
                        f"~ {head_tail_names2relation_name[(head_node_name, tail_node_name)]}"  # use "~" to represent reversed relation
                    )

    if "name" in G.vs.attributes():
        if not args.show_names:
            G.vs["name"] = G.vs["name"]
        else:
            for v in G.vs:
                v["name"] = node_key2name[v["name"]]
    else:
        G.vs["name"] = ["Node {}".format(i) for i in range(G.vcount())]

    # Visualize the graph
    visual_style = {
        "vertex_size": 20,
        "vertex_color": G.vs["color"],
        "vertex_label": G.vs["name"],
        "edge_label": G.es["name"],
        "layout": G.layout("fr"),  # Fruchterman-Reingold layout
        "bbox": (800, 800),
        "margin": 50,
    }

    # Save and display the graph
    plot(G, "knowledge_graph.png", **visual_style)
