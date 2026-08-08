def get_all_node_properties(graph):
    """Get all node labels (names) and their corresponding vertex IDs."""
    nodes_info = []
    for vertex in graph.vs:
        node_info = {
            "id": vertex.index,
            "name": vertex.attributes().get("name", f"vertex_{vertex.index}"),
            "attributes": dict(vertex.attributes()),
        }
        nodes_info.append(node_info)
    return nodes_info


def get_nodes_by_type(graph):
    """
    The type is the prefix before the first `-` in the name. Return a dictionary
    where keys are types and values are lists of node properties of that type.
    """
    node_types = {}
    for vertex in graph.vs:
        name = vertex.attributes().get("name", f"vertex_{vertex.index}")
        if "-" in name:
            type_name = name.split("-")[0]
            if type_name not in node_types:
                node_types[type_name] = []

            node_info = {"id": vertex.index, "name": name, "attributes": dict(vertex.attributes())}
            node_types[type_name].append(node_info)
    return node_types


def get_all_edge_properties(graph):
    """Get all edge properties/labels and their attributes."""
    edges_info = []
    for edge in graph.es:
        edge_info = {
            "edge_id": edge.index,
            "source_id": edge.source,
            "target_id": edge.target,
            "source_hash_id": graph.vs[edge.source].attributes().get("name", f"vertex_{edge.source}"),
            "target_hash_id": graph.vs[edge.target].attributes().get("name", f"vertex_{edge.target}"),
            "predicate": edge.attributes().get("predicate", "unknown"),
            "qualifiers": edge.attributes().get("qualifiers", "{}"),
            "attributes": dict(edge.attributes()),
        }
        edges_info.append(edge_info)
    return edges_info


def get_node_adjacent_edges(graph, node_name, direction="all", node_types=None):
    """
    Get all edges connected to a given node.

    Args:
        node_name: Either the node name (string) or vertex ID (int)
        direction: 'in', 'out', or 'all' for incoming, outgoing, or both edges
        node_types: List of node types to filter by (e.g., ['verbatim', 'entity']). If None, no filtering is applied.

    Returns:
        List of edge information dictionaries
    """
    # Get vertex ID
    if isinstance(node_name, str):
        try:
            # Try igraph's built-in find method first (much faster for 'name' attribute)
            vertex_id = graph.vs.find(name=node_name).index
        except ValueError:
            # Fallback to manual search for 'content' attribute
            vertex_id = None
            for v in graph.vs:
                if v.attributes().get("content", "") == node_name:
                    vertex_id = v.index
                    break
            if vertex_id is None:
                return []
    else:
        vertex_id = node_name

    # Get edges based on direction
    if direction == "in":
        edges = graph.incident(vertex_id, mode="in")
    elif direction == "out":
        edges = graph.incident(vertex_id, mode="out")
    else:  # 'all'
        edges = graph.incident(vertex_id, mode="all")

    if not edges:
        return []

    # Pre-fetch vertex names to avoid repeated attribute access
    vertices = graph.vs
    vertex_names = {}

    # Convert node_types to set for O(1) lookup if provided
    node_types_set = set(node_types) if node_types else None

    # Format edge information
    edges_info = []
    for edge_id in edges:
        edge = graph.es[edge_id]

        # Get connected node (source or target based on direction)
        connected_node_id = edge.source if edge.source != vertex_id else edge.target

        # Cache vertex names to avoid repeated attribute access
        if connected_node_id not in vertex_names:
            connected_node = vertices[connected_node_id]
            vertex_names[connected_node_id] = connected_node.attributes().get("name", f"vertex_{connected_node_id}")

        connected_node_name = vertex_names[connected_node_id]

        # Filter by node type if specified - optimized with set lookup
        if node_types_set is not None:
            node_type = connected_node_name.split("-", 1)[0] if "-" in connected_node_name else None
            if node_type not in node_types_set:
                continue

        # Cache source and target names
        source_id = edge.source
        target_id = edge.target

        if source_id not in vertex_names:
            vertex_names[source_id] = vertices[source_id].attributes().get("name", f"vertex_{source_id}")
        if target_id not in vertex_names:
            vertex_names[target_id] = vertices[target_id].attributes().get("name", f"vertex_{target_id}")

        edge_info = {
            "edge_id": edge.index,
            "source_id": source_id,
            "target_id": target_id,
            "source_hash_id": vertex_names[source_id],
            "target_hash_id": vertex_names[target_id],
            "direction": "out" if source_id == vertex_id else "in",
        }
        edge_info.update(edge.attributes())  # Include all edge attributes
        edges_info.append(edge_info)

    return edges_info


def get_all_predicate_labels(graph):
    """Get all unique predicates in the graph."""
    predicates = set()
    for edge in graph.es:
        predicate = edge.attributes().get("predicate")
        if predicate:
            predicates.add(predicate)
    return sorted(list(predicates))


def find_edges_by_subject_predicate(graph, subject_name, predicate_value):
    # find the vertex index for the given subject_name
    try:
        vid = graph.vs.find(name=subject_name).index
    except ValueError:
        raise ValueError(f"Subject '{subject_name}' not found in the graph.")

    # select _all_ edges sourced from vid with predicate == predicate_value
    edges = graph.es.select(_source=vid, predicate_eq=predicate_value)

    results = []
    for edge in edges:
        target_vertex = graph.vs[edge.target]
        results.append(
            {
                "edge_id": edge.index,
                "source_id": edge.source,
                "target_id": edge.target,
                "source_hash_id": graph.vs[edge.source].attributes().get("name", f"vertex_{edge.source}"),
                "target_hash_id": target_vertex.attributes().get("name", f"vertex_{edge.target}"),
                "attributes": dict(edge.attributes()),
            }
        )
    return results


def find_edges_by_predicate_object(graph, predicate: str, object_name: str):
    """Find edges by predicate and object name."""
    edges = graph.es.select(predicate_eq=predicate)
    results = []
    for edge in edges:
        target_vertex = graph.vs[edge.target]
        if target_vertex.attributes().get("name") == object_name:
            results.append(
                {
                    "edge_id": edge.index,
                    "source_id": edge.source,
                    "target_id": edge.target,
                    "source_hash_id": graph.vs[edge.source].attributes().get("name", f"vertex_{edge.source}"),
                    "target_hash_id": target_vertex.attributes().get("name", f"vertex_{edge.target}"),
                    "attributes": dict(edge.attributes()),
                }
            )
    return results


def get_node_by_name(graph, node_name):
    """Get node information by name."""
    for vertex in graph.vs:
        if vertex.attributes().get("name") == node_name:
            return {
                "id": vertex.index,
                "name": vertex.attributes().get("name"),
                "attributes": dict(vertex.attributes()),
            }
    return None


def search_nodes_by_attribute(graph, attribute_name, attribute_value):
    """Search nodes by a specific attribute value."""
    matching_nodes = []
    for vertex in graph.vs:
        if vertex.attributes().get(attribute_name) == attribute_value:
            node_info = {
                "id": vertex.index,
                "name": vertex.attributes().get("name", f"vertex_{vertex.index}"),
                "attributes": dict(vertex.attributes()),
            }
            matching_nodes.append(node_info)
    return matching_nodes
