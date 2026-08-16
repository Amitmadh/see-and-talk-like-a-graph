import copy
import networkx as nx

from talk_like_a_graph.graph_text_encoders import (
    adjacency_encoder,
    nodes_to_text,
)

def encode_corrupted_graph(graph):
    name_dict = nodes_to_text(graph, "integer")
    return adjacency_encoder(graph, name_dict)
# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def corrupt_sample(sample, task, text_encoding):
    """
    Create a corrupted copy of a Sample.

    The image is unchanged.
    The graph is minimally modified.
    The text_encoding is regenerated from the modified graph.
    The answer becomes the correct answer for the modified graph.
    """
    s = copy.deepcopy(sample)

    graph = graph_dict_to_nx(sample.graph)

    corrupted_graph, corrupted_answer, edit = corrupt_by_task(
        graph=graph,
        task=task,
        expected_answer=sample.answer,
        sample=sample,
    )

    corrupted_answer = str(corrupted_answer)
    original_answer = str(sample.answer)

    if corrupted_answer == original_answer:
        raise ValueError(
            f"Corruption did not change answer for "
            f"{sample.sample_id}: {original_answer} -> {corrupted_answer}"
        )

    # Regenerate the textual graph representation from the corrupted graph.
    s.text_encoding = encode_corrupted_graph(
    corrupted_graph
    )

    # Keep the image exactly as it was.
    # sample.images is untouched.

    # Preserve useful provenance in metadata.
    s.metadata = copy.deepcopy(sample.metadata)
    s.metadata["mixed_signals"] = True
    s.metadata["original_answer"] = original_answer
    s.metadata["corrupted_answer"] = corrupted_answer
    s.metadata["corruption"] = edit

    # Store the corrupted graph as well.
    s.graph = nx_to_graph_dict(corrupted_graph)

    # The dataset's answer now corresponds to its text_encoding.
    s.answer = corrupted_answer

    return s


def corrupt_dataset(samples, task, text_encoding):
    """Corrupt every sample and return new Sample objects."""
    return [
        corrupt_sample(
            sample,
            task=task,
            text_encoding=text_encoding,
        )
        for sample in samples
    ]


# ---------------------------------------------------------------------
# Graph conversion
# ---------------------------------------------------------------------

def graph_dict_to_nx(graph_dict):
    G = nx.Graph()

    G.add_nodes_from(graph_dict["nodes"])

    for edge in graph_dict["edges"]:
        G.add_edge(
            edge["source"],
            edge["target"],
        )

    return G


def nx_to_graph_dict(graph):
    return {
        "directed": graph.is_directed(),
        "nodes": list(graph.nodes()),
        "edges": [
            {
                "source": u,
                "target": v,
            }
            for u, v in graph.edges()
        ],
    }


# ---------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------

def corrupt_by_task(graph, task, expected_answer, sample):
    if task == "shortest_path":
        return corrupt_shortest_path(
            graph,
            sample,
            expected_answer,
        )

    if task == "node_degree":
        return corrupt_degree(
            graph,
            sample,
            expected_answer,
        )

    if task == "connected_nodes":
        return corrupt_connectivity(
            graph,
            sample,
            expected_answer,
        )

    if task == "disconnected_nodes":
        return corrupt_disconnected_nodes(
            graph,
            sample,
            expected_answer,
        )

    if task == "node_count":
        return corrupt_node_count(
            graph,
            sample,
            expected_answer,
        )

    if task == "edge_count":
        return corrupt_edge_count(
            graph,
            sample,
            expected_answer,
        )
    if task == "edge_existence":
        return corrupt_edge_existence(
            graph,
            sample,
            expected_answer
        )

    if task == "triangle_counting":
        return corrupt_triangle_counting(
            graph,
            sample,
            expected_answer,
        )

    raise ValueError(f"No corruption implemented for task: {task}")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def query_nodes(sample):
    """
    Return the nodes referred to by the question.

    Our dataset already stores these in graph['node_ids'].
    """
    nodes = sample.node_ids

    # if nodes is None:
    #     nodes = sample.get("node_ids", [])

    if len(nodes) < 1:
        raise ValueError(
            f"No query nodes found for sample {sample.sample_id}"
        )

    return nodes


def add_edge(graph, u, v):
    if u == v:
        raise ValueError("Cannot add self-loop.")

    if graph.has_edge(u, v):
        raise ValueError(f"Edge ({u}, {v}) already exists.")

    graph.add_edge(u, v)


def remove_edge(graph, u, v):
    if not graph.has_edge(u, v):
        raise ValueError(f"Edge ({u}, {v}) does not exist.")

    graph.remove_edge(u, v)


def choose_non_neighbor(graph, node):
    for other in graph.nodes():
        if other != node and not graph.has_edge(node, other):
            return other

    return None


def shortest_distance(graph, source, target):
    try:
        return nx.shortest_path_length(
            graph,
            source,
            target,
        )
    except nx.NetworkXNoPath:
        return None


# ---------------------------------------------------------------------
# shortest_path
# ---------------------------------------------------------------------

def corrupt_shortest_path(graph, sample, expected_answer):
    """
    Change the shortest-path answer using the smallest practical edit.

    Cases:
      - distance == 1:
          remove the direct edge.
          If no alternate path exists, add a 2-edge path.
      - distance > 1:
          add a direct edge, changing the answer to 1.

    The latter is a single-edge edit.
    """
    nodes = query_nodes(sample)

    if len(nodes) < 2:
        raise ValueError(
            f"shortest_path requires two query nodes: "
            f"{sample.sample_id}"
        )

    source, target = nodes[:2]
    original_distance = shortest_distance(
        graph,
        source,
        target,
    )

    if original_distance is None:
        # Already disconnected. Add a direct edge.
        add_edge(graph, source, target)

        return (
            graph,
            1,
            {
                "type": "add_edge",
                "edge": [source, target],
            },
        )

    if original_distance > 1:
        # One edit changes the answer to 1.
        add_edge(graph, source, target)

        return (
            graph,
            1,
            {
                "type": "add_edge",
                "edge": [source, target],
            },
        )

    # Original distance is 1.
    #
    # Remove the direct edge.
    remove_edge(graph, source, target)

    new_distance = shortest_distance(
        graph,
        source,
        target,
    )

    if new_distance is not None and new_distance != original_distance:
        return (
            graph,
            new_distance,
            {
                "type": "remove_edge",
                "edge": [source, target],
            },
        )

    # No path remains. Construct a path of length 2.
    #
    # Find a node that is not already directly connected to both
    # query nodes. This requires two edge additions after the deletion.
    intermediate = None

    for node in graph.nodes():
        if node in (source, target):
            continue

        if (
            not graph.has_edge(source, node)
            and not graph.has_edge(node, target)
        ):
            intermediate = node
            break

    # If no existing node can be used, add a new node.
    if intermediate is None:
        intermediate = max(graph.nodes()) + 1
        graph.add_node(intermediate)

    add_edge(graph, source, intermediate)
    add_edge(graph, intermediate, target)

    new_distance = shortest_distance(
        graph,
        source,
        target,
    )

    if new_distance != 2:
        raise ValueError(
            f"Failed to create shortest path of length 2 for "
            f"{sample.sample_id}"
        )

    return (
        graph,
        2,
        {
            "type": "replace_direct_edge_with_path",
            "removed_edge": [source, target],
            "added_edges": [
                [source, intermediate],
                [intermediate, target],
            ],
        },
    )


# ---------------------------------------------------------------------
# node_degree
# ---------------------------------------------------------------------

def corrupt_degree(graph, sample, expected_answer):
    nodes = query_nodes(sample)

    node = nodes[0]
    original_degree = graph.degree[node]

    if original_degree > 0:
        other = next(iter(graph.neighbors(node)))

        remove_edge(graph, node, other)

        return (
            graph,
            original_degree - 1,
            {
                "type": "remove_edge",
                "edge": [node, other],
            },
        )

    # Isolated node: add one edge.
    other = choose_non_neighbor(graph, node)

    if other is not None:
        add_edge(graph, node, other)

        return (
            graph,
            1,
            {
                "type": "add_edge",
                "edge": [node, other],
            },
        )

    # Degenerate one-node complete graph.
    new_node = max(graph.nodes()) + 1

    graph.add_node(new_node)
    add_edge(graph, node, new_node)

    return (
        graph,
        1,
        {
            "type": "add_node_and_edge",
            "node": new_node,
            "edge": [node, new_node],
        },
    )


# ---------------------------------------------------------------------
# connected_nodes
# ---------------------------------------------------------------------

def corrupt_connectivity(graph, sample, expected_answer):
    nodes = query_nodes(sample)

    if len(nodes) < 1:
        raise ValueError(
            f"connected_nodes requires one query node: "
            f"{sample.sample_id}"
        )

    source = nodes[0]

    neighbors = list(graph.neighbors(source))

    if neighbors:
        # Remove one incident edge.
        target = neighbors[0]
        graph.remove_edge(source, target)

        corrupted_neighbors = sorted(
            graph.neighbors(source)
        )

        corrupted_answer = ", ".join(
            str(n) for n in corrupted_neighbors
        )

        if corrupted_answer:
            corrupted_answer += "."

        return (
            graph,
            corrupted_answer,
            {
                "type": "remove_edge",
                "edge": [source, target],
            },
        )

    # No neighbors: add an edge to another node.
    candidates = [
        n for n in graph.nodes()
        if n != source and not graph.has_edge(source, n)
    ]

    if not candidates:
        raise ValueError(
            f"Cannot add an edge for node {source} "
            f"in sample {sample.sample_id}"
        )

    target = candidates[0]
    graph.add_edge(source, target)

    corrupted_answer = f"{target}."

    return (
        graph,
        corrupted_answer,
        {
            "type": "add_edge",
            "edge": [source, target],
        },
    )


def _disconnected_answer(graph, source):
    disconnected = sorted(
        n for n in graph.nodes()
        if n != source and not graph.has_edge(source, n)
    )
    if not disconnected:
        return "No nodes."
    return ", ".join(str(n) for n in disconnected) + "."


def corrupt_disconnected_nodes(graph, sample, expected_answer):
    graph, _, edit = corrupt_connectivity(
        graph,
        sample,
        expected_answer,
    )
    source = query_nodes(sample)[0]
    return graph, _disconnected_answer(graph, source), edit


# ---------------------------------------------------------------------
# node_count
# ---------------------------------------------------------------------

def corrupt_node_count(graph, sample, expected_answer):
    new_node = max(graph.nodes()) + 1

    graph.add_node(new_node)

    return (
        graph,
        graph.number_of_nodes(),
        {
            "type": "add_node",
            "node": new_node,
        },
    )


# ---------------------------------------------------------------------
# edge_count
# ---------------------------------------------------------------------

def corrupt_edge_count(graph, sample, expected_answer):
    if graph.number_of_edges() > 0:
        u, v = next(iter(graph.edges()))
        remove_edge(graph, u, v)
        return (
            graph,
            graph.number_of_edges(),
            {
                "type": "remove_edge",
                "edge": [u, v],
            },
        )

    nodes = list(graph.nodes())
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            add_edge(graph, u, v)
            return (
                graph,
                graph.number_of_edges(),
                {
                    "type": "add_edge",
                    "edge": [u, v],
                },
            )

    node = nodes[0]
    new_node = max(nodes) + 1
    graph.add_node(new_node)
    add_edge(graph, node, new_node)
    return (
        graph,
        1,
        {
            "type": "add_node_and_edge",
            "node": new_node,
            "edge": [node, new_node],
        },
    )


def corrupt_edge_existence(graph, sample, expected_answer):
    nodes = query_nodes(sample)

    if len(nodes) < 2:
        raise ValueError(
            f"edge_existence requires two query nodes: "
            f"{sample.sample_id}"
        )

    u, v = nodes[:2]

    if graph.has_edge(u, v):
        graph.remove_edge(u, v)

        return (
            graph,
            "false",
            {
                "type": "remove_edge",
                "edge": [u, v],
            },
        )

    else:
        graph.add_edge(u, v)

        return (
            graph,
            "true",
            {
                "type": "add_edge",
                "edge": [u, v],
            },
        )


def _ntriangles(graph):
    return int(sum(nx.triangles(graph).values()) / 3)


def corrupt_triangle_counting(graph, sample, expected_answer):
    if _ntriangles(graph) > 0:
        for u in graph.nodes():
            nbrs = list(graph.neighbors(u))
            for i, v in enumerate(nbrs):
                for w in nbrs[i + 1:]:
                    if graph.has_edge(v, w):
                        remove_edge(graph, v, w)
                        return (
                            graph,
                            _ntriangles(graph),
                            {
                                "type": "remove_edge",
                                "edge": [v, w],
                            },
                        )

    nodes = list(graph.nodes())
    for u in nodes:
        nbrs = list(graph.neighbors(u))
        for i, v in enumerate(nbrs):
            for w in nbrs[i + 1:]:
                if not graph.has_edge(v, w):
                    add_edge(graph, v, w)
                    return (
                        graph,
                        _ntriangles(graph),
                        {
                            "type": "add_edge",
                            "edge": [v, w],
                        },
                    )

    for u, v in list(graph.edges()):
        for w in nodes:
            if w in (u, v):
                continue
            added = []
            if not graph.has_edge(u, w):
                add_edge(graph, u, w)
                added.append([u, w])
            if not graph.has_edge(v, w):
                add_edge(graph, v, w)
                added.append([v, w])
            return (
                graph,
                _ntriangles(graph),
                {
                    "type": "add_edges",
                    "edges": added,
                },
            )

    while len(graph.nodes()) < 3:
        graph.add_node(max(graph.nodes()) + 1)
    a, b, c = list(graph.nodes())[:3]
    added = []
    for u, v in ((a, b), (b, c), (a, c)):
        if not graph.has_edge(u, v):
            add_edge(graph, u, v)
            added.append([u, v])
    return (
        graph,
        _ntriangles(graph),
        {
            "type": "add_edges",
            "edges": added,
        },
    )


import json
from pathlib import Path


def save_corrupted_dataset(samples, output_path):
    """
    Save a derived JSONL dataset.

    The original dataset is never modified.
    """
    output_path = Path(output_path)

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing dataset: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            record = {
                **sample.metadata,
                "sample_id": sample.sample_id,
                "graph": sample.graph,
                "question": sample.question,
                "answer": sample.answer,
                "text_encoding": sample.text_encoding,
                "images": sample.images,
                "node_ids": sample.node_ids,
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )