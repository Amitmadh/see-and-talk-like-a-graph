"""Text encoders beyond the ones shipped in `graph_text_encoders`.

The released encoders all describe a graph as prose over its *edges* -- even
the one named `adjacency`, which emits an edge list rather than a matrix. This
module adds encodings that the benchmark does not have, without touching
`graph_text_encoders.py`.

`adjacency_matrix` is the obvious missing one: the same graph presented as a
dense table instead of a list. It is interesting for this project because it
changes the *shape* of the textual evidence without changing its content --
the edge list and the matrix are information-equivalent, so any accuracy
difference is about format, not about what the model was told.

Two properties are deliberate:

*   Rows and columns are labelled. Without headers the model has to infer which
    row belongs to which node by counting, and an off-by-one there is
    indistinguishable from a reasoning failure -- exactly the confound the
    project is trying to measure.
*   Weighted graphs put the capacity in the cell. A matrix carries weights
    natively, so unlike the edge-list encoders this one needs no separate
    sentence listing capacities.

Node names come from the same integer mapping `adjacency` uses, so a graph
encoded this way is drawn identically and reuses the cached image.
"""

from __future__ import annotations

from typing import Any, Mapping

import networkx as nx

from . import graph_text_encoders


# Encoders defined here rather than in `graph_text_encoders`.
EXTRA_ENCODERS = ('adjacency_matrix',)

# Which released encoder each extra one borrows its node naming and its
# question phrasing from. The tasks in `graph_tasks` call the released encoder
# directly, so examples are built with the base encoder and only the graph
# description is swapped afterwards. That is sound because the two agree on
# node names, and the gold answer depends on the graph rather than the text.
BASE_ENCODER = {
    'adjacency_matrix': 'adjacency',
}


def base_encoder(encoding_method: str) -> str:
  """The released encoder an extra encoder is built on top of."""
  return BASE_ENCODER.get(encoding_method, encoding_method)


def _cell(graph: nx.Graph, source: Any, target: Any, weighted: bool) -> str:
  """The matrix entry for one ordered pair of nodes."""
  if not graph.has_edge(source, target):
    return '0'
  if weighted:
    return str(graph[source][target].get('weight', 1))
  return '1'


def _is_weighted(graph: nx.Graph) -> bool:
  """True when every edge carries a weight, as in the maximum flow task."""
  if not graph.edges():
    return False
  return all('weight' in data for _, _, data in graph.edges(data=True))


def carries_weights(encoding_method: str) -> bool:
  """Whether this encoder expresses edge weights on its own.

  The edge-list encoders do not, which is why the builders append a separate
  capacity sentence for weighted graphs. The matrix does, so that sentence
  would be redundant.
  """
  return encoding_method == 'adjacency_matrix'


def adjacency_matrix_encoder(
    graph: nx.Graph, name_dict: Mapping[Any, str]
) -> str:
  """Encodes a graph as a labelled adjacency matrix.

  Args:
    graph: the graph to encode.
    name_dict: node -> label mapping, used for the row and column headers.

  Returns:
    The encoded graph, ending in a newline, in the same house style as the
    released encoders: a sentence explaining the notation, a sentence naming
    the nodes, then the data.
  """
  nodes = sorted(graph.nodes())
  labels = [str(name_dict[node]) for node in nodes]
  weighted = _is_weighted(graph)

  if weighted:
    meaning = (
        'entry (i, j) of the adjacency matrix is the capacity of the edge '
        'between node i and node j, and 0 when they are not connected'
    )
  elif graph.is_directed():
    meaning = (
        'entry (i, j) of the adjacency matrix is 1 if there is an edge from '
        'node i to node j, and 0 otherwise'
    )
  else:
    meaning = (
        'entry (i, j) of the adjacency matrix is 1 if node i and node j are '
        'connected with an undirected edge, and 0 otherwise'
    )

  direction = 'directed' if graph.is_directed() else 'undirected'
  if len(labels) > 1:
    node_list = ', '.join(labels[:-1]) + ', and ' + labels[-1]
  else:
    node_list = labels[0] if labels else ''

  rows = [
      [_cell(graph, source, target, weighted) for target in nodes]
      for source in nodes
  ]

  # One width for every column, so the table stays aligned and a value cannot
  # be misread as belonging to its neighbour.
  width = max(
      [len(label) for label in labels]
      + [len(value) for row in rows for value in row]
      + [1]
  )
  gutter = max(len(label) for label in labels) if labels else 1

  lines = [
      'In an %s graph, %s. G describes a graph among nodes %s.'
      % (direction, meaning, node_list),
      'The adjacency matrix of G is:',
      ' ' * gutter + '  ' + ' '.join(label.rjust(width) for label in labels),
  ]
  for label, row in zip(labels, rows):
    lines.append(
        label.ljust(gutter)
        + '  '
        + ' '.join(value.rjust(width) for value in row)
    )
  return '\n'.join(lines) + '\n'


def encode_graph(
    graph: nx.Graph,
    encoding_method: str,
    name_dict: Mapping[Any, str] | None = None,
) -> str:
  """Encodes a graph, dispatching to this module or to the released encoders.

  Args:
    graph: the graph to encode.
    encoding_method: an entry of `EXTRA_ENCODERS`, or any released encoder.
    name_dict: node -> label mapping. Computed when omitted.

  Returns:
    The encoded graph.
  """
  if encoding_method not in EXTRA_ENCODERS:
    return graph_text_encoders.encode_graph(graph, encoding_method)
  if name_dict is None:
    name_dict = get_node_encoder(graph, encoding_method)
  return adjacency_matrix_encoder(graph, name_dict)


def get_node_encoder(graph: nx.Graph, encoding_method: str):
  """Node naming for an extra encoder, or the released mapping otherwise."""
  return graph_text_encoders.get_tlag_node_encoder(
      graph, base_encoder(encoding_method)
  )
