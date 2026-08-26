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

Two properties of the matrix are deliberate:

*   Rows and columns are labelled. Without headers the model has to infer which
    row belongs to which node by counting, and an off-by-one there is
    indistinguishable from a reasoning failure -- exactly the confound the
    project is trying to measure.
*   Weighted graphs put the capacity in the cell. A matrix carries weights
    natively, so unlike the edge-list encoders this one needs no separate
    sentence listing capacities.

`incident` and `node_roster` are count-oriented: they put one record per node
on its own line, including isolated nodes. Mixed-signals results with the
matrix showed that burying a node list in a prose sentence ("among nodes 0, 1,
...") is not enough to make node-count follow text -- the model still prefers
the image. These two encodings make *counting records* the obvious action,
without ever writing the gold count ("G has 17 nodes").

The extra `incident` is not the released `incident_encoder`. The released one
skips isolated nodes (it only emits "Node X is connected to ..." for nodes
with neighbours) and still opens with the buried prose list. Ours lists every
node as `id: neighbours` or `id: (none)`.

`dimacs` is a graph *file* format, not an English description. The first data
line is `p edge n m` (node count, then edge count), then one `e u v` per edge.
Isolates are counted in `n` but do not appear as `e` lines. That puts `|V|` in
the text as a numeral -- the lever line-counting encodings did not have --
without a sentence like "G has 17 nodes." Node ids stay 0-based so they match
the image labels; classic DIMACS is often 1-based.

Node names come from the same integer mapping `adjacency` uses, so a graph
encoded this way is drawn identically and reuses the cached image.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import networkx as nx

from . import graph_text_encoders


# Encoders defined here rather than in `graph_text_encoders`.
# `incident` here overrides the released encoder when this module dispatches.
EXTRA_ENCODERS = ('adjacency_matrix', 'incident', 'node_roster', 'dimacs')

# Which released encoder each extra one borrows its node naming and its
# question phrasing from. The tasks in `graph_tasks` call the released encoder
# directly, so examples are built with the base encoder and only the graph
# description is swapped afterwards. That is sound because the two agree on
# node names, and the gold answer depends on the graph rather than the text.
BASE_ENCODER = {
    'adjacency_matrix': 'adjacency',
    'incident': 'adjacency',
    'node_roster': 'adjacency',
    'dimacs': 'adjacency',
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


def incident_encoder(
    graph: nx.Graph, name_dict: Mapping[Any, str]
) -> str:
  """One neighbourhood line per node, including isolates.

  Isolated nodes are written as `id: (none)` so they still occupy a record
  that can be counted. The released `incident_encoder` omits them.
  """
  direction = 'directed' if graph.is_directed() else 'undirected'
  lines = [
      'G is an %s graph.' % direction,
      'Neighborhoods:',
  ]
  for node in sorted(graph.nodes()):
    neighbours = sorted(graph.neighbors(node))
    if neighbours:
      neighbour_str = ', '.join(str(name_dict[n]) for n in neighbours)
    else:
      neighbour_str = '(none)'
    lines.append('%s: %s' % (name_dict[node], neighbour_str))
  return '\n'.join(lines) + '\n'


def node_roster_encoder(
    graph: nx.Graph, name_dict: Mapping[Any, str]
) -> str:
  """A vertical node roster, then the edge list.

  Never writes the node count in words. Isolated nodes still appear, each on
  its own line, so counting nodes is counting lines under the roster heading.
  """
  direction = 'directed' if graph.is_directed() else 'undirected'
  lines = [
      'G is an %s graph.' % direction,
      'The nodes of G are:',
  ]
  for node in sorted(graph.nodes()):
    lines.append(str(name_dict[node]))

  if graph.edges():
    pairs = []
    for source, target in graph.edges():
      if not graph.is_directed() and source > target:
        source, target = target, source
      pairs.append((source, target))
    pairs.sort()
    edge_parts = [
        '(%s, %s)' % (name_dict[u], name_dict[v]) for u, v in pairs
    ]
    lines.append('The edges of G are: ' + ', '.join(edge_parts) + '.')
  else:
    lines.append('The edges of G are: (none).')
  return '\n'.join(lines) + '\n'


def dimacs_encoder(
    graph: nx.Graph, name_dict: Mapping[Any, str]
) -> str:
  """DIMACS `p edge n m` format, with 0-based ids matching the image.

  Isolated nodes are included in `n` but have no `e` line. Does not write an
  English node-count sentence; the problem line *is* the count.
  """
  n = graph.number_of_nodes()
  pairs = []
  for source, target in graph.edges():
    if not graph.is_directed() and source > target:
      source, target = target, source
    pairs.append((source, target))
  pairs.sort()
  direction = 'directed' if graph.is_directed() else 'undirected'
  lines = [
      'c %s graph' % direction,
      'p edge %d %d' % (n, len(pairs)),
  ]
  for source, target in pairs:
    lines.append('e %s %s' % (name_dict[source], name_dict[target]))
  return '\n'.join(lines) + '\n'


_ENCODERS: dict[str, Callable[[nx.Graph, Mapping[Any, str]], str]] = {
    'adjacency_matrix': adjacency_matrix_encoder,
    'incident': incident_encoder,
    'node_roster': node_roster_encoder,
    'dimacs': dimacs_encoder,
}


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
  return _ENCODERS[encoding_method](graph, name_dict)


def get_node_encoder(graph: nx.Graph, encoding_method: str):
  """Node naming for an extra encoder, or the released mapping otherwise."""
  return graph_text_encoders.get_tlag_node_encoder(
      graph, base_encoder(encoding_method)
  )
