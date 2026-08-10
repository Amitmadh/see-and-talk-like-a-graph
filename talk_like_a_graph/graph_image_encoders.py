"""Library for encoding graphs as images.

This is the visual counterpart of `graph_text_encoders`. For the "See and Talk
Like a Graph" project every GraphQA example is rendered as one or more images so
that a vision-language model can be evaluated in `text`, `image` and
`text+image` settings.

Two properties matter and are enforced here:

1.  Node labels in the image are the *same* labels used by the text encoder.
    They are obtained from `graph_text_encoders.get_tlag_node_encoder`, so an
    `adjacency` encoded graph shows integers while a `friendship` encoded graph
    shows first names. Without this the image and the text would describe two
    different graphs.
2.  Every node and every edge is visible. Node markers are scaled to their
    label, directed graphs are drawn with arrows, and weighted graphs (needed
    for the maximum flow task) show the capacity on each edge.

Note that nothing task specific is drawn: the nodes mentioned in the question
are *not* highlighted and SBM community blocks are *not* colour coded, since
either would leak the answer for the reachability / shortest path / node
classification tasks.
"""

from __future__ import annotations

import math
import os
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import networkx as nx  # pylint: disable=g-import-not-at-top
from PIL import Image  # pylint: disable=g-import-not-at-top

from . import graph_text_encoders  # pylint: disable=g-import-not-at-top


# The layout alternatives we compare in the paper. Each entry maps a graph to a
# dict of node -> (x, y) positions.
LAYOUTS = (
    'spring',
    'kamada_kawai',
    'circular',
    'planar',
)

# Default rendering options. The resulting image is ~768x768 px, which is a
# comfortable size for current VLMs (e.g. Qwen2.5-VL) without being resized down
# so far that the node labels become unreadable.
_FIG_SIZE_INCHES = 6.4
_DPI = 120
_BASE_NODE_SIZE = 700
_FONT_SIZE = 11
_MIN_FONT_SIZE = 7
_EDGE_FONT_SIZE = 9
# Rough width of one character as a fraction of the font size.
_CHAR_WIDTH_RATIO = 0.62
_ANNOTATION_COLOR = '#8a3b1e'
_NODE_COLOR = '#cfe2f3'
_NODE_EDGE_COLOR = '#31537a'
_EDGE_COLOR = '#4a4a4a'


def compute_layout(
    graph: nx.Graph,
    layout: str,
    random_seed: int = 1234,
) -> dict[Any, Any]:
  """Computes node positions for a graph.

  Args:
    graph: the graph to lay out.
    layout: one of `LAYOUTS`.
    random_seed: seed used by the non-deterministic layouts so that the same
      graph always produces the same image.

  Returns:
    A dict mapping each node to its (x, y) position.

  Raises:
    ValueError: if the layout is unknown.
  """
  if layout == 'spring':
    return nx.spring_layout(graph, seed=random_seed)
  elif layout == 'kamada_kawai':
    if graph.number_of_edges() == 0:
      # kamada_kawai is undefined without edges.
      return nx.circular_layout(graph)
    return nx.kamada_kawai_layout(graph)
  elif layout == 'circular':
    return nx.circular_layout(graph)
  elif layout == 'planar':
    try:
      return nx.planar_layout(graph)
    except nx.NetworkXException:
      # Not every generated graph is planar; fall back to a layout that is
      # always defined so the dataset stays rectangular across layouts.
      return nx.spring_layout(graph, seed=random_seed)
  else:
    raise ValueError('Unknown layout: %s' % layout)


def _label_style(labels: Mapping[Any, str]) -> tuple[float, float]:
  """Picks a node marker size and font size that fit the longest label.

  Integer labels are short, but the `friendship` / `got` / `politician`
  encoders use names of up to a dozen characters. Growing the marker alone
  makes nodes overlap, so long labels also shrink the font.

  Args:
    labels: the node -> label mapping to be drawn.

  Returns:
    A `(node_size, font_size)` pair, in matplotlib's points^2 and points.
  """
  if not labels:
    return _BASE_NODE_SIZE, _FONT_SIZE
  longest = max(len(str(label)) for label in labels.values())
  # Shrink the font once labels exceed ~4 characters, but stay legible.
  font_size = min(_FONT_SIZE, max(_MIN_FONT_SIZE, _FONT_SIZE * 4.0 / longest))
  # Approximate rendered text width, plus a little padding on each side.
  text_width = _CHAR_WIDTH_RATIO * longest * font_size + 6.0
  # Marker size is an area in points^2, hence the circle from the diameter.
  node_size = math.pi * (text_width / 2.0) ** 2
  return max(_BASE_NODE_SIZE, node_size), font_size


def _edge_weights(graph: nx.Graph) -> dict[tuple[Any, Any], Any] | None:
  """Returns the edge -> weight map, or None if the graph is unweighted."""
  if not graph.edges():
    return None
  weights = {}
  for source, target, data in graph.edges(data=True):
    if 'weight' not in data:
      return None
    weights[(source, target)] = data['weight']
  return weights


def draw_graph(
    graph: nx.Graph,
    output_path: str,
    encoding_method: str = 'adjacency',
    layout: str = 'kamada_kawai',
    random_seed: int = 1234,
    name_dict: Mapping[Any, str] | None = None,
    node_annotations: Mapping[Any, str] | None = None,
) -> str:
  """Renders a graph to a PNG file.

  Args:
    graph: the graph to draw.
    output_path: where to write the .png file. Parent dirs are created.
    encoding_method: the text encoder the image should agree with. Used to pick
      the node labels, ignored when `name_dict` is given.
    layout: one of `LAYOUTS`.
    random_seed: seed for the layout.
    name_dict: explicit node -> label mapping, overriding `encoding_method`.
    node_annotations: extra text drawn beside particular nodes, e.g. the
      already-known classes in node classification. Only pass information the
      prompt already reveals -- anything else leaks the answer.

  Returns:
    The path the image was written to.
  """
  if name_dict is None:
    name_dict = graph_text_encoders.get_tlag_node_encoder(
        graph, encoding_method
    )
  labels = {node: str(name_dict[node]) for node in graph.nodes()}

  positions = compute_layout(graph, layout, random_seed=random_seed)
  node_size, font_size = _label_style(labels)

  parent = os.path.dirname(output_path)
  if parent:
    os.makedirs(parent, exist_ok=True)

  figure, axes = plt.subplots(
      figsize=(_FIG_SIZE_INCHES, _FIG_SIZE_INCHES), dpi=_DPI
  )
  figure.patch.set_facecolor('white')
  axes.set_facecolor('white')

  nx.draw_networkx_nodes(
      graph,
      positions,
      ax=axes,
      node_size=node_size,
      node_color=_NODE_COLOR,
      edgecolors=_NODE_EDGE_COLOR,
      linewidths=1.5,
  )
  edge_kwargs: dict[str, Any] = {}
  if graph.is_directed():
    # Arrow heads are only drawn for directed graphs; `node_size` shrinks the
    # edge so the head is not hidden under the node marker, and the slight
    # curvature keeps both directions of a mutual edge visible.
    edge_kwargs = {
        'arrows': True,
        'arrowsize': 18,
        'node_size': node_size,
        'connectionstyle': 'arc3,rad=0.08',
    }
  nx.draw_networkx_edges(
      graph,
      positions,
      ax=axes,
      edge_color=_EDGE_COLOR,
      width=1.5,
      **edge_kwargs,
  )
  nx.draw_networkx_labels(
      graph,
      positions,
      labels=labels,
      ax=axes,
      font_size=font_size,
      font_color='black',
  )

  if node_annotations:
    # Placed just outside the node marker so it never covers the node label.
    radius_points = math.sqrt(node_size / math.pi)
    for node, text in node_annotations.items():
      if node not in positions:
        continue
      axes.annotate(
          text,
          xy=positions[node],
          xytext=(0, -(radius_points + 2)),
          textcoords='offset points',
          ha='center',
          va='top',
          fontsize=max(_MIN_FONT_SIZE, font_size - 1),
          color=_ANNOTATION_COLOR,
          bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9),
      )

  weights = _edge_weights(graph)
  if weights is not None:
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=weights,
        ax=axes,
        font_size=_EDGE_FONT_SIZE,
        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85),
    )

  axes.set_axis_off()
  # A small margin keeps labels of the outermost nodes inside the canvas.
  axes.margins(0.12)
  figure.tight_layout(pad=0.3)
  figure.savefig(output_path, format='png', facecolor='white')
  plt.close(figure)

  # matplotlib always writes RGBA. VLM image processors expect 3 channels, so
  # flatten the alpha onto white here rather than relying on the model side.
  with Image.open(output_path) as image:
    if image.mode != 'RGB':
      image.convert('RGB').save(output_path, format='png')
  return output_path


def draw_graph_variants(
    graph: nx.Graph,
    output_dir: str,
    file_stem: str,
    encoding_method: str = 'adjacency',
    layouts: Sequence[str] = LAYOUTS,
    random_seed: int = 1234,
    name_dict: Mapping[Any, str] | None = None,
    node_annotations: Mapping[Any, str] | None = None,
) -> dict[str, str]:
  """Renders one image per layout for the same graph.

  Args:
    graph: the graph to draw.
    output_dir: directory the images are written to.
    file_stem: file name prefix, the layout name is appended to it.
    encoding_method: the text encoder the images should agree with.
    layouts: which layouts to render.
    random_seed: seed for the layouts.
    name_dict: explicit node -> label mapping.

  Returns:
    A dict mapping the layout name to the written image path.
  """
  if name_dict is None:
    name_dict = graph_text_encoders.get_tlag_node_encoder(
        graph, encoding_method
    )
  paths = {}
  for layout in layouts:
    path = os.path.join(output_dir, '%s_%s.png' % (file_stem, layout))
    paths[layout] = draw_graph(
        graph,
        path,
        layout=layout,
        random_seed=random_seed,
        name_dict=name_dict,
        node_annotations=node_annotations,
    )
  return paths
