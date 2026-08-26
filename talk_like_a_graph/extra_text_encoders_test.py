"""Tests for extra_text_encoders.py."""

import unittest

import networkx as nx

from . import extra_text_encoders
from . import graph_text_encoders


def _line_graph_with_isolate():
  graph = nx.Graph()
  graph.add_nodes_from([0, 1, 2, 3])
  graph.add_edge(0, 1)
  graph.add_edge(1, 2)
  return graph


class ExtraTextEncodersTest(unittest.TestCase):

  def test_node_roster_lists_every_node_including_isolates(self):
    text = extra_text_encoders.encode_graph(
        _line_graph_with_isolate(), 'node_roster'
    )
    self.assertEqual(
        text,
        'G is an undirected graph.\n'
        'The nodes of G are:\n'
        '0\n'
        '1\n'
        '2\n'
        '3\n'
        'The edges of G are: (0, 1), (1, 2).\n',
    )

  def test_incident_lists_isolates_as_none(self):
    text = extra_text_encoders.encode_graph(
        _line_graph_with_isolate(), 'incident'
    )
    self.assertEqual(
        text,
        'G is an undirected graph.\n'
        'Neighborhoods:\n'
        '0: 1\n'
        '1: 0, 2\n'
        '2: 1\n'
        '3: (none)\n',
    )

  def test_incident_does_not_use_the_released_encoder(self):
    graph = _line_graph_with_isolate()
    extra = extra_text_encoders.encode_graph(graph, 'incident')
    released = graph_text_encoders.encode_graph(graph, 'incident')
    self.assertIn('3: (none)', extra)
    self.assertNotIn('Node 3', released)
    self.assertNotEqual(extra, released)

  def test_node_roster_never_writes_the_count(self):
    text = extra_text_encoders.encode_graph(
        _line_graph_with_isolate(), 'node_roster'
    )
    self.assertNotIn('4 nodes', text)
    self.assertNotIn('there are 4', text.lower())

  def test_empty_edges(self):
    graph = nx.Graph()
    graph.add_nodes_from([0, 1])
    text = extra_text_encoders.encode_graph(graph, 'node_roster')
    self.assertIn('The edges of G are: (none).', text)

  def test_adjacency_matrix_still_dispatches(self):
    text = extra_text_encoders.encode_graph(
        _line_graph_with_isolate(), 'adjacency_matrix'
    )
    self.assertIn('The adjacency matrix of G is:', text)

  def test_dimacs_problem_line_counts_isolates(self):
    text = extra_text_encoders.encode_graph(
        _line_graph_with_isolate(), 'dimacs'
    )
    self.assertEqual(
        text,
        'c undirected graph\n'
        'p edge 4 2\n'
        'e 0 1\n'
        'e 1 2\n',
    )
    self.assertNotIn('4 nodes', text)
    self.assertNotIn('e 3', text)


if __name__ == '__main__':
  unittest.main()
