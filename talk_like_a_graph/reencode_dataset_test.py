"""Tests for reencode_dataset.py."""

import unittest

from . import reencode_dataset


class ReencodeDatasetTest(unittest.TestCase):

  def test_rewrite_sample_id_adjacency(self):
    self.assertEqual(
        reencode_dataset.rewrite_sample_id(
            'node_count/adjacency/test/0', 'node_roster'
        ),
        'node_count/node_roster/test/0',
    )

  def test_rewrite_sample_id_adjacency_matrix(self):
    self.assertEqual(
        reencode_dataset.rewrite_sample_id(
            'cycle_check/adjacency_matrix/test/12', 'incident'
        ),
        'cycle_check/incident/test/12',
    )

  def test_reencode_record_keeps_question_answer_images(self):
    record = {
        'sample_id': 'node_count/adjacency/test/0',
        'question': 'How many nodes are in this graph?',
        'answer': '4',
        'text_encoding': 'old',
        'text_encoder': 'adjacency',
        'extra_context': '',
        'prompt_text': 'oldHow many nodes are in this graph?',
        'images': {'spring': 'images/abc_spring.png'},
        'graph': {
            'directed': False,
            'nodes': [0, 1, 2, 3],
            'edges': [
                {'source': 0, 'target': 1},
                {'source': 1, 'target': 2},
            ],
        },
    }
    out = reencode_dataset.reencode_record(record, 'node_roster')
    self.assertEqual(out['sample_id'], 'node_count/node_roster/test/0')
    self.assertEqual(out['text_encoder'], 'node_roster')
    self.assertEqual(out['question'], record['question'])
    self.assertEqual(out['answer'], record['answer'])
    self.assertEqual(out['images'], record['images'])
    self.assertIn('The nodes of G are:', out['text_encoding'])
    self.assertTrue(out['prompt_text'].startswith(out['text_encoding']))
    self.assertTrue(out['prompt_text'].endswith(record['question']))


if __name__ == '__main__':
  unittest.main()
