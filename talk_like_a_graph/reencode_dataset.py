"""Rewrite existing GraphQA JSONL files with a different text encoding.

Keeps the same graphs, questions, answers, and image paths so mixed-signals
comparisons stay on identical sample_ids (modulo the encoding segment). Does
not regenerate images.

Example:

```sh
python -m talk_like_a_graph.reencode_dataset \
  --data_dir=data \
  --tasks=node_count,cycle_check \
  --target_encodings=incident,node_roster
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx

from . import extra_text_encoders


def graph_dict_to_nx(graph_dict: dict) -> nx.Graph:
  graph = nx.DiGraph() if graph_dict.get('directed') else nx.Graph()
  graph.add_nodes_from(graph_dict['nodes'])
  for edge in graph_dict['edges']:
    graph.add_edge(edge['source'], edge['target'])
  return graph


def rewrite_sample_id(sample_id: str, target_encoding: str) -> str:
  """task/encoding/split/index -- encoding may itself contain underscores."""
  parts = sample_id.split('/')
  if len(parts) < 4:
    raise ValueError('Unexpected sample_id: %s' % sample_id)
  parts[1] = target_encoding
  return '/'.join(parts)


def find_source(
    data_dir: Path, task: str, preferred: tuple[str, ...]
) -> tuple[Path, str]:
  for encoding in preferred:
    path = data_dir / ('%s_%s_test.jsonl' % (task, encoding))
    if path.is_file():
      return path, encoding
  raise FileNotFoundError(
      'No source JSONL for task %s in %s (tried encodings %s)'
      % (task, data_dir, ', '.join(preferred))
  )


def reencode_record(record: dict, target_encoding: str) -> dict:
  graph = graph_dict_to_nx(record['graph'])
  graph_encoding = extra_text_encoders.encode_graph(graph, target_encoding)
  extra_context = record.get('extra_context') or ''
  question = record['question']
  out = dict(record)
  out['text_encoding'] = graph_encoding
  out['text_encoder'] = target_encoding
  out['sample_id'] = rewrite_sample_id(record['sample_id'], target_encoding)
  if 'prompt_text' in record:
    out['prompt_text'] = graph_encoding + extra_context + question
  return out


def reencode_file(
    source_path: Path,
    target_path: Path,
    target_encoding: str,
) -> int:
  count = 0
  target_path.parent.mkdir(parents=True, exist_ok=True)
  with source_path.open('r', encoding='utf-8') as src, target_path.open(
      'w', encoding='utf-8'
  ) as dst:
    for line in src:
      line = line.strip()
      if not line:
        continue
      record = reencode_record(json.loads(line), target_encoding)
      dst.write(json.dumps(record, ensure_ascii=False) + '\n')
      count += 1
  return count


def main() -> None:
  parser = argparse.ArgumentParser(
      description='Re-encode GraphQA JSONL text without regenerating images.'
  )
  parser.add_argument('--data_dir', default='data')
  parser.add_argument(
      '--tasks',
      default='node_count,cycle_check',
      help='Comma-separated task names.',
  )
  parser.add_argument(
      '--target_encodings',
      default='incident,node_roster',
      help='Comma-separated extra encodings to write.',
  )
  parser.add_argument(
      '--source_encodings',
      default='adjacency,adjacency_matrix',
      help='Preferred source encodings, first match wins.',
  )
  parser.add_argument(
      '--overwrite',
      action='store_true',
      help='Rewrite target JSONLs that already exist.',
  )
  args = parser.parse_args()

  data_dir = Path(args.data_dir)
  tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
  targets = [
      t.strip() for t in args.target_encodings.split(',') if t.strip()
  ]
  preferred = tuple(
      t.strip() for t in args.source_encodings.split(',') if t.strip()
  )

  for encoding in targets:
    if encoding not in extra_text_encoders.EXTRA_ENCODERS:
      raise ValueError(
          'Unknown extra encoding %r. Known: %s'
          % (encoding, extra_text_encoders.EXTRA_ENCODERS)
      )

  for task in tasks:
    source_path, source_encoding = find_source(data_dir, task, preferred)
    for target_encoding in targets:
      if target_encoding == source_encoding:
        print(
            'skip %s: source encoding is already %s'
            % (task, source_encoding)
        )
        continue
      target_path = data_dir / (
          '%s_%s_test.jsonl' % (task, target_encoding)
      )
      if target_path.is_file() and not args.overwrite:
        print('skip existing %s' % target_path)
        continue
      count = reencode_file(source_path, target_path, target_encoding)
      print(
          'wrote %s (%d examples, from %s)'
          % (target_path, count, source_path.name)
      )


if __name__ == '__main__':
  main()
