"""Builds the multimodal GraphQA dataset for "See and Talk Like a Graph".

The original `graph_tasks_generator` writes `tf.train.Example` protos into
recordio files using Google-internal libraries (`recordio`, `seqio`, `gfile`,
`tfgnn`) that are not available outside Google. This module reuses the parts of
the pipeline that *are* portable -- `graph_generators`, `graph_tasks` and
`graph_text_encoders` -- and emits a plain JSONL + PNG dataset instead, so it
can be fed directly to a vision-language model.

Every record is a tuple of

    (sample_id, graph, text encoding, question encoding, image(s), answer)

plus the metadata needed for the later ablation analysis (task, generator
algorithm, text encoder, node/edge counts, node ids referenced by the
question). Each graph is rendered under several layouts, which lets us compare
visualisation alternatives without regenerating the textual half.

The three evaluation settings of the paper map onto the record fields as:

    text-only   -> record["text_encoding"] + record["extra_context"]
                   + record["question"]      (i.e. record["prompt_text"])
    image-only  -> record["images"][layout] + record["extra_context"]
                   + record["question"]
    text+image  -> record["prompt_text"] + record["images"][layout]

`extra_context` is empty for every task except `node_classification`, where the
already-known node labels are part of the prompt and cannot be read off the
image.

By default only the 7 tasks of the published GraphQA benchmark are built, so
the results stay comparable to Table 1 of "Talk like a Graph". The released
code implements 5 further tasks that the paper does not evaluate; see
`EXTRA_TASKS`.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any, Iterator, Sequence

import networkx as nx
import numpy as np

from . import graph_generators
from . import graph_image_encoders
from . import graph_tasks
from . import graph_text_encoders


# The GraphQA benchmark as defined in "Talk like a Graph", Appendix A.2. These
# are the tasks the paper reports on, so they are the only ones whose numbers
# can be compared against its Table 1. The first six are used in the paper's
# Experiments 1-4; `disconnected_nodes` is its Experiment 5.
GRAPHQA_TASKS = {
    'edge_existence': graph_tasks.EdgeExistence,
    'node_degree': graph_tasks.NodeDegree,
    'node_count': graph_tasks.NodeCount,
    'edge_count': graph_tasks.EdgeCount,
    'connected_nodes': graph_tasks.ConnectedNodes,
    'cycle_check': graph_tasks.CycleCheck,
    'disconnected_nodes': graph_tasks.DisconnectedNodes,
}

# Harder tasks implemented in the released code but *not* part of the GraphQA
# benchmark, and not evaluated in the paper: `shortest_path` appears there only
# as a motivating example of a multi-hop task, `node_classification` only in
# related work, and the rest not at all. They are opt-in via `--tasks` because
# they have no published baseline to compare against -- but they are also the
# most plausible place for a graph *image* to help, so they are worth running
# as a secondary experiment.
EXTRA_TASKS = {
    'reachability': graph_tasks.Reachability,
    'shortest_path': graph_tasks.ShortestPath,
    'maximum_flow': graph_tasks.MaximumFlow,
    'node_classification': graph_tasks.NodeClassification,
    'triangle_counting': graph_tasks.TriangleCounting,
}

# Everything the generator can build.
TASKS = {**GRAPHQA_TASKS, **EXTRA_TASKS}

# The generator algorithms used by the GraphQA benchmark.
ALGORITHMS = ('er', 'ba', 'sbm', 'sfn', 'complete', 'star', 'path')

# Seeds per split, matching `graph_generators_runner`.
_SPLIT_SEEDS = {'train': 9876, 'test': 1234, 'validation': 5432}

_MAX_NNODES = 20


def _split_seed(split: str) -> int:
  if split not in _SPLIT_SEEDS:
    raise ValueError('Unknown split: %s' % split)
  return _SPLIT_SEEDS[split]


def generate_random_sbm_graph(random_state: np.random.RandomState) -> nx.Graph:
  """An SBM graph that keeps its `block` node attribute.

  The node classification task needs the community each node belongs to. The
  standard generators strip that attribute (GraphML cannot store it), so those
  graphs are regenerated here. This mirrors the helper of the same name in
  `graph_tasks_generator`.

  Args:
    random_state: numpy random state driving the generator.

  Returns:
    A stochastic block model graph with a `block` attribute on every node.
  """
  small_number = random.uniform(0, 0.05)
  large_number = random.uniform(0.6, 0.8)
  number_of_nodes = random.choice(np.arange(5, 20))
  sizes = [number_of_nodes // 2, number_of_nodes // 2]
  probs = [[large_number, small_number], [small_number, large_number]]
  return nx.stochastic_block_model(sizes, probs, seed=random_state)


def load_graphs(
    graphs_dir: str,
    algorithm: str,
    split: str,
    direction: str,
    max_nnodes: int = _MAX_NNODES,
) -> list[nx.Graph]:
  """Loads `.graphml` graphs written by `graph_generators_runner`.

  This is the portable counterpart of `graph_tasks_utils.load_graphs`, which
  relies on internal file APIs.

  Args:
    graphs_dir: root directory holding `<direction>/<algorithm>/<split>`.
    algorithm: the generator algorithm subdirectory.
    split: the split subdirectory.
    direction: either `directed` or `undirected`.
    max_nnodes: graphs larger than this are skipped.

  Returns:
    The loaded graphs, ordered by file name.
  """
  path = os.path.join(graphs_dir, direction, algorithm, split)
  if not os.path.isdir(path):
    return []
  graphs = []
  for file_name in sorted(os.listdir(path)):
    if not file_name.endswith('.graphml'):
      continue
    graph = nx.read_graphml(os.path.join(path, file_name), node_type=int)
    if graph.number_of_nodes() <= max_nnodes:
      graphs.append(graph)
  return graphs


def build_graphs(
    algorithms: Sequence[str],
    split: str,
    number_of_graphs: int,
    directed: bool = False,
    graphs_dir: str | None = None,
) -> tuple[list[nx.Graph], list[str]]:
  """Collects the benchmark graphs and the algorithm that produced each one.

  Args:
    algorithms: the generator algorithms to include.
    split: `train`, `test` or `validation`.
    number_of_graphs: how many graphs to generate per algorithm. Ignored when
      loading from `graphs_dir`.
    directed: whether to build directed graphs.
    graphs_dir: if given, graphs are read from disk instead of generated.

  Returns:
    A `(graphs, generator_algorithms)` pair of parallel lists.
  """
  direction = 'directed' if directed else 'undirected'
  graphs: list[nx.Graph] = []
  generator_algorithms: list[str] = []
  for algorithm in algorithms:
    if graphs_dir:
      loaded = load_graphs(graphs_dir, algorithm, split, direction)
    else:
      loaded = graph_generators.generate_graphs(
          number_of_graphs=number_of_graphs,
          algorithm=algorithm,
          directed=directed,
          random_seed=_split_seed(split),
      )
      loaded = [g for g in loaded if g.number_of_nodes() <= _MAX_NNODES]
    graphs += loaded
    generator_algorithms += [algorithm] * len(loaded)
  return graphs, generator_algorithms


def serialize_graph(graph: nx.Graph) -> dict[str, Any]:
  """Turns a graph into a JSON-serialisable dict."""
  edges = []
  for source, target, data in graph.edges(data=True):
    edge: dict[str, Any] = {'source': source, 'target': target}
    if 'weight' in data:
      edge['weight'] = int(data['weight'])
    edges.append(edge)
  return {
      'directed': graph.is_directed(),
      'nodes': sorted(graph.nodes()),
      'edges': edges,
  }


def graph_digest(graph: nx.Graph, labels: dict[Any, str]) -> str:
  """A stable id for a (graph, node labelling) pair.

  Used to name and deduplicate images: the same graph drawn with the same node
  labels produces the same file, so it is rendered once and shared by every
  task that uses it.

  Args:
    graph: the graph.
    labels: the node -> label mapping shown in the image.

  Returns:
    A 16 character hex digest.
  """
  payload = json.dumps(
      {
          'graph': serialize_graph(graph),
          'labels': {str(k): v for k, v in sorted(labels.items())},
      },
      sort_keys=True,
  )
  return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _strip_question(question: str, task_description: str) -> str:
  """Returns the prompt with its trailing question removed.

  `graph_tasks` returns a single prompt string. We need the two halves apart so
  that the image-only setting can drop the graph description while keeping the
  question.

  Args:
    question: the full prompt produced by the task.
    task_description: the trailing question, as reported by the task.

  Returns:
    Everything preceding the question: the graph encoding plus, for node
    classification, the already-known node labels.

  Raises:
    ValueError: if the prompt does not end with the task description.
  """
  if not question.endswith(task_description):
    raise ValueError('Prompt does not end with its task description.')
  return question[: len(question) - len(task_description)]


def build_records(
    task_name: str,
    graphs: Sequence[nx.Graph],
    generator_algorithms: Sequence[str],
    encoding_method: str,
    split: str,
    images_dir: str,
    images_rel_dir: str = 'images',
    layouts: Sequence[str] = graph_image_encoders.LAYOUTS,
    random_seed: int = 1234,
    render_images: bool = True,
) -> Iterator[dict[str, Any]]:
  """Yields one dataset record per graph for a single task and text encoder.

  Args:
    task_name: a key of `TASKS`.
    graphs: the graphs to build examples from.
    generator_algorithms: the algorithm behind each graph, same length as
      `graphs`.
    encoding_method: the text encoder, e.g. `adjacency`.
    split: the split name, stored on the record.
    images_dir: absolute directory the PNGs are written to.
    images_rel_dir: how the image paths are written into the record, relative
      to the dataset root.
    layouts: which image layouts to render.
    random_seed: seed for example construction and layouts.
    render_images: set to False to emit the textual half only.

  Yields:
    Dataset records as plain dicts.
  """
  if task_name not in TASKS:
    raise ValueError('Unknown task: %s' % task_name)
  task = TASKS[task_name]()

  # Deep copy, because some tasks mutate the graphs they are given:
  # `MaximumFlow` calls `graph_tasks.add_edge_weight`, which assigns random
  # capacities in place. Passing `list(graphs)` is not enough -- that is a new
  # list of the *same* graph objects, so the weights would leak into every task
  # built afterwards and show up in their images.
  task_graphs = [graph.copy() for graph in graphs]

  # The tasks sample source/target nodes with the global `random` module.
  random.seed(random_seed)
  examples_dict = task.prepare_examples_dict(
      task_graphs, list(generator_algorithms), encoding_method
  )

  for index in sorted(examples_dict.keys()):
    value = examples_dict[index]
    graph = value['graph']
    text_encoding = _strip_question(
        value['question'], value['task_description']
    )
    # `graph_tasks` glues the graph encoding and any extra prompt context
    # together; recompute the encoding to recover the boundary between them.
    graph_encoding = graph_text_encoders.encode_graph(graph, encoding_method)
    if text_encoding.startswith(graph_encoding):
      extra_context = text_encoding[len(graph_encoding) :]
    else:
      # Should not happen, but never lose prompt content if it does.
      graph_encoding, extra_context = text_encoding, ''

    name_dict = graph_text_encoders.get_tlag_node_encoder(
        graph, encoding_method
    )
    labels = {node: str(name_dict[node]) for node in graph.nodes()}
    digest = graph_digest(graph, labels)

    images = {}
    for layout in layouts:
      file_name = '%s_%s.png' % (digest, layout)
      if render_images:
        target = os.path.join(images_dir, file_name)
        if not os.path.exists(target):
          graph_image_encoders.draw_graph(
              graph,
              target,
              layout=layout,
              random_seed=random_seed,
              name_dict=name_dict,
          )
      images[layout] = os.path.join(images_rel_dir, file_name).replace(
          os.sep, '/'
      )

    yield {
        'sample_id': '%s/%s/%s/%d'
        % (task_name, encoding_method, split, index),
        'index': index,
        'task': task_name,
        # Whether the task is part of the published GraphQA benchmark, so the
        # analysis can separate results comparable to the paper's Table 1 from
        # the harder extras.
        'graphqa_task': task_name in GRAPHQA_TASKS,
        'split': split,
        'text_encoder': encoding_method,
        'text_encoding': graph_encoding,
        'extra_context': extra_context,
        'question': value['task_description'],
        'prompt_text': value['question'],
        'answer': value['answer'],
        'images': images,
        'graph': serialize_graph(graph),
        'graph_id': digest,
        'node_labels': {str(k): v for k, v in labels.items()},
        'node_ids': list(value['node_ids']),
        'algorithm': value['algorithm'],
        'directed': graph.is_directed(),
        'nnodes': int(value['nnodes']),
        'nedges': int(value['nedges']),
    }


def write_jsonl(records: Iterator[dict[str, Any]], output_path: str) -> int:
  """Writes records as JSON lines, returning how many were written."""
  parent = os.path.dirname(output_path)
  if parent:
    os.makedirs(parent, exist_ok=True)
  count = 0
  with open(output_path, 'w', encoding='utf-8') as output_file:
    for record in records:
      output_file.write(json.dumps(record, ensure_ascii=False) + '\n')
      count += 1
  return count


def build_dataset(
    output_dir: str,
    tasks: Sequence[str] = tuple(GRAPHQA_TASKS),
    text_encoders: Sequence[str] = ('adjacency',),
    algorithms: Sequence[str] = ALGORITHMS,
    layouts: Sequence[str] = graph_image_encoders.LAYOUTS,
    split: str = 'test',
    number_of_graphs: int = 50,
    directed: bool = False,
    graphs_dir: str | None = None,
    random_seed: int = 1234,
    render_images: bool = True,
) -> dict[str, Any]:
  """Builds the full dataset on disk.

  The output directory ends up as::

      output_dir/
        images/<graph_id>_<layout>.png
        <task>_<encoder>_<split>.jsonl
        dataset_info.json

  Args:
    output_dir: where the dataset is written.
    tasks: which tasks to build. Defaults to the 7 GraphQA tasks of the paper;
      pass names from `EXTRA_TASKS` to add the harder ones.
    text_encoders: which text encoders to build.
    algorithms: which graph generator algorithms to include.
    layouts: which image layouts to render per graph.
    split: the split to build.
    number_of_graphs: graphs generated per algorithm.
    directed: whether to use directed graphs.
    graphs_dir: read graphs from disk instead of generating them.
    random_seed: seed for example construction and layouts.
    render_images: set to False for a text-only dry run.

  Returns:
    The dataset manifest, also written to `dataset_info.json`.
  """
  images_dir = os.path.join(output_dir, 'images')
  os.makedirs(images_dir, exist_ok=True)

  graphs, generator_algorithms = build_graphs(
      algorithms=algorithms,
      split=split,
      number_of_graphs=number_of_graphs,
      directed=directed,
      graphs_dir=graphs_dir,
  )
  if not graphs:
    raise ValueError('No graphs were generated or loaded.')

  # Node classification needs SBM graphs carrying their community labels.
  random.seed(random_seed)
  random_state = np.random.RandomState(random_seed)
  sbm_graphs = [
      generate_random_sbm_graph(random_state) for _ in range(len(graphs))
  ]
  sbm_algorithms = ['sbm'] * len(sbm_graphs)

  manifest: dict[str, Any] = {
      'split': split,
      'directed': directed,
      'algorithms': list(algorithms),
      'layouts': list(layouts),
      'text_encoders': list(text_encoders),
      'random_seed': random_seed,
      'number_of_graphs': len(graphs),
      'files': {},
  }

  for task_name in tasks:
    if task_name == 'node_classification':
      task_graphs, task_algorithms = sbm_graphs, sbm_algorithms
    else:
      task_graphs, task_algorithms = graphs, generator_algorithms
    for encoding_method in text_encoders:
      file_name = '%s_%s_%s.jsonl' % (task_name, encoding_method, split)
      count = write_jsonl(
          build_records(
              task_name=task_name,
              graphs=task_graphs,
              generator_algorithms=task_algorithms,
              encoding_method=encoding_method,
              split=split,
              images_dir=images_dir,
              layouts=layouts,
              random_seed=random_seed,
              render_images=render_images,
          ),
          os.path.join(output_dir, file_name),
      )
      manifest['files'][file_name] = count
      print('wrote %s (%d examples)' % (file_name, count))

  manifest['number_of_images'] = len(os.listdir(images_dir))
  with open(
      os.path.join(output_dir, 'dataset_info.json'), 'w', encoding='utf-8'
  ) as info_file:
    json.dump(manifest, info_file, indent=2)
  return manifest
