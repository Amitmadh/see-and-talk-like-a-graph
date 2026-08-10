r"""CLI for building the "See and Talk Like a Graph" multimodal dataset.

Example:

```sh
python -m talk_like_a_graph.see_and_talk_dataset_runner \
  --output_dir=/tmp/see_and_talk \
  --split=test \
  --number_of_graphs=50
```

This builds the 7 GraphQA tasks of the paper. The released code implements 5
further tasks that the paper does not evaluate; add them with `--all_tasks`:

```sh
python -m talk_like_a_graph.see_and_talk_dataset_runner \
  --output_dir=/tmp/see_and_talk_all \
  --all_tasks
```

or pick individual ones with `--tasks` (mutually exclusive with
`--all_tasks`):

```sh
python -m talk_like_a_graph.see_and_talk_dataset_runner \
  --output_dir=/tmp/see_and_talk_hard \
  --tasks=reachability,shortest_path,maximum_flow,triangle_counting
```
"""

from collections.abc import Sequence

from absl import app
from absl import flags

from . import graph_image_encoders
from . import see_and_talk_dataset


_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'The directory to write the dataset to.', required=True
)
_GRAPHS_DIR = flags.DEFINE_string(
    'graphs_dir',
    None,
    'Optional directory of .graphml graphs written by '
    'graph_generators_runner. Graphs are generated on the fly when unset.',
)
_SPLIT = flags.DEFINE_enum(
    'split', 'test', ['train', 'test', 'validation'], 'The split to build.'
)
_NUMBER_OF_GRAPHS = flags.DEFINE_integer(
    'number_of_graphs', 50, 'Graphs generated per algorithm.'
)
_TASKS = flags.DEFINE_list(
    'tasks',
    list(see_and_talk_dataset.GRAPHQA_TASKS),
    'The tasks to build. Defaults to the 7 GraphQA tasks evaluated in "Talk '
    'like a Graph". Also available, but not part of that benchmark: '
    + ', '.join(see_and_talk_dataset.EXTRA_TASKS)
    + '.',
)
_ALL_TASKS = flags.DEFINE_bool(
    'all_tasks',
    False,
    'Build every implemented task: the %d GraphQA tasks plus the %d extras. '
    'Cannot be combined with --tasks.'
    % (
        len(see_and_talk_dataset.GRAPHQA_TASKS),
        len(see_and_talk_dataset.EXTRA_TASKS),
    ),
)
_TEXT_ENCODERS = flags.DEFINE_list(
    'text_encoders', ['adjacency'], 'The text encoders to build.'
)
_ALGORITHMS = flags.DEFINE_list(
    'algorithms',
    list(see_and_talk_dataset.ALGORITHMS),
    'The graph generator algorithms to include.',
)
_LAYOUTS = flags.DEFINE_list(
    'layouts',
    list(graph_image_encoders.LAYOUTS),
    'The image layouts to render per graph.',
)
_DIRECTED = flags.DEFINE_bool(
    'directed', False, 'Whether to use directed graphs.'
)
_RANDOM_SEED = flags.DEFINE_integer(
    'random_seed', 1234, 'The random seed to use.'
)
_INCLUDE_CAPACITIES = flags.DEFINE_bool(
    'include_capacities',
    True,
    'Append edge capacities to the text for weighted graphs (maximum_flow). '
    'The released encoders omit them, which makes that task unanswerable from '
    'text while the image shows them. Set false to reproduce that behaviour.',
)
_ANNOTATE_KNOWN_LABELS = flags.DEFINE_bool(
    'annotate_known_labels',
    True,
    "Draw node classification's already-known classes on the image, so its "
    'image-only setting does not silently receive them as text. Only the '
    'revealed labels are drawn, never the queried node.',
)
_RENDER_IMAGES = flags.DEFINE_bool(
    'render_images', True, 'Set to false for a text-only dry run.'
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if _ALL_TASKS.value:
    # Fail loudly rather than silently ignoring one of two conflicting flags.
    if flags.FLAGS['tasks'].present:
      raise app.UsageError('Pass either --all_tasks or --tasks, not both.')
    tasks = list(see_and_talk_dataset.TASKS)
  else:
    tasks = _TASKS.value

  unknown = [task for task in tasks if task not in see_and_talk_dataset.TASKS]
  if unknown:
    raise app.UsageError(
        'Unknown task(s): %s. Available: %s.'
        % (', '.join(unknown), ', '.join(see_and_talk_dataset.TASKS))
    )

  manifest = see_and_talk_dataset.build_dataset(
      output_dir=_OUTPUT_DIR.value,
      tasks=tasks,
      text_encoders=_TEXT_ENCODERS.value,
      algorithms=_ALGORITHMS.value,
      layouts=_LAYOUTS.value,
      split=_SPLIT.value,
      number_of_graphs=_NUMBER_OF_GRAPHS.value,
      directed=_DIRECTED.value,
      graphs_dir=_GRAPHS_DIR.value,
      random_seed=_RANDOM_SEED.value,
      render_images=_RENDER_IMAGES.value,
      include_capacities=_INCLUDE_CAPACITIES.value,
      annotate_known_labels=_ANNOTATE_KNOWN_LABELS.value,
  )
  print(
      'Done. %d examples across %d files, %d images.'
      % (
          sum(manifest['files'].values()),
          len(manifest['files']),
          manifest['number_of_images'],
      )
  )


if __name__ == '__main__':
  app.run(main)
