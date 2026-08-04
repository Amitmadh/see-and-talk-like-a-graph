# Using Large Language Models to Solve Graph Problems

This repository contains the code to generate graph reasoning problems with
different graph generator algorithms and graph encoding methods, as well as
different prompting techniques.

The graph tasks are `edge existence`, `node degree`, `node count`, `edge count`,
`connected nodes`, `disconnected nodes`, `cycle check`, `reachability`,
`shortest path`, `maximum flow`, `node classification`, and `triangle counting`.

The datasets used here are proposed in our paper:
[Talk like a Graph: Encoding Graphs for Large Language Models](https://arxiv.org/abs/2310.04560).

### Generating graphs

```sh
./graphqa/graph_generator.sh
```

### Generating files for tasks

```sh
./graphqa/task_generator.sh
```

## See and Talk Like a Graph: the multimodal dataset

`see_and_talk_dataset.py` builds the GraphQA benchmark in a form a
vision-language model can consume: every example carries both a textual
encoding and one or more rendered images of the same graph.

It is a standalone path. The scripts above write `tf.train.Example` protos into
recordio files via Google-internal libraries (`recordio`, `seqio`, `gfile`,
`tfgnn`) that are not importable outside Google, so this builder reuses only
the portable modules (`graph_generators`, `graph_tasks`,
`graph_text_encoders`) and writes JSONL + PNG instead.

```sh
python -m talk_like_a_graph.see_and_talk_dataset_runner \
  --output_dir=/tmp/see_and_talk \
  --split=test \
  --number_of_graphs=50
```

Useful flags: `--tasks`, `--text_encoders`, `--algorithms`, `--layouts`,
`--directed`, `--graphs_dir` (reuse `.graphml` graphs instead of generating
them) and `--render_images=false` for a fast text-only dry run.

### Which tasks are built

By default only the **7 GraphQA tasks** that "Talk like a Graph" actually
evaluates (Appendix A.2): `edge_existence`, `node_degree`, `node_count`,
`edge_count`, `connected_nodes`, `cycle_check` (its Experiments 1-4) and
`disconnected_nodes` (its Experiment 5). Keeping to this set is what makes our
accuracies comparable to Table 1 of the paper.

The released code also implements 5 harder tasks the paper does *not* evaluate
— `reachability`, `shortest_path`, `maximum_flow`, `node_classification` and
`triangle_counting`. In the paper, `shortest_path` appears only as a motivating
example of a multi-hop task, `node_classification` only in related work, and
the other three not at all. They are opt-in — either all 12 tasks at once:

```sh
--all_tasks
```

or a specific selection (`--tasks` and `--all_tasks` are mutually exclusive):

```sh
--tasks=reachability,shortest_path,maximum_flow,triangle_counting
```

They are worth a secondary run: being multi-hop and structural, they are the
likeliest place for a graph image to help. Every record carries a
`graphqa_task` boolean so the analysis can keep the two groups apart.

### Output

```
output_dir/
  images/<graph_id>_<layout>.png
  <task>_<encoder>_<split>.jsonl
  dataset_info.json
```

Each JSONL line is one example:

| field | meaning |
| --- | --- |
| `sample_id`, `index` | unique id, and position within the task file |
| `task`, `split`, `text_encoder`, `algorithm` | provenance |
| `graphqa_task` | whether the task is in the published benchmark, i.e. comparable to the paper |
| `text_encoding` | the graph rendered as text |
| `extra_context` | prompt context that is not the graph. Empty except `node_classification` (known node labels) and `maximum_flow` (edge capacities, see below) |
| `question` | the question alone, e.g. `Q: Is there a cycle in this graph?\nA: ` |
| `prompt_text` | `text_encoding + extra_context + question`, identical to the original pipeline's prompt |
| `answer` | the gold label |
| `images` | layout name -> path relative to the dataset root |
| `graph`, `graph_id`, `node_labels`, `node_ids` | the graph itself, its image id, the labels drawn on it, and the nodes the question refers to |
| `directed`, `nnodes`, `nedges` | metadata for the per-graph-type analysis |

The three evaluation settings are assembled from these fields as:

*   **text-only** — `prompt_text`
*   **image-only** — `images[layout]` + `extra_context` + `question`
*   **text+image** — `prompt_text` + `images[layout]`

### Images

Rendered with `graph_image_encoders.py`. Node labels come from the *same*
`get_tlag_node_encoder` mapping the text encoder uses, so the image and the
text always name nodes identically. Directed graphs get arrow heads and
weighted graphs (maximum flow) get capacities drawn on the edges. Nothing task
specific is drawn — the queried nodes are not highlighted and SBM communities
are not colour coded — since that would leak the answer.

Four layout alternatives are rendered per graph for comparison: `spring`,
`kamada_kawai`, `circular` and `planar` (falling back to `spring` for
non-planar graphs). Images are ~768x768 PNGs, deduplicated by a content hash
of the graph and its labels, so a graph shared across tasks is rendered once.

## Contact us

For questions or comments about the implementation, please contact
baharef@google.com.

## Cite us

If you use this package for published work, please cite the following:

```
@inproceedigs{fatemi2024talk,
  title={Talk like a Graph: Encoding Graphs for Large Language Models},
  author={Bahare Fatemi and Jonathan Halcrow and Bryan Perozzi},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

## Disclaimer

This is not an official Google product.

# Placeholder for internal data notes.