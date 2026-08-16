"""Prompt templates for the three input settings (text-only / image-only / text+image).

The original Talk-Like-A-Graph pipeline (see talk_like_a_graph/graph_tasks.py) builds
prompts like:

    <graph encoded as text>
    Q: Is there a cycle in this graph?
    A:

We keep that question format, but wrap it with a short instruction so the VLM knows
WHICH modality/modalities it is being given and is nudged to use them. Start zero-shot
(no few-shot examples) for the PoC; few-shot / chain-of-thought can be added later.
"""

from __future__ import annotations

# Setting names — must match the modes Itamar's experiment loop uses.
TEXT_ONLY = "text_only"
IMAGE_ONLY = "image_only"
TEXT_AND_IMAGE = "image_and_text"

SETTINGS = {TEXT_ONLY, IMAGE_ONLY, TEXT_AND_IMAGE}

FORMAT_RESTRICTION_PROMPT = """
Answer with exactly one answer and nothing else.

For yes/no questions, output exactly:
True
or
False

For numeric questions, output only the number.

For questions asking for a list of nodes, output only the node numbers,
separated by commas. Do not include brackets or any explanation.

Example:
8, 11, 13
"""



_INSTRUCTIONS = {
    TEXT_ONLY: (
        "You are given a graph described in text. "
        "Read the description carefully and answer the question."
    ),
    IMAGE_ONLY: (
        "You are given an image of a graph. Nodes are circles with numeric "
        "labels and edges are the lines between them. "
        "Look at the image carefully and answer the question."
    ),
    TEXT_AND_IMAGE: (
        "You are given BOTH a text description of a graph AND an image of the "
        "same graph. Use the text description and the visual structure in the "
        "image together to answer the question."
    ),
}


def build_prompt(setting: str, text: str | None, question: str) -> str:
    """Assemble the text prompt for a given setting.

    Args:
        setting: one of TEXT_ONLY / IMAGE_ONLY / TEXT_AND_IMAGE.
        text:    the graph's text encoding (adjacency string). Ignored for IMAGE_ONLY.
        question: the task question, e.g. "Q: Is there a cycle in this graph?\nA: ".

    Returns:
        The full text prompt string. (The image itself is attached separately by the
        model wrapper — this function only builds the text half.)
    """
    if setting not in SETTINGS:
        raise ValueError(f"Unknown setting {setting!r}; expected one of {SETTINGS}")

    instruction = _INSTRUCTIONS[setting]

    if setting == IMAGE_ONLY:
        return f"{instruction}\n\n{FORMAT_RESTRICTION_PROMPT.strip()}\n\n{question}"

    # text_only and image_and_text both include the text encoding.
    return f"{instruction}\n\n{text}\n\n{FORMAT_RESTRICTION_PROMPT.strip()}\n\n{question}"