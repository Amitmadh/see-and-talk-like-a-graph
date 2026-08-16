from dataclasses import dataclass
from pathlib import Path
import json

NON_METADATA_KEYS = {
                    "sample_id",
                    "graph",
                    "question",
                    "answer",
                    "text_encoding",
                    "images",
                }

@dataclass
class GraphQADataset:
    name: str
    samples: list["Sample"]

    def __init__(self, name, samples):
        self.name = name
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    


@dataclass
class Sample:
    sample_id: str

    graph: dict

    question: str
    answer: str

    text_encoding: str

    images: dict[str, str]

    metadata: dict

    node_ids: list[int]

    # sample.text_encoding_corrupted: str | None
    # sample.corrupted_answer: Any | None   (ground truth under the corrupted graph)

def load_graphqa_dataset(dataset_dir: str) -> GraphQADataset:
    dataset_path: Path = Path(dataset_dir)

    assert dataset_path.is_file(), f"Dataset file {dataset_path} does not exist."
    assert dataset_path.suffix == ".jsonl", f"Dataset file {dataset_path} is not a JSON file."


    raw_samples = []
    samples = []


    with open(dataset_path, "r") as f:
        for line in f:
            raw_samples.append(json.loads(line))

    for row in raw_samples:
        sample = Sample(
            sample_id=row["sample_id"],

            graph=row["graph"],

            question=row["question"],
            answer=row["answer"],

            text_encoding=row["text_encoding"],

            images=row.get("images", {}),

            metadata={
                k: v
                for k, v in row.items()
                if k not in NON_METADATA_KEYS
            },
            node_ids=row.get("node_ids")
        )

        samples.append(sample)

    return GraphQADataset(
        name=dataset_path.name,
        samples=samples
    )
