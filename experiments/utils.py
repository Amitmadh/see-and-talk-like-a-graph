import json
from pathlib import Path
import yaml
import datetime


def load_results(results_path):
    """
    Load experiment results saved by save_results().
    """
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}"
        )

    with open(results_path, "r") as f:
        results = json.load(f)

    if not isinstance(results, list):
        raise ValueError(
            f"Expected results file to contain a JSON list, "
            f"got {type(results).__name__}"
        )

    return results
def load_config(config_path):
    """
    Load experiment configuration from yaml file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_results(results, output_path):
    """
    Save experiment results as json.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w") as f:
        json.dump(
            results,
            f,
            indent=2
        )

    print(f"Saved results to {output_path}")



def log(message):
    print(
        f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}"
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get_project_root():
    return PROJECT_ROOT