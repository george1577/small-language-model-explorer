"""The synthetic training set made by the example's generate_training_set.py (no API calls here)."""

import ast
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# What generate_training_set.py sends to Claude for every example.
GENERATION_SYSTEM = "You are a helpful AI assistant that generates high-quality Python code examples."
GENERATION_USER = (
    "Generate a non-trivial but not too complex Python code example. "
    "Focus on common programming patterns and best practices. "
    "Return only the code, no documentation or explanation. "
    "Make sure the code is complete and limited to 1000 tokens."
)

# How create_dataset() wraps each example into the text the student is trained on.
TRAINING_TEMPLATE = """Below is a Python code snippet. Please generate comprehensive documentation for it.

Code:
{code}

Documentation:"""

TRUNCATION_LENGTH = 1024   # knowledge_distillation.py pads/truncates every text to this many tokens
TEST_SPLIT = 0.1           # TRAIN_TEST_SPLIT_RATIO


def load_examples(path: Path) -> tuple[list[str], dict]:
    data = json.loads(Path(path).read_text())
    return data["examples"], data.get("metadata", {})


def parses_as_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def class_names(code: str) -> list[str]:
    return re.findall(r"^class (\w+)", code, re.M)


@dataclass
class DatasetReport:
    num_examples: int
    num_unique: int
    num_parse: int                      # examples that are complete, valid Python
    class_counts: list[tuple[str, int]]  # most common class names
    share_with_top_class: float          # share of examples containing the most common class name


def report(examples: list[str]) -> DatasetReport:
    counts = Counter(name for e in examples for name in set(class_names(e)))
    top = counts.most_common(15)
    share = top[0][1] / len(examples) if top else 0.0
    return DatasetReport(
        num_examples=len(examples),
        num_unique=len(set(examples)),
        num_parse=sum(parses_as_python(e) for e in examples),
        class_counts=top,
        share_with_top_class=share,
    )
