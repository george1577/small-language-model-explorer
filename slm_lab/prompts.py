"""Prompts taken from the example scripts."""

# model_comparison.py: ask for documentation of this code (shortened from the script's snippet so
# the large model reads it in reasonable time on a laptop CPU).
COMPARISON_CODE = '''@dataclass
class Task:
    id: int
    title: str
    description: str
    completed: bool = False
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def mark_completed(self) -> None:
        self.completed = True

    def add_tag(self, tag: str) -> None:
        if tag not in self.tags:
            self.tags.append(tag)


class TaskManager:
    def __init__(self, storage_path: Optional[Path] = None):
        self.tasks: Dict[int, Task] = {}
        self.storage_path = storage_path
        self.next_id = 1

    def add_task(self, title: str, description: str, tags: Optional[List[str]] = None) -> Task:
        task = Task(id=self.next_id, title=title, description=description, tags=tags or [])
        self.tasks[task.id] = task
        self.next_id += 1
        logger.info(f"Added task {task.id}: {title}")
        return task
'''

COMPARISON_PROMPT = f"""Below is a Python code snippet. Please generate documentation for the code.

Code:
{COMPARISON_CODE}
Documentation:"""

# model_quantization.py
QUANTIZATION_PROMPT = "Explain quantum computing in simple terms."

# speculative_decoding.py
SPECULATIVE_PROMPTS = [
    "The future of AI is",
    "The black forest is located in",
    "The cathedral of Notre Dame is known for its",
]

NEXT_TOKEN_TEXT = "def fibonacci(n):\n    if n <= 1:\n        return"
