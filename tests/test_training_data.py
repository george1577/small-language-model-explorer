from pathlib import Path

from slm_lab.training_data import class_names, load_examples, parses_as_python, report


def test_bundled_dataset_loads():
    examples, meta = load_examples(Path("data/python_code_examples.json"))
    assert len(examples) == meta["num_examples"] == 1000


def test_report_helpers():
    assert class_names("class A:\n    pass\nclass B(A):\n    pass") == ["A", "B"]
    assert parses_as_python("x = 1") and not parses_as_python("def f(:")
    r = report(["class A: pass", "class A: pass", "x ="])
    assert (r.num_unique, r.num_parse, r.class_counts[0]) == (2, 2, ("A", 2))
