import importlib

MODULES = ["data", "retrieve", "schema", "agent", "providers", "probes",
           "baselines", "runner", "metrics", "report"]


def test_modules_import():
    for m in MODULES:
        importlib.import_module(f"asrs_eval.{m}")
