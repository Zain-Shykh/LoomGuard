import importlib


def test_package_imports():
    for name in ["loomguard", "loomguard.data", "loomguard.models", "loomguard.eval", "loomguard.inspection"]:
        importlib.import_module(name)
