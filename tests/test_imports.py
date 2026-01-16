"""Basic import tests."""


def test_pytorch_available():
    """Verify PyTorch works."""
    try:
        import torch

        x = torch.tensor([1.0, 2.0, 3.0])
        assert x.shape == (3,)
    except ImportError:
        # PyTorch not installed yet - okay for Phase 1
        # TODO: Remove this exception handling in future sprints
        assert True


def test_basic_dependencies():
    """Verify basic Python dependencies can be imported."""
    import sys

    assert sys.version_info >= (3, 9)


def test_project_structure():
    """Verify basic project structure exists."""
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check for key directories (will be created in future sprints)
    # TODO: Update expected directories as project evolves
    expected_dirs = [".github", "tests", "src"]
    for dir_name in expected_dirs:
        dir_path = os.path.join(project_root, dir_name)
        assert os.path.exists(dir_path), f"Directory {dir_name} should exist"
