"""Packaging and distribution metadata checks."""

from importlib import resources


def test_py_typed_marker_is_present() -> None:
    assert resources.files("neurosim").joinpath("py.typed").is_file()
