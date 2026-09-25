"""Packaging and distribution metadata checks."""

from importlib import metadata, resources

import jaxphys


def test_py_typed_marker_is_present() -> None:
    assert resources.files("jaxphys").joinpath("py.typed").is_file()


def test_version_matches_distribution_metadata() -> None:
    assert jaxphys.__version__ == metadata.version("jaxphys")
