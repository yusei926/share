"""Extensible, command-free model contracts for physical-G1 evaluation."""

from .registry import ModelSpec, get_model_spec, load_registry

__all__ = ("ModelSpec", "get_model_spec", "load_registry")
