"""
Specialized manifest generators for different dataset formats.

Each generator module contains logic to parse a specific dataset structure
and produce standardized CSV manifests.
"""

__all__ = [
    "librispeech",
    "musan",
    "rirs",
    "primock57",
    "hf_generic",
    "afrimedqa",
]
