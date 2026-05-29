"""Neural Network Modules for Conditional Diffusion.

This module provides modular building blocks for constructing
conditional diffusion models:

- Embeddings: Class, Timestep, Combined
- Normalization: AdaGroupNorm, AdaLayerNorm, FiLM
- Attention: CrossAttention
"""

from acg.modules.attention import CrossAttention
from acg.modules.embeddings import ClassEmbedding, CombinedEmbedding, TimestepEmbedding
from acg.modules.normalization import AdaGroupNorm, AdaLayerNorm, FiLM

__all__ = [
    # Embeddings
    "ClassEmbedding",
    "TimestepEmbedding",
    "CombinedEmbedding",
    # Normalization
    "AdaGroupNorm",
    "AdaLayerNorm",
    "FiLM",
    # Attention
    "CrossAttention",
]
