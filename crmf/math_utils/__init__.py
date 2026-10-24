"""General-purpose numerical helpers used by the CRMF domain modules."""

from .arrays import block_matrix, symmetric_lower
from .eigensolver import symmetric_diagonalization

__all__ = ["block_matrix", "symmetric_lower", "symmetric_diagonalization"]
