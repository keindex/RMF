"""Array layout helpers independent of nuclear physics models."""

import numpy as np


def block_matrix(array, size, column):
    r"""View a packed Fortran column as a square matrix.

    The active data occupy the first $n^2$ entries and use column-major
    ordering.

    Args:
        array: Padded two-dimensional array containing packed matrix columns.
        size (int): Active matrix dimension $n$.
        column (int): Column in the padded storage array.

    Returns:
        numpy.ndarray: A view with shape $(n, n)$.
    """
    packed_values = array[1:size * size + 1, column]
    return packed_values.reshape((size, size), order='F')


def symmetric_lower(matrix):
    r"""Mirror a matrix's lower triangle to produce a symmetric matrix.

    This implements $A\leftarrow L+L^T-\operatorname{diag}(L)$,
    where $L=\operatorname{tril}(A)$.

    Args:
        matrix (numpy.ndarray): Square matrix whose lower triangle is used.

    Returns:
        numpy.ndarray: Symmetric matrix $A_{ij}=A_{ji}$.
    """
    lower = np.tril(matrix)
    return lower + np.tril(lower, -1).T
