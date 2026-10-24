r"""Source-compatible symmetric eigensolver, implemented in Python with Numba.

This is the Householder reduction and implicit QR algorithm of SDIAG in
``dirhbs_CRMF.f``. The original stopping tolerance is preserved deliberately
for reproducibility. It is less strict than LAPACK's machine-precision
stopping rule, especially for matrices with Dirac-sea energies.
"""
import math

import numpy as np
from numba import njit


@njit(cache=True)
def _symmetric_diagonalization_kernel(matrix, tolerance, epsilon):
    r"""Reduce a real symmetric matrix and solve its tridiagonal form.

    Householder transformations produce $Q^TAQ=T$, then implicit QR
    iterations solve $Tz_i=\lambda_i z_i$ and return $v_i=Qz_i$.
    """
    matrix_dimension = matrix.shape[0]
    transformation_matrix = np.zeros((matrix_dimension + 1, matrix_dimension + 1))
    diagonal_values = np.zeros(matrix_dimension + 1)
    off_diagonal_values = np.zeros(matrix_dimension + 1)
    if matrix_dimension == 1:
        diagonal_values[1] = matrix[0, 0]
        transformation_matrix[1, 1] = 1.0
        return diagonal_values[1:], transformation_matrix[1:, 1:]
    for row_index in range(1, matrix_dimension + 1):
        for column_index in range(1, row_index + 1):
            transformation_matrix[row_index, column_index] = matrix[row_index - 1, column_index - 1]
    for row_index in range(matrix_dimension, 1, -1):
        reduction_length = row_index - 2
        factor = transformation_matrix[row_index, row_index - 1]
        magnitude = factor
        work_value = 0.0
        for inner_index in range(1, reduction_length + 1):
            work_value += transformation_matrix[row_index, inner_index] * transformation_matrix[row_index, inner_index]
        total = work_value + factor * factor
        if total < tolerance:
            work_value = 0.0
        elif work_value > 0.0:
            reduction_length += 1
            magnitude = math.sqrt(total)
            if factor >= 0.0:
                magnitude = -magnitude
            work_value = total - factor * magnitude
            inverse_work_value = 1.0 / work_value
            transformation_matrix[row_index, row_index - 1] = factor - magnitude
            factor = 0.0
            for column_index in range(1, reduction_length + 1):
                transformation_matrix[column_index, row_index] = (
                    transformation_matrix[row_index, column_index] * inverse_work_value
                )
                total = 0.0
                for inner_index in range(1, column_index + 1):
                    total += (
                        transformation_matrix[column_index, inner_index]
                        * transformation_matrix[row_index, inner_index]
                    )
                for inner_index in range(column_index + 1, reduction_length + 1):
                    total += (
                        transformation_matrix[inner_index, column_index]
                        * transformation_matrix[row_index, inner_index]
                    )
                off_diagonal_values[column_index] = total * inverse_work_value
                factor += total * transformation_matrix[column_index, row_index]
            factor = factor * inverse_work_value * 0.5
            for column_index in range(1, reduction_length + 1):
                total = transformation_matrix[row_index, column_index]
                off_diagonal_values[column_index] -= factor * total
                pivot_value = off_diagonal_values[column_index]
                for inner_index in range(1, column_index + 1):
                    transformation_matrix[column_index, inner_index] = (
                        transformation_matrix[column_index, inner_index]
                        - total * off_diagonal_values[inner_index]
                        - transformation_matrix[row_index, inner_index] * pivot_value
                    )
        diagonal_values[row_index] = work_value
        off_diagonal_values[row_index - 1] = magnitude
    diagonal_values[1] = 0.0
    off_diagonal_values[matrix_dimension] = 0.0
    threshold = 0.0
    shift = 0.0
    for row_index in range(1, matrix_dimension + 1):
        reduction_length = row_index - 1
        if diagonal_values[row_index] != 0.0:
            for column_index in range(1, reduction_length + 1):
                total = 0.0
                for inner_index in range(1, reduction_length + 1):
                    total += (
                        transformation_matrix[row_index, inner_index]
                        * transformation_matrix[inner_index, column_index]
                    )
                for inner_index in range(1, reduction_length + 1):
                    transformation_matrix[inner_index, column_index] -= (
                        total * transformation_matrix[inner_index, row_index]
                    )
        diagonal_values[row_index] = transformation_matrix[row_index, row_index]
        transformation_matrix[row_index, row_index] = 1.0
        for column_index in range(1, reduction_length + 1):
            transformation_matrix[row_index, column_index] = 0.0
            transformation_matrix[column_index, row_index] = 0.0
    for reduction_length in range(1, matrix_dimension + 1):
        work_value = epsilon * (
            abs(diagonal_values[reduction_length])
            + abs(off_diagonal_values[reduction_length])
        )
        threshold = max(threshold, work_value)
        column_index = reduction_length
        while abs(off_diagonal_values[column_index]) > threshold:
            column_index += 1
        if column_index != reduction_length:
            for iteration in range(1000):
                pivot_value = (
                    diagonal_values[reduction_length + 1]
                    - diagonal_values[reduction_length]
                ) / (2 * off_diagonal_values[reduction_length])
                rotation_radius = math.sqrt(pivot_value * pivot_value + 1.0)
                adjusted_pivot = (
                    pivot_value + rotation_radius
                    if pivot_value >= 0
                    else pivot_value - rotation_radius
                )
                work_value = (
                    diagonal_values[reduction_length]
                    - off_diagonal_values[reduction_length] / adjusted_pivot
                )
                for row_index in range(reduction_length, matrix_dimension + 1):
                    diagonal_values[row_index] -= work_value
                shift += work_value
                pivot_value, cosine, sine = diagonal_values[column_index], 1.0, 0.0
                for row_index in range(column_index - 1, reduction_length - 1, -1):
                    magnitude, work_value = cosine * off_diagonal_values[row_index], cosine * pivot_value
                    if abs(pivot_value) >= abs(off_diagonal_values[row_index]):
                        cosine = off_diagonal_values[row_index] / pivot_value
                        rotation_radius = math.sqrt(cosine * cosine + 1.0)
                        off_diagonal_values[row_index + 1] = sine * pivot_value * rotation_radius
                        sine, cosine = cosine / rotation_radius, 1.0 / rotation_radius
                    else:
                        cosine = pivot_value / off_diagonal_values[row_index]
                        rotation_radius = math.sqrt(cosine * cosine + 1.0)
                        off_diagonal_values[row_index + 1] = sine * off_diagonal_values[row_index] * rotation_radius
                        sine, cosine = 1.0 / rotation_radius, cosine / rotation_radius
                    pivot_value = cosine * diagonal_values[row_index] - sine * magnitude
                    diagonal_values[row_index + 1] = work_value + sine * (
                        cosine * magnitude + sine * diagonal_values[row_index]
                    )
                    for inner_index in range(1, matrix_dimension + 1):
                        work_value = transformation_matrix[inner_index, row_index + 1]
                        transformation_matrix[inner_index, row_index + 1] = (
                            transformation_matrix[inner_index, row_index] * sine
                            + work_value * cosine
                        )
                        transformation_matrix[inner_index, row_index] = (
                            transformation_matrix[inner_index, row_index] * cosine
                            - work_value * sine
                        )
                off_diagonal_values[reduction_length], diagonal_values[reduction_length] = sine * pivot_value, cosine * pivot_value
                if abs(off_diagonal_values[reduction_length]) <= threshold:
                    break
            else:
                raise RuntimeError("SDIAG QR iteration failed to converge")
        diagonal_values[reduction_length] += shift
    for row_index in range(1, matrix_dimension + 1):
        selected_index = row_index
        pivot_value = diagonal_values[row_index]
        for column_index in range(row_index + 1, matrix_dimension + 1):
            if diagonal_values[column_index] < pivot_value:
                selected_index, pivot_value = column_index, diagonal_values[column_index]
        if selected_index != row_index:
            diagonal_values[selected_index], diagonal_values[row_index] = diagonal_values[row_index], pivot_value
            for column_index in range(1, matrix_dimension + 1):
                transformation_matrix[column_index, row_index], transformation_matrix[column_index, selected_index] = (
                    transformation_matrix[column_index, selected_index], transformation_matrix[column_index, row_index]
                )
    for column_index in range(1, matrix_dimension + 1):
        largest = 0.0
        largest_component_index = 1
        for row_index in range(1, matrix_dimension + 1):
            if abs(transformation_matrix[row_index, column_index]) > largest:
                largest, largest_component_index = abs(transformation_matrix[row_index, column_index]), row_index
        if transformation_matrix[largest_component_index, column_index] < 0.0:
            for row_index in range(1, matrix_dimension + 1):
                transformation_matrix[row_index, column_index] = -transformation_matrix[row_index, column_index]
    return diagonal_values[1:], transformation_matrix[1:, 1:]


def symmetric_diagonalization(matrix):
    r"""Diagonalize a real symmetric matrix with the source-compatible SDIAG.

    $$
        A=Q\Lambda Q^T.
    $$

    The lower triangle supplies the matrix data. Householder reduction and
    implicit QR retain the original stopping thresholds for reproducibility.

    Args:
        matrix (numpy.ndarray): Finite square real matrix.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Ascending eigenvalues and
        corresponding orthonormal eigenvectors.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("SDIAG requires a square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("SDIAG requires finite matrix entries")
    if matrix.shape[0] == 0:
        return np.empty(0), np.empty((0, 0))
    # These DATA literals have default REAL precision in the Fortran source.
    eigenvalues, eigenvectors = _symmetric_diagonalization_kernel(
        matrix, float(np.float32(1e-32)), float(np.float32(1e-10)))
    if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
        raise FloatingPointError("SDIAG returned non-finite eigenpairs")
    return eigenvalues, eigenvectors


# Compatibility alias for the original Fortran routine name.
sdiag = symmetric_diagonalization
