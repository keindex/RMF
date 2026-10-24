"""Separable pairing and Talmi--Moshinsky brackets translated from DIRHB.

Numba compiles Python loops only; this module neither uses nor compiles the
Fortran source. Radial quantum numbers in the angular routines start at zero.
"""
from functools import lru_cache
import math
import numpy as np
from numba import njit


def _create_factorial_tables():
    r"""Build cached factorial and half-integer gamma tables.

    The recurrences are $f_n=n f_{n-1}$ and
    $\sqrt{\Gamma(n+1/2)}=\sqrt{n-1/2}\sqrt{\Gamma(n-1/2)}$.
    """
    table_length = 101
    integer_signs = np.array([(-1)**index for index in range(table_length)], dtype=np.int64)
    factorials = np.empty(table_length)
    factorial_square_roots = np.empty(table_length)
    gamma_half_integer_square_roots = np.empty(table_length)
    factorials[0] = factorial_square_roots[0] = 1.0
    gamma_half_integer_square_roots[0] = math.pi**0.25
    for index in range(1, table_length):
        factorials[index] = index*factorials[index-1]
        factorial_square_roots[index] = math.sqrt(index)*factorial_square_roots[index-1]
        gamma_half_integer_square_roots[index] = math.sqrt(index-.5)*gamma_half_integer_square_roots[index-1]
    return (integer_signs, factorials, 1/factorials, factorial_square_roots,
            1/factorial_square_roots, gamma_half_integer_square_roots,
            1/gamma_half_integer_square_roots)


(INTEGER_SIGNS, FACTORIALS, INVERSE_FACTORIALS, FACTORIAL_SQUARE_ROOTS,
 INVERSE_FACTORIAL_SQUARE_ROOTS, GAMMA_HALF_INTEGER_SQUARE_ROOTS,
 INVERSE_GAMMA_HALF_INTEGER_SQUARE_ROOTS) = _create_factorial_tables()


@njit(cache=True)
def _triangle_factor(first_angular_momentum, second_angular_momentum,
                     third_angular_momentum):
    r"""Return the triangle normalization used by a Racah sum.

    In factorial notation, $\Delta(a,b,c)=\sqrt{(a+b-c)!
    (a-b+c)!(b+c-a)!/(a+b+c+1)!}$.
    """
    return (FACTORIAL_SQUARE_ROOTS[first_angular_momentum + second_angular_momentum - third_angular_momentum]
            * INVERSE_FACTORIAL_SQUARE_ROOTS[first_angular_momentum + second_angular_momentum + third_angular_momentum + 1]
            * FACTORIAL_SQUARE_ROOTS[first_angular_momentum - second_angular_momentum + third_angular_momentum]
            * FACTORIAL_SQUARE_ROOTS[second_angular_momentum + third_angular_momentum - first_angular_momentum])


@njit(cache=True)
def _racah_coefficient(first_j, second_j, third_j, first_l, second_l, third_l):
    r"""Evaluate the Racah coefficient by its finite factorial sum.

    The implementation uses $W=\Delta_1\Delta_2\Delta_3\Delta_4
    \sum_z(-1)^z(z+1)!/\prod_i(z-a_i)!\prod_j(b_j-z)!$.
    """
    for first_angular_momentum, second_angular_momentum, third_angular_momentum in (
            (first_j, second_j, third_j), (first_j, second_l, third_l),
            (first_l, second_j, third_l), (first_l, second_l, third_j)):
        if (third_angular_momentum <
                abs(first_angular_momentum-second_angular_momentum) or
                third_angular_momentum > first_angular_momentum+second_angular_momentum):
            return 0.0
    first_sum = first_j + second_j + third_j
    second_sum = first_j + second_l + third_l
    third_sum = first_l + second_j + third_l
    fourth_sum = first_l + second_l + third_j
    fifth_sum = first_j + second_j + first_l + second_l
    sixth_sum = second_j + third_j + second_l + third_l
    seventh_sum = third_j + first_j + third_l + first_l
    lower_sum = max(first_sum, second_sum, third_sum, fourth_sum)
    upper_sum = min(fifth_sum, sixth_sum, seventh_sum)
    result = 0.0
    for summation_index in range(lower_sum, upper_sum+1):
        result += (INTEGER_SIGNS[summation_index]*FACTORIALS[summation_index+1]
                   * INVERSE_FACTORIALS[summation_index-first_sum]
                   * INVERSE_FACTORIALS[summation_index-second_sum]
                   * INVERSE_FACTORIALS[summation_index-third_sum]
                   * INVERSE_FACTORIALS[summation_index-fourth_sum]
                   * INVERSE_FACTORIALS[fifth_sum-summation_index]
                   * INVERSE_FACTORIALS[sixth_sum-summation_index]
                   * INVERSE_FACTORIALS[seventh_sum-summation_index])
    return (_triangle_factor(first_j, second_j, third_j)
            *_triangle_factor(first_j, second_l, third_l)
            *_triangle_factor(first_l, second_j, third_l)
            *_triangle_factor(first_l, second_l, third_j)*result)


@njit(cache=True)
def _wigner_nine_j_symbol(first_row_first, first_row_second, first_row_third,
                           second_row_first, second_row_second, second_row_third,
                           third_row_first, third_row_second, third_row_third):
    r"""Evaluate a Wigner $9j$ symbol by recoupling three Racah coefficients.

    With doubled angular momenta, the sum is
    $\left\{\begin{smallmatrix}a&b&c\\d&e&f\\g&h&i\end{smallmatrix}\right\}
    =\sum_x(2x+1)W_1(x)W_2(x)W_3(x)$.
    """
    for first_angular_momentum, second_angular_momentum, third_angular_momentum in (
            (first_row_first, first_row_second, first_row_third),
            (second_row_first, second_row_second, second_row_third),
            (third_row_first, third_row_second, third_row_third),
            (first_row_first, second_row_first, third_row_first),
            (first_row_second, second_row_second, third_row_second),
            (first_row_third, second_row_third, third_row_third)):
        if (third_angular_momentum <
                abs(first_angular_momentum-second_angular_momentum) or
                third_angular_momentum > first_angular_momentum+second_angular_momentum):
            return 0.0
    lower_intermediate = max(abs(second_row_first-third_row_second),
                             abs(first_row_first-third_row_third),
                             abs(first_row_second-second_row_third))
    upper_intermediate = min(second_row_first+third_row_second,
                             first_row_second+second_row_third,
                             first_row_first+third_row_third)
    result = 0.0
    for intermediate_angular_momentum in range(lower_intermediate, upper_intermediate+1):
        result += ((2*intermediate_angular_momentum+1)
                   *_racah_coefficient(third_row_second, second_row_first,
                                       intermediate_angular_momentum,
                                       first_row_first, third_row_third,
                                       third_row_first)
                   *_racah_coefficient(first_row_second, second_row_second,
                                       third_row_second, second_row_first,
                                       intermediate_angular_momentum,
                                       second_row_third)
                   *_racah_coefficient(intermediate_angular_momentum,
                                       second_row_third, first_row_second,
                                       first_row_third, first_row_first,
                                       third_row_third))
    return result


@njit(cache=True)
def _wigner_three_j_prefactor(first_angular_momentum, second_angular_momentum,
                              third_angular_momentum):
    r"""Return the triangle and factorial prefactor of a $3j$ symbol.

    It vanishes unless $|a-b|\le c\le a+b$ and $a+b+c$ is even.
    """
    if (abs(first_angular_momentum-second_angular_momentum) > third_angular_momentum
            or first_angular_momentum+second_angular_momentum < third_angular_momentum
            or (first_angular_momentum+second_angular_momentum+third_angular_momentum)%2):
        return 0.0
    half_sum = (first_angular_momentum+second_angular_momentum+third_angular_momentum)//2
    return (GAMMA_HALF_INTEGER_SQUARE_ROOTS[half_sum-first_angular_momentum]
            * INVERSE_FACTORIAL_SQUARE_ROOTS[half_sum-first_angular_momentum]
            * GAMMA_HALF_INTEGER_SQUARE_ROOTS[half_sum-second_angular_momentum]
            * INVERSE_FACTORIAL_SQUARE_ROOTS[half_sum-second_angular_momentum]
            * GAMMA_HALF_INTEGER_SQUARE_ROOTS[half_sum-third_angular_momentum]
            * INVERSE_FACTORIAL_SQUARE_ROOTS[half_sum-third_angular_momentum]
            * FACTORIAL_SQUARE_ROOTS[half_sum]
            * INVERSE_GAMMA_HALF_INTEGER_SQUARE_ROOTS[half_sum+1]*math.sqrt(.5))


@njit(cache=True)
def _talmi_moshinsky_bracket(radial_quantum_number_one,
                             orbital_angular_momentum_one,
                             radial_quantum_number_two,
                             orbital_angular_momentum_two,
                             radial_quantum_number_three,
                             orbital_angular_momentum_three,
                             radial_quantum_number_four,
                             orbital_angular_momentum_four,
                             coupling_angular_momentum):
    r"""Evaluate the finite Talmi-Moshinsky transformation sum.

    Oscillator quanta are conserved, $2n_1+l_1+2n_2+l_2
    =2n_3+l_3+2n_4+l_4$, and each allowed term is weighted by
    $3j$ and $9j$ factors with the appropriate phase.
    """
    if (abs(orbital_angular_momentum_one-orbital_angular_momentum_two) > coupling_angular_momentum
            or orbital_angular_momentum_one+orbital_angular_momentum_two < coupling_angular_momentum
            or abs(orbital_angular_momentum_three-orbital_angular_momentum_four) > coupling_angular_momentum
            or orbital_angular_momentum_three+orbital_angular_momentum_four < coupling_angular_momentum):
        return 0.0
    oscillator_quanta_one = (2*radial_quantum_number_one
                             + orbital_angular_momentum_one)
    oscillator_quanta_two = (2*radial_quantum_number_two
                             + orbital_angular_momentum_two)
    oscillator_quanta_three = (2*radial_quantum_number_three
                               + orbital_angular_momentum_three)
    oscillator_quanta_four = (2*radial_quantum_number_four
                              + orbital_angular_momentum_four)
    total_oscillator_quanta = oscillator_quanta_one + oscillator_quanta_two
    if total_oscillator_quanta != oscillator_quanta_three + oscillator_quanta_four:
        return 0.0
    if min(orbital_angular_momentum_one, orbital_angular_momentum_two) < min(
            orbital_angular_momentum_three, orbital_angular_momentum_four):
        if orbital_angular_momentum_one > orbital_angular_momentum_two:
            radial_number_three, angular_momentum_three = (
                radial_quantum_number_two, orbital_angular_momentum_two)
            radial_number_four, angular_momentum_four = (
                radial_quantum_number_one, orbital_angular_momentum_one)
            if oscillator_quanta_three > oscillator_quanta_four:
                radial_number_one, angular_momentum_one = (
                    radial_quantum_number_four, orbital_angular_momentum_four)
                radial_number_two, angular_momentum_two = (
                    radial_quantum_number_three, orbital_angular_momentum_three)
                phase_factor = INTEGER_SIGNS[angular_momentum_one+angular_momentum_four]
            else:
                radial_number_one, angular_momentum_one = (
                    radial_quantum_number_three, orbital_angular_momentum_three)
                radial_number_two, angular_momentum_two = (
                    radial_quantum_number_four, orbital_angular_momentum_four)
                phase_factor = INTEGER_SIGNS[angular_momentum_one+coupling_angular_momentum]
        else:
            radial_number_three, angular_momentum_three = (
                radial_quantum_number_one, orbital_angular_momentum_one)
            radial_number_four, angular_momentum_four = (
                radial_quantum_number_two, orbital_angular_momentum_two)
            if oscillator_quanta_three > oscillator_quanta_four:
                radial_number_one, angular_momentum_one = (
                    radial_quantum_number_four, orbital_angular_momentum_four)
                radial_number_two, angular_momentum_two = (
                    radial_quantum_number_three, orbital_angular_momentum_three)
                phase_factor = INTEGER_SIGNS[angular_momentum_three+coupling_angular_momentum]
            else:
                radial_number_one, angular_momentum_one = (
                    radial_quantum_number_three, orbital_angular_momentum_three)
                radial_number_two, angular_momentum_two = (
                    radial_quantum_number_four, orbital_angular_momentum_four)
                phase_factor = 1
    else:
        if orbital_angular_momentum_three > orbital_angular_momentum_four:
            radial_number_three, angular_momentum_three = (
                radial_quantum_number_four, orbital_angular_momentum_four)
            radial_number_four, angular_momentum_four = (
                radial_quantum_number_three, orbital_angular_momentum_three)
            if oscillator_quanta_one > oscillator_quanta_two:
                radial_number_one, angular_momentum_one, radial_number_two, angular_momentum_two = (
                    radial_quantum_number_two, orbital_angular_momentum_two,
                    radial_quantum_number_one, orbital_angular_momentum_one)
                phase_factor = INTEGER_SIGNS[angular_momentum_one+angular_momentum_three]
            else:
                radial_number_one, angular_momentum_one = (
                    radial_quantum_number_one, orbital_angular_momentum_one)
                radial_number_two, angular_momentum_two = (
                    radial_quantum_number_two, orbital_angular_momentum_two)
                phase_factor = INTEGER_SIGNS[angular_momentum_one+coupling_angular_momentum]
        else:
            radial_number_three, angular_momentum_three = (
                radial_quantum_number_three, orbital_angular_momentum_three)
            radial_number_four, angular_momentum_four = (
                radial_quantum_number_four, orbital_angular_momentum_four)
            if oscillator_quanta_one > oscillator_quanta_two:
                radial_number_one, angular_momentum_one, radial_number_two, angular_momentum_two = (
                    radial_quantum_number_two, orbital_angular_momentum_two,
                    radial_quantum_number_one, orbital_angular_momentum_one)
                phase_factor = INTEGER_SIGNS[angular_momentum_three+coupling_angular_momentum]
            else:
                radial_number_one, angular_momentum_one = (
                    radial_quantum_number_one, orbital_angular_momentum_one)
                radial_number_two, angular_momentum_two = (
                    radial_quantum_number_two, orbital_angular_momentum_two)
                phase_factor = 1
    oscillator_quanta_one = 2*radial_number_one + angular_momentum_one
    oscillator_quanta_two = 2*radial_number_two + angular_momentum_two
    oscillator_quanta_three = 2*radial_number_three + angular_momentum_three
    oscillator_quanta_four = 2*radial_number_four + angular_momentum_four
    normalization_factor = (math.sqrt(angular_momentum_one+.5)
                            * FACTORIAL_SQUARE_ROOTS[radial_number_one]
                            * GAMMA_HALF_INTEGER_SQUARE_ROOTS[radial_number_one+angular_momentum_one+1]
                            * math.sqrt(angular_momentum_two+.5)
                            * FACTORIAL_SQUARE_ROOTS[radial_number_two]
                            * GAMMA_HALF_INTEGER_SQUARE_ROOTS[radial_number_two+angular_momentum_two+1]
                            * math.sqrt(angular_momentum_three+.5)
                            * FACTORIAL_SQUARE_ROOTS[radial_number_three]
                            * GAMMA_HALF_INTEGER_SQUARE_ROOTS[radial_number_three+angular_momentum_three+1]
                            * math.sqrt(angular_momentum_four+.5)
                            * FACTORIAL_SQUARE_ROOTS[radial_number_four]
                            * GAMMA_HALF_INTEGER_SQUARE_ROOTS[radial_number_four+angular_momentum_four+1])
    oscillator_scale_factor = math.sqrt(.5)**total_oscillator_quanta
    result = 0.0
    for angular_momentum_a in range(min(oscillator_quanta_one, oscillator_quanta_three)+1):
        lower_angular_momentum_b = abs(angular_momentum_a-angular_momentum_three)
        starting_angular_momentum_b = (lower_angular_momentum_b
                                       + (angular_momentum_a+lower_angular_momentum_b
                                          + angular_momentum_three)%2)
        upper_angular_momentum_b = min(oscillator_quanta_two,
                                       oscillator_quanta_three-angular_momentum_a,
                                       angular_momentum_a+angular_momentum_three)
        ending_angular_momentum_b = (upper_angular_momentum_b
                                     - (angular_momentum_a+upper_angular_momentum_b
                                        + angular_momentum_three)%2)
        for angular_momentum_b in range(starting_angular_momentum_b,
                                        ending_angular_momentum_b+1, 2):
            lower_angular_momentum_c = abs(angular_momentum_a-angular_momentum_one)
            starting_angular_momentum_c = (lower_angular_momentum_c
                                           + (angular_momentum_a+lower_angular_momentum_c
                                              + angular_momentum_one)%2)
            upper_angular_momentum_c = min(oscillator_quanta_one-angular_momentum_a,
                                           oscillator_quanta_four,
                                           angular_momentum_a+angular_momentum_one)
            ending_angular_momentum_c = (upper_angular_momentum_c
                                         - (angular_momentum_a+upper_angular_momentum_c
                                            + angular_momentum_one)%2)
            wigner_coefficient_three = _wigner_three_j_prefactor(
                angular_momentum_a, angular_momentum_b, angular_momentum_three)
            for angular_momentum_c in range(starting_angular_momentum_c,
                                            ending_angular_momentum_c+1, 2):
                lower_angular_momentum_d = max(
                    abs(angular_momentum_b-angular_momentum_two),
                    abs(angular_momentum_c-angular_momentum_four))
                starting_angular_momentum_d = (lower_angular_momentum_d
                                               + (angular_momentum_b+lower_angular_momentum_d
                                                  + angular_momentum_two)%2)
                upper_angular_momentum_d = min(
                    oscillator_quanta_two-angular_momentum_b,
                    oscillator_quanta_four-angular_momentum_c,
                    angular_momentum_b+angular_momentum_two,
                    angular_momentum_c+angular_momentum_four)
                ending_angular_momentum_d = (upper_angular_momentum_d
                                             - (angular_momentum_b+upper_angular_momentum_d
                                                + angular_momentum_two)%2)
                wigner_coefficient_one = _wigner_three_j_prefactor(
                    angular_momentum_a, angular_momentum_c, angular_momentum_one)
                for angular_momentum_d in range(starting_angular_momentum_d,
                                                ending_angular_momentum_d+1, 2):
                    remaining_oscillator_quanta_one = (oscillator_quanta_four
                                                       - angular_momentum_c
                                                       - angular_momentum_d)
                    remaining_oscillator_quanta_two = (oscillator_quanta_two
                                                       - angular_momentum_b
                                                       - angular_momentum_d)
                    maximum_radial_quantum_number = min(
                        int(remaining_oscillator_quanta_one/2),
                        int(remaining_oscillator_quanta_two/2))
                    if maximum_radial_quantum_number < 0:
                        continue
                    wigner_coefficient_two = _wigner_three_j_prefactor(
                        angular_momentum_b, angular_momentum_d, angular_momentum_two)
                    wigner_coefficient_four = _wigner_three_j_prefactor(
                        angular_momentum_c, angular_momentum_d, angular_momentum_four)
                    if min(coupling_angular_momentum, angular_momentum_a) > angular_momentum_three:
                        wigner_nine_j = (_wigner_nine_j_symbol(
                            angular_momentum_one, angular_momentum_c, angular_momentum_a,
                            angular_momentum_two, angular_momentum_d, angular_momentum_b,
                            coupling_angular_momentum, angular_momentum_four,
                            angular_momentum_three)
                            * INTEGER_SIGNS[angular_momentum_one
                                             + angular_momentum_two
                                             + coupling_angular_momentum])
                    else:
                        wigner_nine_j = _wigner_nine_j_symbol(
                            angular_momentum_a, angular_momentum_c, angular_momentum_one,
                            angular_momentum_b, angular_momentum_d, angular_momentum_two,
                            angular_momentum_three, angular_momentum_four,
                            coupling_angular_momentum)
                    for radial_quantum_number_d in range(maximum_radial_quantum_number+1):
                        transformed_oscillator_quanta_d = (2*radial_quantum_number_d
                                                           + angular_momentum_d)
                        radial_quantum_number_a = int((oscillator_quanta_three
                                                      - oscillator_quanta_two
                                                      + transformed_oscillator_quanta_d
                                                      - angular_momentum_a)/2)
                        radial_quantum_number_b = int((oscillator_quanta_two
                                                      - transformed_oscillator_quanta_d
                                                      - angular_momentum_b)/2)
                        radial_quantum_number_c = int((oscillator_quanta_four
                                                      - transformed_oscillator_quanta_d
                                                      - angular_momentum_c)/2)
                        if (radial_quantum_number_a < 0 or radial_quantum_number_b < 0
                                or radial_quantum_number_c < 0):
                            continue
                        degeneracy_factor = ((2*angular_momentum_a+1)
                                             *(2*angular_momentum_b+1)
                                             *(2*angular_momentum_c+1)
                                             *(2*angular_momentum_d+1))
                        gamma_factor = (
                            INVERSE_FACTORIALS[radial_quantum_number_a]
                            / GAMMA_HALF_INTEGER_SQUARE_ROOTS[
                                radial_quantum_number_a+angular_momentum_a+1]**2
                            * INVERSE_FACTORIALS[radial_quantum_number_b]
                            / GAMMA_HALF_INTEGER_SQUARE_ROOTS[
                                radial_quantum_number_b+angular_momentum_b+1]**2
                            * INVERSE_FACTORIALS[radial_quantum_number_c]
                            / GAMMA_HALF_INTEGER_SQUARE_ROOTS[
                                radial_quantum_number_c+angular_momentum_c+1]**2
                            * INVERSE_FACTORIALS[radial_quantum_number_d]
                            / GAMMA_HALF_INTEGER_SQUARE_ROOTS[
                                radial_quantum_number_d+angular_momentum_d+1]**2)
                        result += (INTEGER_SIGNS[angular_momentum_d]*degeneracy_factor
                                   *wigner_coefficient_one*wigner_coefficient_two
                                   *wigner_coefficient_three*wigner_coefficient_four
                                   *wigner_nine_j*gamma_factor*normalization_factor)
    result = phase_factor*result*oscillator_scale_factor/math.pi
    return 0.0 if abs(result) < 1e-10 else result


@lru_cache(maxsize=65536)
def talmi_moshinsky_bracket(radial_quantum_number_one,
                            orbital_angular_momentum_one,
                            radial_quantum_number_two,
                            orbital_angular_momentum_two,
                            radial_quantum_number_three,
                            orbital_angular_momentum_three,
                            radial_quantum_number_four,
                            orbital_angular_momentum_four,
                            coupling_angular_momentum):
    r"""计算无量纲谐振子 Talmi-Moshinsky 变换系数。

    The coefficient transforms a coupled product basis
    $|(n_1l_1)(n_2l_2);L\rangle$ to relative and center-of-mass
    oscillator coordinates $|(n_3l_3)(n_4l_4);L\rangle$.

    Args:
        radial_quantum_number_one, radial_quantum_number_two (int):
            Single-particle radial quantum numbers.
        orbital_angular_momentum_one, orbital_angular_momentum_two (int):
            Single-particle orbital angular momenta.
        radial_quantum_number_three, radial_quantum_number_four (int):
            Transformed radial quantum numbers.
        orbital_angular_momentum_three, orbital_angular_momentum_four (int):
            Transformed orbital angular momenta.
        coupling_angular_momentum (int): Total coupled orbital momentum.

    Returns:
        float: Talmi-Moshinsky transformation coefficient.
    """
    values = (radial_quantum_number_one, orbital_angular_momentum_one,
              radial_quantum_number_two, orbital_angular_momentum_two,
              radial_quantum_number_three, orbital_angular_momentum_three,
              radial_quantum_number_four, orbital_angular_momentum_four,
              coupling_angular_momentum)
    if any(value < 0 or int(value) != value for value in values):
        raise ValueError('Talmi--Moshinsky indices must be nonnegative integers.')
    if (2*radial_quantum_number_one + orbital_angular_momentum_one
            + 2*radial_quantum_number_two + orbital_angular_momentum_two > 40):
        raise ValueError('Talmi--Moshinsky tables support total oscillator quanta <= 40.')
    return _talmi_moshinsky_bracket(*values)


def racah_coefficient(first_angular_momentum, second_angular_momentum,
                      third_angular_momentum, first_orbital_angular_momentum,
                      second_orbital_angular_momentum, third_orbital_angular_momentum):
    r"""计算 Wigner-Racah 角动量重耦合系数。

    Args:
        first_angular_momentum, second_angular_momentum, third_angular_momentum:
            Integer or half-integer angular momenta in doubled-integer
            source convention.
        first_orbital_angular_momentum, second_orbital_angular_momentum,
        third_orbital_angular_momentum: Orbital angular momenta.

    Returns:
        float: The corresponding $W(abcd;ef)$ coefficient. It is evaluated
            as the finite Racah sum
            $W=\Delta_1\Delta_2\Delta_3\Delta_4\sum_z(-1)^z(z+1)!/D_z$.
    """
    return _racah_coefficient(first_angular_momentum, second_angular_momentum,
                              third_angular_momentum, first_orbital_angular_momentum,
                              second_orbital_angular_momentum,
                              third_orbital_angular_momentum)


def wigner_nine_j_symbol(first_row_first, first_row_second, first_row_third,
                         second_row_first, second_row_second, second_row_third,
                         third_row_first, third_row_second, third_row_third):
    r"""计算角动量耦合中的 Wigner $9j$ 符号。

    Args:
        first_row_first, first_row_second, first_row_third: First row entries.
        second_row_first, second_row_second, second_row_third: Second row entries.
        third_row_first, third_row_second, third_row_third: Third row entries.

    Returns:
        float: The angular-momentum recoupling symbol, namely
            $\left\{\begin{smallmatrix}j_{11}&j_{12}&j_{13}\\
            j_{21}&j_{22}&j_{23}\\j_{31}&j_{32}&j_{33}\end{smallmatrix}\right\}$.
    """
    return _wigner_nine_j_symbol(first_row_first, first_row_second, first_row_third,
                                 second_row_first, second_row_second, second_row_third,
                                 third_row_first, third_row_second, third_row_third)


# Original short names remain available for source compatibility.
talmos = talmi_moshinsky_bracket
racah0 = racah_coefficient
sninj0 = wigner_nine_j_symbol


@lru_cache(maxsize=24)
def _separable_pairing_brackets(maximum_fermion_shell):
    r"""Construct the basis transform used by the separable force.

    Each row stores $P_N=\langle nn|N0;L=0\rangle$ and its transformed
    oscillator quantum number, with $N=2n+l$ conserved.
    """
    rows = []
    for total_angular_momentum in range(1, maximum_fermion_shell+2):
        for parity_sign in (1, -1):
            orbital_angular_momentum = total_angular_momentum - (
                1-parity_sign*((-1)**total_angular_momentum))//2
            if orbital_angular_momentum > maximum_fermion_shell:
                continue
            maximum_radial_quantum_number = ((maximum_fermion_shell
                                              - orbital_angular_momentum)//2+1)
            for second_radial_index in range(1, maximum_radial_quantum_number+1):
                for first_radial_index in range(second_radial_index,
                                                maximum_radial_quantum_number+1):
                    maximum_transformed_oscillator_quanta = (
                        first_radial_index+second_radial_index-2
                        + orbital_angular_momentum)
                    row = np.zeros((maximum_fermion_shell+1, 2))
                    for transformed_oscillator_quanta in range(
                            maximum_transformed_oscillator_quanta+1):
                        complementary_oscillator_quanta = (
                            maximum_transformed_oscillator_quanta
                            - transformed_oscillator_quanta)
                        row[transformed_oscillator_quanta] = (
                            _talmi_moshinsky_bracket(
                                first_radial_index-1, orbital_angular_momentum,
                                second_radial_index-1, orbital_angular_momentum,
                                transformed_oscillator_quanta, 0,
                                complementary_oscillator_quanta, 0, 0)
                            *((-1)**orbital_angular_momentum)
                            / math.sqrt(2*orbital_angular_momentum+1),
                            complementary_oscillator_quanta)
                    rows.append(row)
    return np.asarray(rows)


def separable_pairing_form_factor(state):
    r"""计算有限程可分离配对相互作用的径向 form factor。

    The form factor is evaluated in the oscillator basis with length
    $b_0$ and interaction range $a=\sqrt{\texttt{state.gal}}$;
    the shell dependence is carried by factorial and half-integer gamma
    factors and the geometric ratio $(b_0^2-a^2)/(b_0^2+a^2)$.

    Args:
        state: Initialized finite-nucleus state with oscillator dimensions.

    Returns:
        numpy.ndarray: Dimensionful radial factors for oscillator shells.
    """
    radial_quantum_numbers = np.arange(state.n0fx+1)
    oscillator_length = state.b0
    pairing_range_squared = state.gal
    return (math.sqrt(2)/math.pi/2**.75 * oscillator_length**-1.5
            * GAMMA_HALF_INTEGER_SQUARE_ROOTS[radial_quantum_numbers+1]
            * INVERSE_FACTORIAL_SQUARE_ROOTS[radial_quantum_numbers]
            *(oscillator_length*oscillator_length
              /(oscillator_length*oscillator_length+pairing_range_squared))**1.5
            *((oscillator_length*oscillator_length-pairing_range_squared)
              /(oscillator_length*oscillator_length+pairing_range_squared))**radial_quantum_numbers)


def initialize_separable_pairing(state):
    r"""将径向 form factor 和 Talmi-Moshinsky 系数组合为配对矩阵元。

    Each stored element combines the Talmi-Moshinsky transform with the
    radial form factor, $V_{nN}\propto\sum_N\langle nn|NN\rangle P_N$.

    Args:
        state: Initialized finite-nucleus state and pairing basis storage.

    Returns:
        None: Fills ``state.vn`` in place.
    """
    transformation_brackets = _separable_pairing_brackets(state.n0f)
    radial_form_factor = separable_pairing_form_factor(state)
    state.vn.fill(0)
    state.vn[1:state.mv+1, :state.n0f+1] = (
        transformation_brackets[:, :, 0]
        * radial_form_factor[transformation_brackets[:, :, 1].astype(int)])


# Original short names remain available for source compatibility.
vn0 = separable_pairing_form_factor
singd = initialize_separable_pairing
