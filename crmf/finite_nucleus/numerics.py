"""Quantum-number helpers for the spherical Dirac oscillator basis."""


def large_component_orbital(kappa):
    r"""Return orbital angular momentum $l$ for the large component.

    In the Dirac spherical basis, :math:`l=\kappa` for positive
    :math:`\kappa` and :math:`l=-\kappa-1` otherwise.
    """
    return kappa if kappa > 0 else -kappa - 1


def small_component_orbital(kappa):
    r"""Return orbital angular momentum $l$ for the small component.

    The spinor partner has :math:`l=\kappa-1` for positive
    :math:`\kappa` and :math:`l=-\kappa` otherwise.
    """
    return kappa - 1 if kappa > 0 else -kappa


def component_orbital(kappa, component):
    r"""Select $l$ for component 1 (large) or 2 (small).

    Equivalently, :math:`l=l_{\rm large}` when ``component == 1`` and
    :math:`l=l_{\rm small}` for the lower Dirac component.
    """
    return large_component_orbital(kappa) if component == 1 else small_component_orbital(kappa)


# Original short names remain available for translated source routines.
lfkap = large_component_orbital
lgkap = small_component_orbital
lfgkap = component_orbital
