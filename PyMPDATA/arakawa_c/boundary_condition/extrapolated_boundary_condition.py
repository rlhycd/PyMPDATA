import numba
from ..enumerations import ARG_FOCUS, INNER, SIGN_LEFT
from functools import lru_cache

# TODO: 1D only


@lru_cache()
def _make_scalar(eps, at, halo):
    @numba.njit()
    def fill_halos_scalar(psi, n, sign):
        if sign == SIGN_LEFT:
            edg = halo - psi[ARG_FOCUS][INNER]
            nom = at(*psi, edg + 1) - at(*psi, edg)
            den = at(*psi, edg + 2) - at(*psi, edg + 1)
            cnst = nom / den if abs(den) > eps else 0
            return max(at(*psi, 1) - (at(*psi, 2) - at(*psi, 1)) * cnst, 0)
        else:
            edg = n + halo - 1 - psi[ARG_FOCUS][INNER]
            den = at(*psi, edg - 1) - at(*psi, edg - 2)
            nom = at(*psi, edg) - at(*psi, edg - 1)
            cnst = nom/den if abs(den) > eps else 0
            return max(at(*psi, - 1) + (at(*psi, -1) - at(*psi, -2)) * cnst, 0)
    return fill_halos_scalar


@lru_cache()
def _make_vector(at):
    @numba.njit()
    def fill_halos(psi, _, sign):
        return at(*psi, sign)
    return fill_halos


class ExtrapolatedBoundaryCondition:
    def __init__(self, eps=1e-10):
        self._eps = eps

    def make_scalar(self, at, halo):
        return _make_scalar(self._eps, at, halo)

    def make_vector(self, at):
        return _make_vector(at)
