import scipy as sci
import numpy as np
from typing import Iterable, Any
import sympy as sp


class PDE:
    def __init__(self, lhs: sp.Function, rhs: sp.Function) -> None:
        self.lhs = lhs
        self.rhs = rhs
        self.eq = sp.Eq(lhs, rhs)

    def solve(self) -> sp.Eq | sp.Rel | sp.Ne | None:
        pass


class WaveEquation (PDE):
    def __init__(self, c_val: float) -> None:
        self.c = c_val
        c, t, x, y, z = sp.symbols('c t x y z')
        u = sp.Function('u')(t, x, y, z)  # type: ignore
        lhs = sp.diff(u, t, 2)
        rhs = c**2 * (sp.diff(u, x, 2) + sp.diff(u, y, 2) + sp.diff(u, z, 2))  # type: ignore
        super().__init__(lhs, rhs)  # type: ignore

    def solve(self) -> sp.Eq | sp.Rel | sp.Ne | None:
        print(self.eq)
        t = sp.symbols('t')
        f = fourier_transform(self.eq, t)
        return f


def fourier_transform(equation: sp.Eq | sp.Rel | sp.Ne, var: sp.Symbol) -> sp.Eq | sp.Rel | sp.Ne | None:
    pass


testeq = WaveEquation(1)
testeq.solve()
