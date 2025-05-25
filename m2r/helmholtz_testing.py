import scipy as sci
import numpy as np
from typing import Sequence
import sympy as sp
from funcs import hankel_estimate


class HelmholtzSystem:
    def __init__(self, wave_no: float) -> None:
        self.k = wave_no

        y = sp.symbols('y')
        self.w_inc = np.exp(1.0j * wave_no * y)

        self.dw_scat = -1.0j * wave_no

    def G(self, x, y):
        return hankel_estimate(0, self.k * np.abs(x - y)) / 4.0j
    
    def DG(self, x, y):
        return self.k * hankel_estimate(1, self.k * np.abs(x - y)) / 4.0j