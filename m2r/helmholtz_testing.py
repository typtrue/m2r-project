import scipy as sci
import numpy as np
from typing import Sequence
import sympy as sp


class HelmholtzSystem:
    def __init__(self, wave_speed: float, source_strength: float, plate_length: float = 2, plate_center: tuple[float, float] = (0, 0)) -> None:
        self.c = wave_speed
        self.source_strength = source_strength
        self.plate_length = plate_length
        self.plate_center = plate_center
        l = plate_center[0] - plate_length # noqa e741
        r = plate_center[0] + plate_length
        t = plate_center[1] + plate_length
        b = plate_center[1] - plate_length
        self.domain = (b, t, l, r)

    def kirchoff_helmholtz(self, reciever: tuple[float, float], surface, n):
        x, y = reciever
        Q = self.source_strength
        domain = self.domain
        r_Q = y - self.domain[0]
        v_S = 0

        k = 1

        dx = (domain[3] - domain[2]) / n
        dy = (domain[1] - domain[0]) / n


        I1 = Q * np.exp(1j * k * r_Q) / r_Q

        # I2 = v_S * np.exp(
