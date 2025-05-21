import scipy as sp
import numpy as np
from typing import Iterable


class Wave:
    def __init__(self, wavelength: float, pos: Iterable[float], dir: Iterable[float]) -> None:
        self.wavelength = wavelength
        self.pos = pos
        self.dir = dir
