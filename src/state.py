# module state

import numpy as np
import pandas as pd

class State:
    def __init__(self, name: str, consts: pd.DataFrame) -> None:
        self.name:          str              = name
        self.consts:        dict[str, float] = consts.loc[self.name].to_dict()
        self.cross_section: float            = np.pi * (2 * self.consts['rad'])**2
