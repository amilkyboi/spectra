# module molecule
"""
Contains the implementation of the Molecule class.
"""

import numpy as np
import pandas as pd

from atom import Atom


class Molecule:
    """
    A molecule.
    """

    def __init__(self, name: str, atom_1: str, atom_2: str) -> None:
        self.name: str = name
        self.atom_1: Atom = Atom(atom_1)
        self.atom_2: Atom = Atom(atom_2)
        self.consts: pd.DataFrame = pd.read_csv(
            f"../data/molecular_constants/{self.name}.csv", index_col=0
        )
        self.prediss: pd.DataFrame = pd.read_csv(f"../data/predissociation/{self.name}.csv")
        self.fc_data: np.ndarray = np.loadtxt(
            f"../data/franck-condon/{self.name}.csv", delimiter=","
        )
        self.molecular_mass: float = self.atom_1.mass + self.atom_2.mass
        self.reduced_mass: float = self.atom_1.mass * self.atom_2.mass / self.molecular_mass
