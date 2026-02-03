import joblib
import pandas as pd
import numpy as np
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element

# Load training feature template
FEATURE_COLUMNS = ['contains_transition_metal', 'n_elements', 'avg_atomic_number', 'avg_atomic_mass', 'electronegativity_mean', 'electronegativity_std', 'Li', 'O', 'S', 'Sn', 'Be', 'B', 'Mg', 'N', 'Bi', 'Ba', 'K', 'Eu', 'Ni', 'Hg', 'Yb', 'Pb', 'Tb', 'Y', 'Ga', 'Ge', 'Tl', 'Rh', 'Si', 'Ag', 'Cu', 'Sr', 'In', 'Ho', 'Rb', 'Al', 'Au', 'Ca', 'As', 'P', 'Te', 'Cl', 'F', 'I', 'Br', 'Pt', 'U', 'Pa', 'Mn', 'Cd', 'Zn', 'Ir', 'Pd', 'Hf', 'C', 'Fe', 'Ti', 'Cr', 'Co', 'Na', 'Zr', 'V', 'Mo', 'Se', 'H', 'Sb', 'Cs', 'Nb', 'Nd', 'Er', 'Dy', 'Gd', 'Lu', 'La', 'Tm', 'Pm', 'Ac', 'Ce', 'Pr', 'Sm', 'W', 'Pu', 'Np', 'Re', 'Os', 'Ru', 'Ta', 'Sc', 'Tc', 'Th', 'Xe']

def featurize_formula(formula: str) -> pd.DataFrame:
    comp = Composition(formula)
    total_atoms = comp.num_atoms

    features = {}

    # ---------- Global features ----------
    features["contains_transition_metal"] = bool(
        any(el.is_transition_metal for el in comp.elements)
    )

    features["n_elements"] = len(comp.elements)

    features["avg_atomic_number"] = sum(
        el.Z * amt / total_atoms for el, amt in comp.items()
    )

    features["avg_atomic_mass"] = sum(
        el.atomic_mass * amt / total_atoms for el, amt in comp.items()
    )

    # Electronegativity
    xs, weights = [], []
    for el, amt in comp.items():
        if el.X is not None:
            xs.append(el.X)
            weights.append(amt)

    xs = np.array(xs)
    weights = np.array(weights)

    mean_x = np.average(xs, weights=weights)
    std_x = np.sqrt(np.average((xs - mean_x) ** 2, weights=weights))

    features["electronegativity_mean"] = mean_x
    features["electronegativity_std"] = std_x

    # ---------- Element fraction features ----------
    for el in comp.elements:
        features[str(el)] = comp.get_atomic_fraction(el)

    # ---------- Build dataframe with correct schema ----------
    X = pd.DataFrame([features])

    # Add missing columns (VERY IMPORTANT)
    for col in FEATURE_COLUMNS:
        if col not in  X.columns:
            X[col] = 0.0
        

    # Enforce correct order
    X = X[FEATURE_COLUMNS]

    return X








