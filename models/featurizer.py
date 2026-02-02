import joblib
import pandas as pd
import numpy as np
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element

# Load training feature template
FEATURE_COLUMNS = joblib.load(r"C:\Materials and it's Mechanical Properties\Data\feature_columns.pkl")


def featurize_formula(formula: str) -> pd.DataFrame:
    comp = Composition(formula)
    total_atoms = comp.num_atoms

    features = {}

    # ---------- Global features ----------
    features["n_elements"] = len(comp.elements)

    features["contains_transition_metal"] = bool(
        any(el.is_transition_metal for el in comp.elements)
    )

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
    # for el in comp.elements:
    #     features[str(el)] = comp.get_atomic_fraction(el)

    # ---------- Build dataframe with correct schema ----------
    X = pd.DataFrame([features])

    # Add missing columns (VERY IMPORTANT)
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0.0

    # Enforce correct order
    X = X[FEATURE_COLUMNS]

    return X


