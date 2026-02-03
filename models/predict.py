import joblib
from models.featurizer import featurize_formula

band_gap_model = joblib.load("models/band_gap.pkl")
density_model = joblib.load("models/density.pkl")
volume_model = joblib.load("models/volume.pkl")
semiconductor = joblib.load("models/semiconductor.pkl")
energy_per_atom = joblib.load("models/energy_per_atom.pkl")
formation_energy = joblib.load("models/formation_energy_per_atom_model.pkl")

import xgboost as xgb

def predict_properties(formula: str):
    X = featurize_formula(formula)

    dX = xgb.DMatrix(X, feature_names=X.columns.tolist())

    return {
        "Band gap": float(band_gap_model.get_booster().predict(dX)[0]),
        "Density": float(density_model.get_booster().predict(dX)[0]),
        "Volume": float(volume_model.get_booster().predict(dX)[0]),
        "Semiconductor": bool(semiconductor.get_booster().predict(dX)[0] > 0.5),
        "Energy per atom": float(energy_per_atom.get_booster().predict(dX)[0]),
        "Formation energy per atom":float(formation_energy.get_booster().predict(dX)[0]) 
    }



    