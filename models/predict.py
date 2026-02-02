import joblib
from models.featurizer import featurize_formula

band_gap_model = joblib.load("models/formation_energy_per_atom_model.pkl")
density_model = joblib.load("models/density.pkl")
volume_model = joblib.load("models/volume.pkl")

def predict_properties(formula: str):
    X = featurize_formula(formula)

    return {
        "formation energy": float(band_gap_model.predict(X)[0]),
        "density": float(density_model.predict(X)[0]),
        "volume": float(volume_model.predict(X)[0]),
    }



    