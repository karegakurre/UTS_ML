"""Utilitas bersama. Data dibaca dari DATA_DIR (default: ./data berisi file *.csv.gz dari data_uts_ml.zip)."""
import os, json, numpy as np, pandas as pd
SEED = 42
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(__file__), "..", "hasil"))
ROOT = os.path.join(os.path.dirname(__file__), "..")
os.makedirs(OUT_DIR, exist_ok=True)

def load(name):
    return pd.read_csv(os.path.join(DATA_DIR, name + ".csv.gz"))

def save_json(obj, name):
    with open(os.path.join(OUT_DIR, name), "w") as f:
        json.dump(obj, f, indent=2, default=float)

def pinball(y, q_pred, q):
    d = y - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))

def log(*a):
    print(*a, flush=True)
