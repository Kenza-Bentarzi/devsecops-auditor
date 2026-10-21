"""
Smoke tests for Agent 1. Run from the project root:
    python test_agent1.py
Uses synthetic NASA-like data in a temp folder and a temp model file, so it never
overwrites the real model (models/agent1_rf.joblib) and leaves no files behind.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from agents import agent1_metrics as a1

tmp = Path(tempfile.mkdtemp())
a1.ARTIFACT = tmp / "test_model.joblib"      # redirect model output away from models/
a1._bundle = None

rng = np.random.default_rng(0)
n = 600
loc = rng.integers(5, 300, n)
vg = (loc / 15 + rng.integers(1, 6, n)).round()
df = pd.DataFrame({
    "loc": loc, "v(g)": vg, "ev(g)": rng.integers(1, 10, n), "iv(G)": rng.integers(1, 12, n),
    "n": loc * 3, "v": loc * 12.5, "e": loc * loc * 2.0, "lOComment": rng.integers(0, 30, n),
    "uniq_Opnd": rng.integers(3, 60, n), "total_Opnd": rng.integers(10, 200, n),
    "branchCount": vg * 2,
})
# defects correlated with complexity (~10-20% positives)
p = 1 / (1 + np.exp(-(2.0 * (vg - vg.mean()) / vg.std() - 2.0)))
df["defects"] = np.where(rng.random(n) < p, "true", "false")
df.loc[rng.choice(n, 20, replace=False), "loc"] = np.nan   # missing values in training data
csv_path = tmp / "synthetic.csv"
df.to_csv(csv_path, index=False, na_rep="?")

# 1) training runs end-to-end and saves the artifact
a1.train(str(csv_path))
assert a1.ARTIFACT.exists(), "model artifact not saved"

# 2) valid input -> score in [0,1], status ok
sample = {"loc": 200, "v(g)": 25, "ev(g)": 5, "iv(G)": 8, "n": 600, "v": 2500.0,
          "e": 80000.0, "lOComment": 4, "uniq_Opnd": 30, "total_Opnd": 120, "branchCount": 50}
r = a1.predict_struct_risk(sample)
print("valid input   ->", r)
assert r["status"] == "ok" and 0.0 <= r["struct_risk_score"] <= 1.0
assert isinstance(r["high_risk"], bool)

# 3) complex code should score higher than trivial code
low = dict(sample, loc=8, **{"v(g)": 1, "branchCount": 1, "e": 100.0, "n": 20, "v": 90.0})
r_low = a1.predict_struct_risk(low)
print("trivial code  ->", r_low)
assert r["struct_risk_score"] > r_low["struct_risk_score"], "risk does not increase with complexity"

# 4) fail-safe: missing feature -> error + None (Agent 3 must map to REVIEW)
bad = {k: v for k, v in sample.items() if k != "v(g)"}
r_bad = a1.predict_struct_risk(bad)
print("missing field ->", r_bad)
assert r_bad["status"] == "error" and r_bad["struct_risk_score"] is None

# 5) fail-safe: non-numeric value -> error (must NOT be silently imputed)
r_nan = a1.predict_struct_risk(dict(sample, loc="abc"))
print("garbage value ->", r_nan)
assert r_nan["status"] == "error" and r_nan["struct_risk_score"] is None

# 6) fail-safe: model file missing -> error, no crash
a1._bundle = None
a1.ARTIFACT = tmp / "does_not_exist.joblib"
r_nomodel = a1.predict_struct_risk(sample)
print("no model file ->", r_nomodel)
assert r_nomodel["status"] == "error" and r_nomodel["struct_risk_score"] is None

print("\nALL TESTS PASSED")