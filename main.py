"""DeepDTA (Davis + KIBA) unified backend: FastAPI + ONNX Runtime."""
import json
import re
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).parent

opts = ort.SessionOptions()
opts.intra_op_num_threads = 1


def load_model_bundle(prefix: str, defaults: dict) -> dict:
    cfg = json.loads((BASE / f"{prefix}_config.json").read_text())
    examples = json.loads((BASE / f"{prefix}_examples.json").read_text())
    metrics = json.loads((BASE / f"{prefix}_metrics.json").read_text())
    test_points = json.loads((BASE / f"{prefix}_test_points.json").read_text())
    model_path = BASE / f"deepdta_{prefix}.onnx"
    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    return {
        "name": defaults["name"],
        "unit": cfg.get("unit", defaults["unit"]),
        "binds_cutoff": float(cfg.get("binds_cutoff", defaults["binds_cutoff"])),
        "weak_below": defaults["weak_below"],
        "scale_min": defaults["scale_min"],
        "scale_max": defaults["scale_max"],
        "max_smiles": int(cfg["max_smiles_length"]),
        "max_protein": int(cfg["max_protein_length"]),
        "smiles_vocab": cfg["smiles_vocab"],
        "protein_vocab": cfg["protein_vocab"],
        "smiles_style": cfg.get("smiles_style", defaults["smiles_style"]),
        "session": session,
        "examples": examples,
        "metrics": metrics,
        "test_points": test_points,
    }


MODELS = {
    "Davis": load_model_bundle(
        "davis",
        {
            "name": "Davis",
            "unit": "pKd",
            "binds_cutoff": 7.0,
            "weak_below": 6.0,
            "scale_min": 5.0,
            "scale_max": 10.5,
            "smiles_style": {
                "lowercase_aromatic": False,
                "brackets": False,
                "stereo_chars": False,
                "plus_or_minus": False,
            },
        },
    ),
    "KIBA": load_model_bundle(
        "kiba",
        {
            "name": "KIBA",
            "unit": "KIBA",
            "binds_cutoff": 12.1,
            "weak_below": 11.2,
            "scale_min": 9.0,
            "scale_max": 15.5,
            "smiles_style": {
                "lowercase_aromatic": False,
                "brackets": True,
                "stereo_chars": False,
                "plus_or_minus": True,
            },
        },
    ),
}

app = FastAPI(title="DeepDTA Binding Affinity Studio")


class PredictRequest(BaseModel):
    dataset: Literal["Davis", "KIBA"] = "Davis"
    smiles: str = Field(..., min_length=1, max_length=2500)
    protein: str = Field(..., min_length=1, max_length=8000)


def clean_smiles(text: str) -> str:
    return re.sub(r"\s+", "", text)


def clean_protein(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith(">")]
    return re.sub(r"\s+", "", "".join(lines)).upper()


def encode(sequence: str, vocab: dict, max_len: int) -> np.ndarray:
    out = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(sequence[:max_len]):
        out[i] = vocab.get(ch, 0)
    return out


def interpret_score(dataset: str, score: float, weak_below: float, cutoff: float) -> str:
    unit = MODELS[dataset]["unit"]
    if score < weak_below:
        return f"Weak predicted binding (< {weak_below} {unit})"
    if score < cutoff:
        return f"Moderate predicted binding ({weak_below}–{cutoff} {unit})"
    return f"Strong predicted binding (≥ {cutoff} {unit})"


@app.get("/metadata")
def metadata():
    """Return configs, examples, benchmark metrics, and test points for both models."""
    out = {}
    for key, m in MODELS.items():
        out[key] = {
            "name": m["name"],
            "unit": m["unit"],
            "binds_cutoff": m["binds_cutoff"],
            "weak_below": m["weak_below"],
            "scale_min": m["scale_min"],
            "scale_max": m["scale_max"],
            "max_smiles": m["max_smiles"],
            "max_protein": m["max_protein"],
            "examples": m["examples"],
            "metrics": m["metrics"],
            "test_points": m["test_points"],
        }
    return out


@app.post("/predict")
def predict(req: PredictRequest):
    if req.dataset not in MODELS:
        raise HTTPException(status_code=400, detail="Unsupported dataset")

    m = MODELS[req.dataset]
    smiles = clean_smiles(req.smiles)
    protein = clean_protein(req.protein)
    warnings = []

    if not smiles or not protein:
        return {"error": "Please enter both a drug SMILES string and a protein sequence."}

    bad_s = sorted({c for c in smiles if c not in m["smiles_vocab"]})
    bad_p = sorted({c for c in protein if c not in m["protein_vocab"]})

    if bad_s:
        msg = (
            f"Unrecognized SMILES character(s) ignored by the {req.dataset} model: "
            + " ".join(bad_s)
            + ". The prediction may be unreliable."
        )
        if any(c in "cnosp@/\\" for c in bad_s) or (
            req.dataset == "Davis" and any(c in "[]+-" for c in bad_s)
        ):
            msg += (
                f" The {req.dataset} model expects canonical SMILES with uppercase atom symbols "
                "(e.g., C1=CC=CC=C1 instead of c1ccccc1)."
            )
        warnings.append(msg)

    if bad_p:
        warnings.append(
            f"Unrecognized amino acid character(s) ignored by the {req.dataset} model: "
            + " ".join(bad_p)
            + ". The prediction may be unreliable."
        )

    if len(smiles) > m["max_smiles"]:
        warnings.append(
            f"SMILES exceeds {m['max_smiles']} characters; truncated to the first {m['max_smiles']} characters."
        )
    if len(protein) > m["max_protein"]:
        warnings.append(
            f"Protein exceeds {m['max_protein']} residues; truncated to the first {m['max_protein']} residues."
        )

    d = encode(smiles, m["smiles_vocab"], m["max_smiles"])[None, :]
    p = encode(protein, m["protein_vocab"], m["max_protein"])[None, :]

    input_names = [inp.name for inp in m["session"].get_inputs()]
    feed = {input_names[0]: d, input_names[1]: p}
    val = float(m["session"].run(None, feed)[0][0][0])

    return {
        "dataset": req.dataset,
        "unit": m["unit"],
        "value": round(val, 3),
        "binds_cutoff": m["binds_cutoff"],
        "weak_below": m["weak_below"],
        "label": interpret_score(req.dataset, val, m["weak_below"], m["binds_cutoff"]),
        "warnings": warnings,
    }


@app.get("/health")
def health():
    return {"status": "ok", "models": list(MODELS.keys())}


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")
