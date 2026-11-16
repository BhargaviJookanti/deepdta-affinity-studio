# DeepDTA — Binding Affinity Studio

Unified FastAPI + ONNX Runtime web application supporting both **Davis** ($pK_d$) and **KIBA** binding affinity models.

## Setup
1. Ensure `deepdta_davis.onnx` and `deepdta_kiba.onnx` are placed in this root folder alongside `main.py`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run locally:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   Open `http://localhost:8000` in your browser.
