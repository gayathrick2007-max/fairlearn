"""Run once:  python train.py   -> writes artifacts/results.json and artifacts/models.joblib"""
import json, time, joblib
from pathlib import Path
from pipeline import build_results

t = time.time()
results, models = build_results()
out = Path(__file__).parent / "artifacts"; out.mkdir(exist_ok=True)
(out / "results.json").write_text(json.dumps(results, indent=2))
joblib.dump(models, out / "models.joblib", compress=3)
print(f"done in {time.time()-t:.0f}s")
