import csv,json,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).parents[1]
def test_fixture_metrics_and_control():
 with tempfile.TemporaryDirectory() as d:
  subprocess.run(['python3','src/claim1_synthetic_grod_toy.py','--out',d],cwd=ROOT,check=True)
  s=json.load(open(Path(d)/'summary.json')); rows=list(csv.DictReader(open(Path(d)/'results.csv')))
  assert s['verdict']=='toy' and len(rows)==3
  assert all(float(r['synthetic_after_filter']) < float(r['synthetic_before_filter']) for r in rows)
  assert all(0 <= float(r['grod_auroc']) <= 1 and 0 <= float(r['grod_fpr95']) <= 1 for r in rows)
