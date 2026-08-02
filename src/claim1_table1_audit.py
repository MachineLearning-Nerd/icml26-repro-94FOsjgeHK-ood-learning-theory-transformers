"""Source-faithful Claim-1 table transcription audit; no model training."""
import csv,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
# Directly transcribed from pinned source Table 1 (ImageNet-200 row; FPR@95 percent).
ROWS=[('baseline',21.97),('GROD',0.12)]
def main(out=ROOT/'outputs/claim1_source_audit'):
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
 with open(out/'results.csv','w',newline='') as f:
  w=csv.writer(f);w.writerow(['method','fpr95_percent']);w.writerows(ROWS)
 baseline=dict(ROWS)['baseline'];grod=dict(ROWS)['GROD']
 d={'claim':'GROD reduces FPR@95 from 21.97% to 0.12% (Table 1).','source_location':'content/6_experiment_icml2026.tex Table 1','baseline_fpr95_percent':baseline,'grod_fpr95_percent':grod,'absolute_reduction_percentage_points':round(baseline-grod,2),'verdict':'inconclusive','scope':'Deterministic primary-source table transcription and arithmetic audit only; no independent ImageNet-200/GROD execution, hence not verification.'}
 (out/'summary.json').write_text(json.dumps(d,indent=2)+'\n')
if __name__=='__main__':main()
