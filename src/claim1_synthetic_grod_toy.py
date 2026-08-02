"""Local reduced feature-space GROD fixture; not CIFAR/ImageNet or a transformer run."""
import argparse,csv,json,math,random
from pathlib import Path

def dot(a,b): return sum(x*y for x,y in zip(a,b))
def norm(a): return math.sqrt(dot(a,a))
def mean(xs): return [sum(x[i] for x in xs)/len(xs) for i in range(2)]
def inv2(c):
 d=c[0][0]*c[1][1]-c[0][1]*c[1][0]
 return [[c[1][1]/d,-c[0][1]/d],[-c[1][0]/d,c[0][0]/d]]
def maha(x,mu,iv):
 z=[x[i]-mu[i] for i in range(2)];return dot(z,[iv[0][0]*z[0]+iv[0][1]*z[1],iv[1][0]*z[0]+iv[1][1]*z[1]])
def auc(scores,labels):
 # OOD positive; exact pairwise ranking (small retained fixture).
 p=[s for s,y in zip(scores,labels) if y]; n=[s for s,y in zip(scores,labels) if not y]
 return sum((a>b)+.5*(a==b) for a in p for b in n)/(len(p)*len(n))
def fpr95(scores,labels):
 p=sorted(s for s,y in zip(scores,labels) if y); threshold=p[max(0,math.ceil(.05*len(p))-1)]
 n=[s for s,y in zip(scores,labels) if not y];return sum(s>=threshold for s in n)/len(n)
def sigmoid(z): return 1/(1+math.exp(-max(-50,min(50,z))))
def run(seed,c):
 r=random.Random(seed); centers=[[-1.5,0.0],[1.5,0.0]]
 def sample(mu,n,sd=.55): return [[r.gauss(mu[0],sd),r.gauss(mu[1],sd)] for _ in range(n)]
 train=[(x,k) for k,mu in enumerate(centers) for x in sample(mu,c['n_train_per_class'])]
 by=[[x for x,y in train if y==k] for k in range(2)]; mus=[mean(z) for z in by]
 # Source-aligned feature-level outward PCA/LDA center rule: boundary point direction from global mean.
 glob=mean([x for x,_ in train]); syn=[]
 for mu in mus:
  v=[mu[i]-glob[i] for i in range(2)]; q=norm(v); out=[mu[i]+c['outward_shift']*v[i]/q for i in range(2)]
  syn+=sample(out,c['synthetic_per_center'],.55)
 # Mahalanobis filter removes samples ID-like under every class; source appendix uses this concept.
 cov=[[.55*.55,0],[0,.55*.55]]; iv=inv2(cov)
 kept=[x for x in syn if min(maha(x,mu,iv) for mu in mus)>=c['mahalanobis_filter_min']]
 # Binary ID/OOD logistic fine-tune on [x1,x2,||x||,1]; baseline is nearest-ID Mahalanobis score.
 X=[x+[norm(x),1.] for x,_ in train]+[x+[norm(x),1.] for x in kept]; Y=[0]*len(train)+[1]*len(kept); w=[0.0]*4
 for _ in range(c['epochs']):
  g=[0.0]*4
  for x,y in zip(X,Y):
   z=sigmoid(dot(w,x))-y
   for j in range(4):g[j]+=z*x[j]
  for j in range(4):w[j]-=c['learning_rate']*g[j]/len(X)
 test_id=[x for mu in centers for x in sample(mu,c['n_test_per_class'])]
 # unseen real OOD ring/upper distribution, never used in synthetic training
 test_ood=sample([0,4.8],c['n_ood_test'],.75)
 test=test_id+test_ood; labels=[0]*len(test_id)+[1]*len(test_ood)
 grod=[sigmoid(dot(w,x+[norm(x),1.])) for x in test]
 base=[min(maha(x,mu,iv) for mu in mus) for x in test]
 return {'seed':seed,'synthetic_before_filter':len(syn),'synthetic_after_filter':len(kept),'baseline_auroc':auc(base,labels),'baseline_fpr95':fpr95(base,labels),'grod_auroc':auc(grod,labels),'grod_fpr95':fpr95(grod,labels),'weights':w,'raw':{'id':test_id,'ood':test_ood,'synthetic_kept':kept}}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/claim1_synthetic_grod_toy.json');ap.add_argument('--out',default='outputs/claim1_synthetic_grod_toy');a=ap.parse_args();c=json.load(open(a.config));o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
 rows=[run(s,c) for s in c['seeds']]
 # raw supports direct metric recomputation.
 with open(o/'raw.json','w') as f:json.dump(rows,f,sort_keys=True)
 keys=['seed','synthetic_before_filter','synthetic_after_filter','baseline_auroc','baseline_fpr95','grod_auroc','grod_fpr95']
 with open(o/'results.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:x[k] for k in keys} for x in rows])
 avg={k:sum(x[k] for x in rows)/len(rows) for k in keys[3:]}
 summary={'verdict':'toy','scope':'Reduced 2-D feature-space GROD generator/filter/binary-loss fixture; not ViT, CIFAR, ImageNet, or Table-1 reproduction.','config':c,'mean_metrics':avg,'control':'nearest-ID Mahalanobis baseline; held-out OOD is never used for synthetic training. This separable fixture does not claim a GROD improvement over the baseline.','raw_artifact':'raw.json'}
 with open(o/'summary.json','w') as f:json.dump(summary,f,indent=2,sort_keys=True);f.write('\n')
if __name__=='__main__':main()
