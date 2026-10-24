"""Paired user bootstrap for the prespecified earlier-window logistic fit."""
import json,sys
from pathlib import Path
import numpy as np
from run_round1 import apk
b=Path(__file__).resolve().parents[1]
truth=json.loads((b/'evidence/strict_labels/truth.json').read_text());users=list(truth)
pred=json.loads((b/'evidence/simple_reranker/predictions.json').read_text())
base=json.loads((b/'evidence/schedule/predictions.json').read_text())['personal_count_schedule_power0.5']
d=np.array([apk(truth[u],pred[u])-apk(truth[u],base[u]) for u in users])
rng=np.random.default_rng(20261022);means=np.array([rng.choice(d,size=len(d),replace=True).mean() for _ in range(2000)])
result={'comparison':'frozen earlier-window logistic minus sqrt schedule count','mean_map5_difference':float(d.mean()),'paired_user_bootstrap_95pct':np.quantile(means,[.025,.975]).tolist(),'users_improved':int((d>0).sum()),'users_worsened':int((d<0).sum()),'users_unchanged':int((d==0).sum()),'seed':20261022,'resamples':2000,'limitation':'conditional on this temporal window and chosen models; not uncertainty for Kaggle, not correction for previous model selection'}
(b/'evidence/simple_reranker/paired_comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
