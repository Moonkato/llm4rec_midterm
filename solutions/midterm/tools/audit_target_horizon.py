"""No-fit sensitivity of the top-five-count proxy to forecast horizon."""
import json, sqlite3, time
import numpy as np
from run_deep_temporal import BASE, full_truth

def main():
    out=BASE/'evidence/target_horizon_diagnostic.json'
    if out.exists():raise SystemExit('Preserve completed diagnostic')
    started=time.perf_counter()
    with np.load(BASE/'cache/context_features/eval.npz') as z:users=list(map(str,z['users']))
    db=sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro',uri=True)
    db.execute('PRAGMA cache_size=-32000');results=[]
    for cutoff,short_end,long_end in [('2020-04-13','2020-05-18','2020-06-22'),('2020-05-18','2020-06-22','2020-07-27')]:
        short=full_truth(db,cutoff,short_end)['top5_count80']
        long=full_truth(db,cutoff,long_end)['top5_count80']
        changed=0;both=0;jaccard=[];retained=[];size_both5=0;changed_both5=0
        for u in users:
            a,b=short.get(u,set()),long.get(u,set())
            changed+=a!=b
            if a and b:
                both+=1;jaccard.append(len(a&b)/len(a|b));retained.append(len(a&b)/len(a))
            if len(a)==len(b)==5:
                size_both5+=1;changed_both5+=a!=b
        results.append({'cutoff':cutoff,'short_end':short_end,'long_end':long_end,'short_days':35,'long_days':70,
          'users':len(users),'changed_top5_sets':changed,'both_nonempty_users':both,
          'mean_jaccard_both_nonempty':float(np.mean(jaccard)),'mean_short_top5_retained_in_long':float(np.mean(retained)),
          'both_exactly5_users':size_both5,'changed_sets_among_both_exactly5':changed_both5})
    db.close()
    out.write_text(json.dumps({'results':results,'fits':0,'submissions':0,'wall_seconds':time.perf_counter()-started,
      'interpretation_limit':'Same cutoff and users, nested35/70day realized horizons. Measures proxy label sensitivity, not causal effect on model accuracy. Longer horizon includes later calendar weeks and activity changes. Official target formula remains unknown; no12week holdout claim.'},indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
