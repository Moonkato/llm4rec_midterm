"""Small logistic reranker trained on an earlier chronological window; NumPy/SciPy only."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import json,sqlite3,time,math
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from build_interactions import tick
from run_round1 import apk
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'evidence/simple_reranker'

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'results.json').exists():raise SystemExit('Completed experiment exists')
    protocol={'model':'L2 logistic regression, standardized features; scipy L-BFGS max80 iterations','l2':0.001,'training_origin':'2020-03-09','fit_cutoff':'2020-04-13','fit_labels_end':'2020-05-18','evaluation_cutoff':'2020-05-18','evaluation_end':'2020-07-27','labels':'single raw fragment >80% full airing; fragments clipped at window boundaries','candidates':'up to100 personal history programmes +30 popularity programmes, available in forecast','features':['log_count30','log_count80','log_decay60_count30','log_future_airings_per_day','log_past_airings_per_day','log_personal_schedule_score','log_pop_viewers','log_user_count','days_since_last_watch30'],'caveat':'same previously explored outer holdout; not independent confirmation. Artificial split fragments are not joined without session identity. No synthetic labels.','submissions':0}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    began=time.perf_counter();db=sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro',uri=True)
    origin=tick('2020-03-09 00:00:00');cat=json.loads((BASE/'evidence/round1/catalog.json').read_text());target_users=cat['users']
    def window(cut,end,training):
        hist=defaultdict(dict);truth=defaultdict(set)
        q='''SELECT o.user,e.item,e.duration,MAX(MAX(0,MIN(o.end,?)-MAX(o.start,?))),MAX(MIN(o.end,?)) FROM overlaps o JOIN events e USING(event_id) WHERE o.start<? AND o.end>? GROUP BY o.user,o.event_id'''
        for u,i,d,s,last in db.execute(q,(cut,origin,cut,cut,origin)):
            if s<=.3*d:continue
            v=hist[u].setdefault(i,[0,0,0.,0]);v[0]+=1;v[1]+=s>.8*d;v[2]+=math.exp(-math.log(2)*(cut-last)/(60*86400));v[3]=max(v[3],last)
        for u,i,d,s,last in db.execute(q,(end,cut,end,end,cut)):
            if s>.8*d:truth[u].add(i)
        past=dict(db.execute('SELECT item,COUNT(*) FROM events WHERE start<? AND end>? GROUP BY item',(cut,origin)))
        future=dict(db.execute('SELECT item,COUNT(*) FROM events WHERE start<? AND end>? GROUP BY item',(end,cut)))
        pop=Counter()
        for h in hist.values():pop.update(h.keys())
        fallback=sorted(future,key=lambda i:(-pop[i],int(i)));glob=fallback[:30]
        us=sorted(hist,key=int) if training else target_users
        X=[];Y=[];groups=[]
        for u in us:
            h=hist.get(u,{});total=sum(v[0] for v in h.values())
            def base(i):return h[i][0]*math.sqrt((future[i]/(end-cut))/(past.get(i,1)/(cut-origin)))
            personal=sorted((i for i in h if i in future),key=lambda i:(-base(i),int(i)))[:100]
            cand=list(dict.fromkeys(personal+glob));start=len(X)
            for i in cand:
                c,s,r,last=h.get(i,[0,0,0.,0]);fp=future[i]*86400/(end-cut);pp=past.get(i,0)*86400/(cut-origin)
                score=c*math.sqrt(fp/pp) if pp else 0
                X.append([math.log1p(c),math.log1p(s),math.log1p(r),math.log1p(fp),math.log1p(pp),math.log1p(score),math.log1p(pop[i]),math.log1p(total),min((cut-last)/86400,365) if last else 365]);Y.append(i in truth[u])
            groups.append((u,start,len(X),cand,not h))
        return np.array(X),np.array(Y,dtype=float),groups,truth,fallback
    fit=window(tick('2020-04-13 00:00:00'),tick('2020-05-18 00:00:00'),True)
    x,y,_,_,_=fit;mu=x.mean(axis=0);sd=x.std(axis=0);sd[sd==0]=1;x=(x-mu)/sd
    print(json.dumps({'phase':'fit','rows':len(y),'positive_rate':float(y.mean())}),flush=True)
    def loss(w):
        z=x@w[:-1]+w[-1];err=expit(z)-y
        return np.mean(np.logaddexp(0,z)-y*z)+.0005*np.dot(w[:-1],w[:-1]),np.r_[x.T@err/len(y)+.001*w[:-1],err.mean()]
    res=minimize(loss,np.zeros(x.shape[1]+1),jac=True,method='L-BFGS-B',options={'maxiter':80})
    ev=window(tick('2020-05-18 00:00:00'),tick('2020-07-27 00:00:00'),False)
    xx,yy,groups,truth,fallback=ev;score=((xx-mu)/sd)@res.x[:-1]+res.x[-1];pred={};aps=[];oracle=[]
    reference=json.loads((BASE/'evidence/strict_labels/truth.json').read_text())
    assert all(set(reference[u])==truth[u] for u in target_users)
    for u,a,b,cand,cold in groups:
        rec=fallback[:5] if cold else sorted(cand,key=lambda i:(-score[a+cand.index(i)],int(i)))[:5]
        pred[u]=rec;aps.append(apk(truth[u],rec));oracle.append(min(5,len(set(cand)&truth[u]))/min(5,len(truth[u])) if truth[u] else 0)
    result={'map5_strict':float(np.mean(aps)),'candidate_oracle_map5':float(np.mean(oracle)),'fit_rows':len(y),'fit_positive_rate':float(y.mean()),'features':protocol['features'],'coefficients_standardized':res.x.tolist(),'optimizer_success':bool(res.success),'optimizer_message':str(res.message),'iterations':int(res.nit),'wall_seconds':time.perf_counter()-began,'submissions':0}
    (OUT/'results.json').write_text(json.dumps(result,indent=2));(OUT/'predictions.json').write_text(json.dumps(pred,indent=2));np.savez(BASE/'cache/simple_reranker.npz',coefficients=res.x,mean=mu,scale=sd);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
