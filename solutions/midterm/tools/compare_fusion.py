"""Small prespecified rank-fusion selection on the existing temporal holdout."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json, sqlite3, time
from pathlib import Path
import numpy as np
from run_round1 import apk
BASE=Path(__file__).resolve().parents[1]
out=BASE/'evidence/fusion'; out.mkdir(exist_ok=True)
config={'components':['personal_frequency','popularity_unique_viewers'],'personal_weights':[0.25,0.5,0.75,0.9],'rrf_constant':60,'selection':'exploratory selection on round1 holdout; not independent confirmation','training_runs':0,'kaggle_submissions':0}
(out/'protocol.json').write_text(json.dumps(config,indent=2))
t=time.perf_counter()
catalog=json.loads((BASE/'evidence/round1/catalog.json').read_text())
users=catalog['users']; candidates=catalog['candidates']; truth={u:set() for u in users}
db=sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro',uri=True)
for u,i in db.execute('SELECT DISTINCT w.user,e.item FROM watches w JOIN events e USING(event_id) WHERE w.split=1 AND w.seconds>=180'):
    if u in truth: truth[u].add(i)
db.close()
components=[]
for name in config['components']:
    scores=np.load(BASE/'cache/round1'/f'{name}_scores.npy')
    order=np.argsort(-scores,axis=1,kind='stable')
    ranks=np.empty_like(order)
    np.put_along_axis(ranks,order,np.broadcast_to(np.arange(1,len(candidates)+1),order.shape),axis=1)
    components.append(np.where(np.isfinite(scores),1.0/(60+ranks),0.0))
results=[]; predictions={}
for weight in config['personal_weights']:
    scores=weight*components[0]+(1-weight)*components[1]
    top=np.argsort(-scores,axis=1,kind='stable')[:,:5]
    recs={u:[candidates[i] for i in row] for u,row in zip(users,top)}
    values=np.array([apk(truth[u],recs[u]) for u in users])
    result={'personal_weight':weight,'map5':float(values.mean())}
    results.append(result); predictions[str(weight)]=recs
    print(json.dumps(result),flush=True)
(out/'results.json').write_text(json.dumps({'results':results,'wall_seconds':time.perf_counter()-t},indent=2))
(out/'predictions.json').write_text(json.dumps(predictions,indent=2))
state=json.loads((BASE/'run_state.json').read_text()); state.update(phase='round1_and_small_fusion_complete',training_runs=4,round1_results='evidence/round1/results.json',fusion_results='evidence/fusion/results.json')
(BASE/'run_state.json').write_text(json.dumps(state,indent=2)+'\n')
