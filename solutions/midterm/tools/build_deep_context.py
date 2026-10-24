"""Prepare nonoverlapping inner fit/early-stop windows; no model fitting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import hashlib,json,sqlite3,time
from build_context_features import BASE,FEATURES,build_window

def main():
    out=BASE/'cache/deep_context';evidence=BASE/'evidence/deep_context'
    if (evidence/'results.json').exists():raise SystemExit('Preserve completed deep-context inputs.')
    out.mkdir(parents=True,exist_ok=True);evidence.mkdir(parents=True,exist_ok=True)
    windows=[('inner_fit','2020-04-13 00:00:00','2020-05-04 00:00:00',True),
             ('inner_stop','2020-05-04 00:00:00','2020-05-18 00:00:00',False)]
    if any((out/f'{name}.npz').exists() for name,*_ in windows):
        raise SystemExit('Partial inputs exist; preserve them and inspect before rebuilding.')
    protocol={'windows':windows,'history_origin':'2020-03-09','features':FEATURES,
      'purpose':'Chronologically nonoverlapping fitting labels and inner stopping labels, both before outerMay18 evaluation.',
      'fit_cohort':'all users with usable preApr13 history','stop_cohort':'1260 template users; empty truth retained',
      'labels':'maximum clipped single stored fragment >80% full airing duration',
      'candidate_generation':'unchanged context builder: personal100+popularity30+context30',
      'caveats':['OldMay18 fragment identity limitation retained.','Inner fit and stopping horizons differ:21days and14days.','Previously explored outer evaluation is still not an untouched final test.'],
      'training_runs':0,'submissions':0,
      'builder_sha256':hashlib.sha256((BASE/'tools/build_context_features.py').read_bytes()).hexdigest()}
    (evidence/'protocol.json').write_text(json.dumps(protocol,indent=2))
    started=time.perf_counter()
    db=sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro',uri=True)
    db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-64000')
    results=[build_window(db,*window,out,evidence) for window in windows]
    db.close()
    report={'status':'complete','results':results,'wall_seconds':time.perf_counter()-started,
      'code_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
      'training_runs':0,'submissions':0}
    (evidence/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
