"""Submit only the explicitly approved first candidate; never retry ambiguous writes."""
import os,json,sys,hashlib,csv
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'solutions/midterm'
os.environ['KAGGLE_API_TOKEN']=(ROOT/'midterm/api-token').read_text().strip()
os.environ['KAGGLE_CONFIG_DIR']=str(BASE/'cache/kaggle_config')
NAME='midterm_01_logistic_schedule';SLUG='sweettv-tv-program-recommender'
OUT=BASE/'evidence/submissions';OUT.mkdir(exist_ok=True)
import truststore
truststore.inject_into_ssl()
from kaggle.api.kaggle_api_extended import KaggleApi

def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,default=str)+'\n')
def main():
    operation=sys.argv[1];assert operation in ['status','submit']
    api=KaggleApi();api.authenticate()
    rows=api.competition_submissions(SLUG,page_size=100)
    data=[{k:getattr(r,k,None) for k in ['ref','description','file_name','status','public_score','private_score','error_description','date']} for r in rows or []]
    save('latest_status.json',data)
    ours=[r for r in data if r['description']==NAME]
    print(json.dumps({'submissions_in_history':len(data),'matching_candidate':ours},default=str),flush=True)
    statepath=BASE/'run_state.json';state=json.loads(statepath.read_text());state['proposed_submission_rules']['approval']='approved by user'
    if ours:
        state['submissions_made']=max(1,state['submissions_made']);state['phase']='first_submission_recorded';state['first_submission']=ours[0];statepath.write_text(json.dumps(state,indent=2,default=str)+'\n')
        save('first_submission.json',ours[0]);return
    if operation=='status':return
    attempt=OUT/'first_attempt.json'
    if attempt.exists():raise RuntimeError('Prior submission attempt exists; reconcile history before any retry')
    file=BASE/'candidates/logistic_v1.csv';digest=hashlib.sha256(file.read_bytes()).hexdigest()
    assert digest=='90e5341edb9ff7e30f1c54cc48148f409d4ae93ac55dbf3d733d872c54631346'
    assert state['submissions_made']==0
    with file.open() as f: records=list(csv.DictReader(f))
    assert len(records)==1260 and all(len(set(r['tv_show_id'].split()))==5 for r in records)
    save('first_attempt.json',{'name':NAME,'file':str(file.relative_to(ROOT)),'sha256':digest,'utc':datetime.now(timezone.utc).isoformat(),'status':'attempt started; do not automatically retry'})
    response=api.competition_submit(str(file),NAME,SLUG,quiet=True)
    result={k:getattr(response,k,None) for k in ['ref','message','error','has_error']}
    save('first_response.json',result);print(json.dumps(result,default=str),flush=True)
    state['phase']='first_submission_response_received';statepath.write_text(json.dumps(state,indent=2)+'\n')
try:main()
except Exception as e:
    # Avoid displaying URLs, headers or exception text which might expose credentials.
    detail={'error_type':type(e).__name__,'operation':sys.argv[1],'http_status':getattr(getattr(e,'response',None),'status_code',None)}
    save('last_error.json',detail);print(json.dumps(detail),flush=True);sys.exit(1)
