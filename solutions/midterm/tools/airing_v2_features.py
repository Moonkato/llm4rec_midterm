"""History-only broadcast recurrence and exact single-session weekly replay."""
import csv, math
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
from run_airing_model import ROOT
from build_interactions import tick
RECURRENCE=['ui_channel_p80','ui_channel_p30','ui_channel_hour_p80','u_channel_hour_p80','log_ui_channel_count80','log_ui_channel_hour_count80']
REPLAY=['weekly_replay_p30','weekly_replay_p80','recent28_replay_p80','log_replay_dates']

class Context:
    def __init__(self,db,users,cut,guard,with_replay=True):
        self.cut=cut;self.guard=guard;self.wanted=set(map(str,users));self.counts=defaultdict(lambda:[0,0]);self.exposure=Counter()
        for item,ch,civil in db.execute('SELECT item,channel,civil_start FROM events WHERE start>=0 AND end<=?',(cut,)):
            hour=datetime.fromisoformat(civil).hour
            for key in [('i',item),('ic',item,ch),('ich',item,ch,hour),('c',ch),('ch',ch,hour)]:self.exposure[key]+=1
        q='''SELECT o.user,e.item,e.channel,e.civil_start,e.duration,MAX(MAX(0,MIN(o.end,?)-MAX(o.start,0))) FROM overlaps o JOIN events e USING(event_id) WHERE e.start>=0 AND e.end<=? AND o.start<? AND o.end>0 GROUP BY o.user,o.event_id'''
        for n,(u,item,ch,civil,dur,overlap) in enumerate(db.execute(q,(cut,cut,cut))):
            if n%100000==0:guard('recurrence_history')
            if u not in self.wanted or overlap<=.3*dur:continue
            hour=datetime.fromisoformat(civil).hour
            for key in [('ui',u,item),('uic',u,item,ch),('uich',u,item,ch,hour),('uc',u,ch),('uch',u,ch,hour)]:
                v=self.counts[key];v[0]+=1;v[1]+=overlap>.8*dur
        if not with_replay:return
        self.sessions=defaultdict(list)
        with (ROOT/'midterm/data/dataset11-30.csv').open(encoding='utf-8-sig',newline='') as f:
            for n,r in enumerate(csv.DictReader(f)):
                if n%100000==0:guard('replay_raw_history')
                if r['user_id'] not in self.wanted:continue
                a=max(0,tick(r['start_time']));b=min(cut,tick(r['stop_time']))
                if b>a:self.sessions[(r['user_id'],r['vsetv_id'])].append((a,b))
        self.current=None;self.indices={};self.memo={}
        # Civil day0 is Monday2020-03-09. tick removes the missing DST hour.
        self.days={dow:np.arange(dow,math.ceil((cut+3600)/86400),7,dtype=np.int64) for dow in range(7)}
        self.dst_civil=(datetime(2020,3,29,4)-datetime(2020,3,9)).total_seconds()
        cutoff_civil=cut+(3600 if cut>=self.dst_civil-3600 else 0)
        recent_civil=cutoff_civil-28*86400
        self.recent_start=recent_civil-(3600 if recent_civil>=self.dst_civil else 0)
    def recurrence(self,u,item,ch,hour):
        def c(key,k):return self.counts.get(key,(0,0))[k]
        ni=self.exposure[('i',item)];nic=self.exposure[('ic',item,ch)];nich=self.exposure[('ich',item,ch,hour)]
        nc=self.exposure[('c',ch)];nch=self.exposure[('ch',ch,hour)]
        pi80=c(('ui',u,item),1)/max(ni,1);pi30=c(('ui',u,item),0)/max(ni,1)
        pc80=c(('uc',u,ch),1)/max(nc,1)
        pic80=(c(('uic',u,item,ch),1)+5*pi80)/(nic+5)
        pic30=(c(('uic',u,item,ch),0)+5*pi30)/(nic+5)
        pich80=(c(('uich',u,item,ch,hour),1)+5*pic80)/(nich+5)
        pch80=(c(('uch',u,ch,hour),1)+5*pc80)/(nch+5)
        return [pic80,pic30,pich80,pch80,math.log1p(c(('uic',u,item,ch),1)),math.log1p(c(('uich',u,item,ch,hour),1))]
    def index(self,u,ch):
        if ch in self.indices:return self.indices[ch]
        seq=sorted(self.sessions.get((u,ch),[]));starts=np.array([a for a,b in seq],dtype=np.int64);ends=np.array([b for a,b in seq],dtype=np.int64)
        n=1
        while n<len(seq):n*=2
        tree=np.zeros(2*n,dtype=np.int64);tree[n:n+len(seq)]=ends-starts
        for j in range(n-1,0,-1):tree[j]=max(tree[2*j],tree[2*j+1])
        result=(starts,np.maximum.accumulate(ends) if len(ends) else ends,tree,n);self.indices[ch]=result;return result
    @staticmethod
    def qualifying(index,s,d,threshold):
        starts,prefix,tree,n=index
        if not len(starts):return np.zeros(s.shape,dtype=bool)
        left=np.searchsorted(starts,s,side='right');ok=np.zeros(s.shape,dtype=bool);valid=left>0
        ok[valid]=prefix[left[valid]-1]>s[valid]+threshold*d[valid]
        # Later-starting sessions need both sufficient own duration and enough time before airing end.
        right=np.searchsorted(starts,s+(1-threshold)*d,side='left');l=left+n;r=right+n;maximum=np.zeros(s.shape,dtype=np.int64)
        while np.any(l<r):
            live=l<r;mask=live & ((l&1)==1);maximum[mask]=np.maximum(maximum[mask],tree[l[mask]]);l[mask]+=1
            mask=live & ((r&1)==1);r[mask]-=1;maximum[mask]=np.maximum(maximum[mask],tree[r[mask]])
            l//=2;r//=2
        return ok | (maximum>threshold*d)
    def replay(self,u,ch,keys):
        if self.current!=u:self.current=u;self.indices={};self.memo={}
        missing=list(dict.fromkeys(k for k in keys if (ch,*k) not in self.memo))
        index=self.index(u,ch)
        for dow in range(7):
            selected=[k for k in missing if k[0]==dow]
            if not selected:continue
            days=self.days[dow];seconds=np.array([k[1] for k in selected],dtype=np.int64);dur=np.array([k[2] for k in selected],dtype=np.int64)
            civil=days[None,:]*86400+seconds[:,None]
            s=civil-(civil>=self.dst_civil)*3600;duration=np.broadcast_to(dur[:,None],s.shape)
            missinghour=(civil>=self.dst_civil-3600)&(civil<self.dst_civil)
            valid=(s>=0)&(s+duration<=self.cut)&~missinghour
            recent=valid&(s>=self.recent_start)
            q30=self.qualifying(index,s.ravel(),duration.ravel(),.3).reshape(s.shape)&valid
            q80=self.qualifying(index,s.ravel(),duration.ravel(),.8).reshape(s.shape)&valid
            denom=valid.sum(axis=1);rden=recent.sum(axis=1)
            for j,k in enumerate(selected):
                self.memo[(ch,*k)]=[q30[j].sum()/max(1,denom[j]),q80[j].sum()/max(1,denom[j]),(q80[j]&recent[j]).sum()/max(1,rden[j]),math.log1p(denom[j])]
        return np.array([self.memo[(ch,*k)] for k in keys],dtype=np.float32)
    def features(self,u,item,tuples):
        out=np.empty((len(tuples),10),dtype=np.float32)
        for j,t in enumerate(tuples):out[j,:6]=self.recurrence(u,item,t[0],t[2])
        for ch in dict.fromkeys(t[0] for t in tuples):
            ix=[j for j,t in enumerate(tuples) if t[0]==ch]
            keys=[(tuples[j][1],tuples[j][5],tuples[j][6]) for j in ix]
            out[ix,6:]=self.replay(u,ch,keys)
        return out
