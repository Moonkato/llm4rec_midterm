"""Replay-only initialization and batched queries; verified interval methods unchanged."""
import csv, hashlib, math
from collections import defaultdict
from datetime import datetime
import numpy as np
import airing_v2_features as verified
from build_interactions import tick
from run_airing_model import B, ROOT
REPLAY=verified.REPLAY
VERIFIED_HELPER_SHA256='1d638c2af23bf1ee5414b42db6a68981ed0d0e11d4cc179be1d1d47622ed3943'

class FinalReplay(verified.Context):
    def __init__(self,users,cut,guard):
        assert hashlib.sha256((B/'tools/airing_v2_features.py').read_bytes()).hexdigest()==VERIFIED_HELPER_SHA256
        # Do not invoke the recurrence initializer or run any recurrence SQL.
        self.cut=cut;self.guard=guard;self.wanted=set(map(str,users));self.sessions=defaultdict(list)
        with (ROOT/'midterm/data/dataset11-30.csv').open(encoding='utf-8-sig',newline='') as f:
            for n,r in enumerate(csv.DictReader(f)):
                if n%100000==0:guard('replay_raw_history')
                if r['user_id'] not in self.wanted:continue
                a=max(0,tick(r['start_time']));b=min(cut,tick(r['stop_time']))
                if b>a:self.sessions[(r['user_id'],r['vsetv_id'])].append((a,b))
        self.current=None;self.indices={};self.memo={}
        self.days={dow:np.arange(dow,math.ceil((cut+3600)/86400),7,dtype=np.int64) for dow in range(7)}
        self.dst_civil=(datetime(2020,3,29,4)-datetime(2020,3,9)).total_seconds()
        cutoff_civil=cut+(3600 if cut>=self.dst_civil-3600 else 0)
        recent_civil=cutoff_civil-28*86400
        self.recent_start=recent_civil-(3600 if recent_civil>=self.dst_civil else 0)
    def prepare_user(self,u,tuples):
        self.current=u;self.indices={};self.memo={};needed=defaultdict(set)
        for t in tuples:needed[t[0]].add((t[1],t[5],t[6]))
        for ch,keys in needed.items():
            # One vectorized weekday grouping per user/channel, not per item.
            self.replay(u,ch,sorted(keys));self.guard('replay_user_batch')
        return sum(map(len,needed.values()))
    def features(self,u,item,tuples):
        assert self.current==u
        return np.asarray([self.memo[(t[0],t[1],t[5],t[6])] for t in tuples],dtype=np.float32)
