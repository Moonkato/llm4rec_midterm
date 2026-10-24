"""Full-universe count labels and exact expected AP for random boundary ties."""
from collections import defaultdict
from build_interactions import tick


def count_qualified(db,start,end):
    a,b=tick(start+' 00:00:00'),tick(end+' 00:00:00')
    sql='''SELECT o.user,e.item,COUNT(*) FROM
      (SELECT user,event_id FROM overlaps WHERE start<? AND end>?
       GROUP BY user,event_id HAVING MAX(MAX(0,MIN(end,?)-MAX(start,?)))>
       .8*(SELECT duration FROM events WHERE events.event_id=overlaps.event_id)) o
      JOIN events e USING(event_id) GROUP BY o.user,e.item'''
    result=defaultdict(dict)
    for u,i,n in db.execute(sql,(b,a,b,a)):result[u][i]=n
    return dict(result)


def tie_membership(counts,k=5):
    ordered=sorted(counts,key=lambda i:(-counts[i],int(i)))
    hard=set(ordered[:k]);den=min(k,len(ordered))
    if len(ordered)<=k:
        return {'hard':hard,'mandatory':set(ordered),'tied':set(),'remaining':0,'denominator':den}
    boundary=counts[ordered[k-1]]
    mandatory={i for i,n in counts.items() if n>boundary}
    tied={i for i,n in counts.items() if n==boundary}
    return {'hard':hard,'mandatory':mandatory,'tied':tied,'remaining':k-len(mandatory),'denominator':den}


def probability(item,info):
    if item in info['mandatory']:return 1.
    if item in info['tied']:return info['remaining']/len(info['tied'])
    return 0.


def expected_ap(predicted,info,k=5):
    """Analytical expectation over uniform subsets at true fifth-count boundary.

    Precision contains pairwise membership terms. Multiplying marginal probabilities
    for two tied items is incorrect because selection is without replacement.
    """
    if not info['denominator']:return 0.
    value=0.;seen=set();previous=[]
    for rank,item in enumerate(predicted[:k],1):
        if item in seen:continue
        seen.add(item);p=probability(item,info);joint=0.
        for other in previous:
            if item in info['tied'] and other in info['tied']:
                n=len(info['tied']);r=info['remaining']
                joint+=r*(r-1)/(n*(n-1)) if n>1 else 0.
            else:joint+=p*probability(other,info)
        value+=(p+joint)/rank;previous.append(item)
    return value/info['denominator']
