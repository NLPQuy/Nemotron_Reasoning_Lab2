import json, glob, math
from collections import Counter
def load():
    out=[]
    for p in sorted(glob.glob('problems/*.jsonl')):
        d=json.load(open(p))
        if d.get('category','').startswith('cryptarithm'): out.append(d)
    return out
def parse(d):
    ex=[]
    for e in d['examples']:
        iv=str(e['input_value']);ov=str(e['output_value'])
        if len(iv)!=5: return None
        ex.append((iv[0],iv[1],iv[2],iv[3],iv[4],tuple(ov)))
    q=str(d['question'])
    if len(q)!=5: return None
    return ex,(q[0],q[1],q[2],q[3],q[4])
def verify(stored,pred):
    import re
    stored=stored.strip();pred=pred.strip()
    if re.fullmatch(r'[01]+',stored): return pred==stored
    try: return math.isclose(float(stored),float(pred),rel_tol=1e-2,abs_tol=1e-5)
    except: return pred==stored
def idig(n):
    if n==0:return(0,)
    o=[]
    while n>0:o.append(n%10);n//=10
    return tuple(reversed(o))
OPS=[('add',lambda a,b:idig(a+b)),('abs_diff',lambda a,b:idig(abs(a-b))),('mul',lambda a,b:idig(a*b)),
 ('concat',lambda a,b:idig(a*100+b) if a>=10 else (0,)+idig(a*100+b)[ -3:] if False else None),
 ('rev_concat',None)]
# simpler: concat/rev as fixed-4 tuples
def concat4(a,b): return (a//10,a%10,b//10,b%10)
def rev4(a,b): return (b//10,b%10,a//10,a%10)
OPS=[('add',lambda a,b:idig(a+b)),('abs_diff',lambda a,b:idig(abs(a-b))),('mul',lambda a,b:idig(a*b)),
 ('concat',concat4),('rev_concat',rev4)]
class S:  # NON-unique bijection (digits may repeat), all examples must fit (skip=0)
    def __init__(s,ex,q,budget=200000):
        s.ex=ex;s.q=q;s.map={};s.opa={};s.budget=budget;s.nodes=0;s.answers=Counter();s.maxans=50
    def solve(s):
        s.order=sorted(range(len(s.ex)),key=lambda i:0 if s.ex[i][2]==s.q[2] else 1)
        s._p(0); return s.answers.most_common(1)[0][0] if s.answers else None
    def _vals(s,sy):
        return (s.map[sy],) if sy in s.map else range(10)
    def _as(s,sy,d):
        if sy in s.map: return False if s.map[sy]==d else None
        s.map[sy]=d; return True
    def _un(s,sy,w):
        if w is True: del s.map[sy]
    def _p(s,oi):
        if s.nodes>s.budget or len(s.answers)>=s.maxans: return
        s.nodes+=1
        if oi==len(s.order): s._q(); return
        idx=s.order[oi]; a0,a1,op,b0,b1,rs=s.ex[idx]
        for d0 in s._vals(a0):
            n0=s._as(a0,d0)
            if n0 is None: continue
            for d1 in s._vals(a1):
                n1=s._as(a1,d1)
                if n1 is None: continue
                l=d0*10+d1
                for d3 in s._vals(b0):
                    n3=s._as(b0,d3)
                    if n3 is None: continue
                    for d4 in s._vals(b1):
                        n4=s._as(b1,d4)
                        if n4 is None: continue
                        r=d3*10+d4
                        tryops=[s.opa[op]] if op in s.opa else range(len(OPS))
                        for k in tryops:
                            res=OPS[k][1](l,r)
                            if res is None or len(res)!=len(rs): continue
                            ass=[];ok=True
                            for c,dg in zip(rs,res):
                                x=s._as(c,dg)
                                if x is None: ok=False;break
                                ass.append((c,x))
                            if ok:
                                new=op not in s.opa
                                if new: s.opa[op]=k
                                s._p(oi+1)
                                if new: del s.opa[op]
                            for c,x in reversed(ass): s._un(c,x)
                            if s.nodes>s.budget or len(s.answers)>=s.maxans:
                                s._un(b1,n4);s._un(b0,n3);s._un(a1,n1);s._un(a0,n0);return
                        s._un(b1,n4)
                    s._un(b0,n3)
                s._un(a1,n1)
            s._un(a0,n0)
    def _q(s):
        q0,q1,qop,q3,q4=s.q
        for c in (q0,q1,q3,q4):
            if c not in s.map: return
        if qop not in s.opa: return
        res=OPS[s.opa[qop]][1](s.map[q0]*10+s.map[q1], s.map[q3]*10+s.map[q4])
        if res is None: return
        d2s={}
        for c,d in s.map.items():
            d2s.setdefault(d,c)
        parts=[]
        for d in res:
            if d not in d2s: return
            parts.append(d2s[d])
        s.answers[''.join(parts)]+=1
probs=load()
corr=anyc=0
for d in probs:
    p=parse(d)
    if p is None: continue
    ex,q=p
    s=S(ex,q,budget=200000)
    a=s.solve()
    if a is not None:
        anyc+=1
        if verify(str(d['answer']),a): corr+=1
print('NON-unique bijection, base ops, skip=0:  any=%d correct=%d'%(anyc,corr),flush=True)
print('done',flush=True)
