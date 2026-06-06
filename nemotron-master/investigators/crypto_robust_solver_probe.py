import json, glob, math
from collections import Counter, defaultdict
def load():
    out=[]
    for p in sorted(glob.glob('problems/*.jsonl')):
        d=json.load(open(p))
        if d.get('category','').startswith('cryptarithm'): out.append((p,d))
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
def n2d(n):
    if n==0: return (0,)
    o=[]
    while n>0:o.append(n%10);n//=10
    return tuple(reversed(o))
def safe(fn):
    def g(a,b):
        try:
            v=fn(a,b)
            if v is None or v<0 or v>=100000: return None
            return int(v)
        except: return None
    return g
def verify(stored,pred):
    import re
    stored=stored.strip();pred=pred.strip()
    if re.fullmatch(r'[01]+',stored): return pred==stored
    try: return math.isclose(float(stored),float(pred),rel_tol=1e-2,abs_tol=1e-5)
    except: return pred==stored
OPS=[('add',safe(lambda a,b:a+b)),('abs_diff',safe(lambda a,b:abs(a-b))),
 ('mul',safe(lambda a,b:a*b)),('concat',safe(lambda a,b:a*100+b)),('rev_concat',safe(lambda a,b:b*100+a))]

# ROBUST solver: find a bijection + per-op assignment that explains as MANY examples as possible,
# but REQUIRE the examples that share the query operator to be explained, and require >= the
# query op examples. Allow OTHER-operator examples to be 'skipped' if unexplainable.
# Implementation: backtracking over examples; at each example, either fit it with some op (under
# bijection) OR skip it (cost++), with max_skip budget. Must fit all examples whose op == query op.
class S:
    def __init__(s,ex,q,max_skip=2,budget=400000):
        s.ex=ex;s.q=q;s.qop=q[2];s.map={};s.used=set();s.opa={}
        s.budget=budget;s.nodes=0;s.answers=Counter();s.maxans=60;s.max_skip=max_skip
    def solve(s):
        # order examples: those with query op first (must fit), then others
        order=sorted(range(len(s.ex)), key=lambda i: 0 if s.ex[i][2]==s.qop else 1)
        s.order=order
        s._p(0,0)
        return s.answers.most_common(1)[0][0] if s.answers else None
    def _vals(s,sy):
        if sy in s.map: return (s.map[sy],)
        return [d for d in range(10) if d not in s.used]
    def _as(s,sy,d):
        if sy in s.map: return False if s.map[sy]==d else None
        if d in s.used: return None
        s.map[sy]=d;s.used.add(d);return True
    def _un(s,sy,w):
        if w is True: s.used.discard(s.map[sy]);del s.map[sy]
    def _p(s,oi_idx,skips):
        if s.nodes>s.budget or len(s.answers)>=s.maxans: return
        s.nodes+=1
        if oi_idx==len(s.order): s._q(); return
        idx=s.order[oi_idx]
        a0,a1,op,b0,b1,rs=s.ex[idx]
        is_qop=(op==s.qop)
        # try to fit this example
        fitted_any=False
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
                        for oi in tryops:
                            res=OPS[oi][1](l,r)
                            if res is None: continue
                            rd=n2d(res)
                            if len(rd)!=len(rs): continue
                            ass=[];ok=True
                            for c,dg in zip(rs,rd):
                                x=s._as(c,dg)
                                if x is None: ok=False;break
                                ass.append((c,x))
                            if ok:
                                new=op not in s.opa
                                if new: s.opa[op]=oi
                                s._p(oi_idx+1,skips)
                                if new: del s.opa[op]
                            for c,x in reversed(ass): s._un(c,x)
                            if s.nodes>s.budget or len(s.answers)>=s.maxans:
                                s._un(b1,n4);s._un(b0,n3);s._un(a1,n1);s._un(a0,n0);return
                        s._un(b1,n4)
                    s._un(b0,n3)
                s._un(a1,n1)
            s._un(a0,n0)
        # option: skip this example (only if not query-op and budget remains)
        if not is_qop and skips < s.max_skip:
            s._p(oi_idx+1, skips+1)
    def _q(s):
        q0,q1,qop,q3,q4=s.q
        for c in (q0,q1,q3,q4):
            if c not in s.map: return
        ql=s.map[q0]*10+s.map[q1];qr=s.map[q3]*10+s.map[q4]
        if qop not in s.opa: return  # query op must be determined by an example
        oi=s.opa[qop]
        d2s={}
        for c,d in s.map.items():
            if d not in d2s: d2s[d]=c
        res=OPS[oi][1](ql,qr)
        if res is None: return
        parts=[]
        for d in n2d(res):
            if d not in d2s: return
            parts.append(d2s[d])
        s.answers[''.join(parts)]+=1

probs=load()
# Compare: strict (max_skip=0, all must fit, query op must appear) vs robust (max_skip=2)
for ms in (0,1,2):
    corr=anyc=0
    for _,d in probs:
        p=parse(d)
        if p is None: continue
        ex,q=p
        s=S(ex,q,max_skip=ms,budget=400000)
        a=s.solve()
        if a is not None:
            anyc+=1
            if verify(str(d['answer']),a): corr+=1
    print('max_skip=%d  any=%d correct=%d'%(ms,anyc,corr),flush=True)
print('done',flush=True)
