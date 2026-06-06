import json, glob, math
from collections import Counter, defaultdict
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
def n2d(n):
    if n==0:return(0,)
    o=[]
    while n>0:o.append(n%10);n//=10
    return tuple(reversed(o))
def safe(fn):
    def g(a,b):
        try:
            v=fn(a,b)
            if v is None or v<0 or v>=100000:return None
            return int(v)
        except:return None
    return g
OPS=[('add',safe(lambda a,b:a+b)),('abs_diff',safe(lambda a,b:abs(a-b))),
 ('mul',safe(lambda a,b:a*b)),('concat',safe(lambda a,b:a*100+b)),('rev_concat',safe(lambda a,b:b*100+a))]
# For each problem, find a globally-consistent bijection+op (base, unique). Record glyph->op for
# each operator that gets a forced assignment in the BEST solution.
class S:
    def __init__(s,ex,budget=300000):
        s.ex=ex;s.map={};s.used=set();s.opa={};s.budget=budget;s.nodes=0;s.sol=None
    def solve(s): s._p(0); return s.sol
    def _vals(s,sy):
        if sy in s.map:return(s.map[sy],)
        return [d for d in range(10) if d not in s.used]
    def _as(s,sy,d):
        if sy in s.map:return False if s.map[sy]==d else None
        if d in s.used:return None
        s.map[sy]=d;s.used.add(d);return True
    def _un(s,sy,w):
        if w is True:s.used.discard(s.map[sy]);del s.map[sy]
    def _p(s,idx):
        if s.sol is not None or s.nodes>s.budget:return
        s.nodes+=1
        if idx==len(s.ex):
            s.sol={op:OPS[oi][0] for op,oi in s.opa.items()};return
        a0,a1,op,b0,b1,rs=s.ex[idx]
        for d0 in s._vals(a0):
            n0=s._as(a0,d0)
            if n0 is None:continue
            for d1 in s._vals(a1):
                n1=s._as(a1,d1)
                if n1 is None:continue
                l=d0*10+d1
                for d3 in s._vals(b0):
                    n3=s._as(b0,d3)
                    if n3 is None:continue
                    for d4 in s._vals(b1):
                        n4=s._as(b1,d4)
                        if n4 is None:continue
                        r=d3*10+d4
                        tryops=[s.opa[op]] if op in s.opa else range(len(OPS))
                        for oi in tryops:
                            res=OPS[oi][1](l,r)
                            if res is None:continue
                            rd=n2d(res)
                            if len(rd)!=len(rs):continue
                            ass=[];ok=True
                            for c,dg in zip(rs,rd):
                                x=s._as(c,dg)
                                if x is None:ok=False;break
                                ass.append((c,x))
                            if ok:
                                new=op not in s.opa
                                if new:s.opa[op]=oi
                                s._p(idx+1)
                                if new and s.sol is None:del s.opa[op]
                            for c,x in reversed(ass):s._un(c,x)
                            if s.sol is not None or s.nodes>s.budget:
                                s._un(b1,n4);s._un(b0,n3);s._un(a1,n1);s._un(a0,n0);return
                        s._un(b1,n4)
                    s._un(b0,n3)
                s._un(a1,n1)
            s._un(a0,n0)
probs=load()
glyph=defaultdict(Counter)
ndet=0
for d in probs:
    p=parse(d)
    if p is None:continue
    ex,q=p
    s=S(ex)
    sol=s.solve()
    if sol:
        ndet+=1
        for g,opname in sol.items():
            glyph[g][opname]+=1
print('problems with a consistent base solution:',ndet,flush=True)
print('glyph -> operation distribution (only glyphs seen >=5x):',flush=True)
for g in sorted(glyph, key=lambda x:-sum(glyph[x].values())):
    tot=sum(glyph[g].values())
    if tot>=5:
        top=glyph[g].most_common()
        purity=top[0][1]/tot
        print('  %r  n=%3d  purity=%.2f  %s'%(g,tot,purity,top),flush=True)
print('done',flush=True)
