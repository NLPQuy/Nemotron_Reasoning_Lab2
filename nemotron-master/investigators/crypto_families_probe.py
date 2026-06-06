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
def idig(n):  # int -> digit tuple (no leading zeros)
    if n==0: return (0,)
    o=[]
    while n>0:o.append(n%10);n//=10
    return tuple(reversed(o))
# An op maps (d0,d1,d3,d4) -> tuple of result digits (or None). left=10d0+d1, right=10d3+d4.
def mk():
    ops={}
    ops['add']=lambda a,b,c,d: idig((10*a+b)+(10*c+d))
    ops['abs_diff']=lambda a,b,c,d: idig(abs((10*a+b)-(10*c+d)))
    ops['mul']=lambda a,b,c,d: idig((10*a+b)*(10*c+d))
    ops['concat']=lambda a,b,c,d: (a,b,c,d)
    ops['rev_concat']=lambda a,b,c,d: (c,d,a,b)
    # leading-zero padded arithmetic (width 2,3,4 variants for add/abs_diff/mul)
    def padded(fn,w):
        def g(a,b,c,d):
            v=fn(a,b,c,d)
            if v<0: return None
            ds=idig(v)
            if len(ds)>w: return None
            return (0,)*(w-len(ds))+ds
        return g
    ops['add_pad2']=padded(lambda a,b,c,d:(10*a+b)+(10*c+d),2)
    ops['add_pad3']=padded(lambda a,b,c,d:(10*a+b)+(10*c+d),3)
    ops['absdiff_pad2']=padded(lambda a,b,c,d:abs((10*a+b)-(10*c+d)),2)
    ops['mul_pad4']=padded(lambda a,b,c,d:(10*a+b)*(10*c+d),4)
    ops['mul_pad3']=padded(lambda a,b,c,d:(10*a+b)*(10*c+d),3)
    # digit-wise (2-digit results)
    ops['dw_add_mod10']=lambda a,b,c,d: ((a+c)%10,(b+d)%10)
    ops['dw_absdiff']=lambda a,b,c,d: (abs(a-c),abs(b-d))
    ops['dw_max']=lambda a,b,c,d: (max(a,c),max(b,d))
    ops['dw_min']=lambda a,b,c,d: (min(a,c),min(b,d))
    ops['dw_mul_mod10']=lambda a,b,c,d: ((a*c)%10,(b*d)%10)
    # string / positional (4-digit results) by digit value
    ops['sort_asc']=lambda a,b,c,d: tuple(sorted([a,b,c,d]))
    ops['sort_desc']=lambda a,b,c,d: tuple(sorted([a,b,c,d],reverse=True))
    ops['interleave']=lambda a,b,c,d: (a,c,b,d)
    ops['reverse_all']=lambda a,b,c,d: (d,c,b,a)
    ops['swap_pairs']=lambda a,b,c,d: (b,a,d,c)
    # reductions
    ops['sum_all']=lambda a,b,c,d: idig(a+b+c+d)
    ops['prod_all']=lambda a,b,c,d: idig(a*b*c*d)
    ops['sum_digits_each']=lambda a,b,c,d: idig((10*a+b))  # left only (sanity)
    # wrap to guard exceptions/range
    def wrap(fn):
        def g(a,b,c,d):
            try:
                v=fn(a,b,c,d)
                if v is None: return None
                if any(x<0 or x>9 for x in v): return None
                return v
            except: return None
        return g
    return {k:wrap(v) for k,v in ops.items()}
ALLOPS=mk()
NAMES=list(ALLOPS)
class S:  # robust skip<=1, op library = given subset (list of names)
    def __init__(s,ex,q,opnames,max_skip=1,budget=400000):
        s.ex=ex;s.q=q;s.qop=q[2];s.opn=opnames;s.fns=[ALLOPS[n] for n in opnames]
        s.map={};s.used=set();s.opa={};s.budget=budget;s.nodes=0;s.answers=Counter();s.maxans=60;s.ms=max_skip
    def solve(s):
        s.order=sorted(range(len(s.ex)),key=lambda i:0 if s.ex[i][2]==s.qop else 1)
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
    def _p(s,oi,skips):
        if s.nodes>s.budget or len(s.answers)>=s.maxans: return
        s.nodes+=1
        if oi==len(s.order): s._q(); return
        idx=s.order[oi]; a0,a1,op,b0,b1,rs=s.ex[idx]; isq=(op==s.qop)
        for d0 in s._vals(a0):
            n0=s._as(a0,d0)
            if n0 is None: continue
            for d1 in s._vals(a1):
                n1=s._as(a1,d1)
                if n1 is None: continue
                for d3 in s._vals(b0):
                    n3=s._as(b0,d3)
                    if n3 is None: continue
                    for d4 in s._vals(b1):
                        n4=s._as(b1,d4)
                        if n4 is None: continue
                        tryops=[s.opa[op]] if op in s.opa else range(len(s.fns))
                        for k in tryops:
                            res=s.fns[k](d0,d1,d3,d4)
                            if res is None or len(res)!=len(rs): continue
                            ass=[];ok=True
                            for c,dg in zip(rs,res):
                                x=s._as(c,dg)
                                if x is None: ok=False;break
                                ass.append((c,x))
                            if ok:
                                new=op not in s.opa
                                if new: s.opa[op]=k
                                s._p(oi+1,skips)
                                if new: del s.opa[op]
                            for c,x in reversed(ass): s._un(c,x)
                            if s.nodes>s.budget or len(s.answers)>=s.maxans:
                                s._un(b1,n4);s._un(b0,n3);s._un(a1,n1);s._un(a0,n0);return
                        s._un(b1,n4)
                    s._un(b0,n3)
                s._un(a1,n1)
            s._un(a0,n0)
        if not isq and skips<s.ms:
            s._p(oi+1,skips+1)
    def _q(s):
        q0,q1,qop,q3,q4=s.q
        for c in (q0,q1,q3,q4):
            if c not in s.map: return
        if qop not in s.opa: return
        k=s.opa[qop]
        d2s={}
        for c,d in s.map.items():
            if d not in d2s: d2s[d]=c
        res=s.fns[k](s.map[q0],s.map[q1],s.map[q3],s.map[q4])
        if res is None: return
        parts=[]
        for d in res:
            if d not in d2s: return
            parts.append(d2s[d])
        s.answers[''.join(parts)]+=1

probs=load()
BASE=['add','abs_diff','mul','concat','rev_concat']
configs={
 'base_robust': BASE,
 '+leadzero': BASE+['add_pad2','add_pad3','absdiff_pad2','mul_pad4','mul_pad3'],
 '+digitwise': BASE+['dw_add_mod10','dw_absdiff','dw_max','dw_min','dw_mul_mod10'],
 '+stringops': BASE+['sort_asc','sort_desc','interleave','reverse_all','swap_pairs'],
 '+reductions': BASE+['sum_all','prod_all'],
 'ALL': [n for n in NAMES if n!='sum_digits_each'],
}
for name,opn in configs.items():
    corr=anyc=0
    for d in probs:
        p=parse(d)
        if p is None: continue
        ex,q=p
        s=S(ex,q,opn,max_skip=1,budget=400000)
        a=s.solve()
        if a is not None:
            anyc+=1
            if verify(str(d['answer']),a): corr+=1
    print('%-14s any=%4d correct=%4d'%(name,anyc,corr),flush=True)
print('done',flush=True)
