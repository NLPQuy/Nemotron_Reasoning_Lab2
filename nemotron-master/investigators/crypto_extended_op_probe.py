import json, glob, math, sys
from collections import Counter, defaultdict

def load():
    out=[]
    for path in sorted(glob.glob('problems/*.jsonl')):
        d=json.load(open(path))
        if d.get('category','').startswith('cryptarithm'): out.append(d)
    return out

def parse(d):
    ex=[]
    for e in d['examples']:
        iv=str(e['input_value']); ov=str(e['output_value'])
        if len(iv)!=5: return None
        ex.append((iv[0],iv[1],iv[2],iv[3],iv[4],tuple(ov)))
    q=str(d['question'])
    if len(q)!=5: return None
    return ex,(q[0],q[1],q[2],q[3],q[4])

def n2d(n):
    if n==0: return (0,)
    o=[]
    while n>0: o.append(n%10); n//=10
    return tuple(reversed(o))

def safe(fn):
    def g(a,b):
        try:
            v=fn(a,b)
            if v is None or v<0 or v>=100000: return None
            return int(v)
        except: return None
    return g

BASE=[('add',safe(lambda a,b:a+b)),('abs_diff',safe(lambda a,b:abs(a-b))),
 ('mul',safe(lambda a,b:a*b)),('concat',safe(lambda a,b:a*100+b)),('rev_concat',safe(lambda a,b:b*100+a))]
EXTRA=[('floordiv',safe(lambda a,b:a//b if b else None)),('mod',safe(lambda a,b:a%b if b else None)),
 ('min',safe(lambda a,b:min(a,b))),('max',safe(lambda a,b:max(a,b))),
 ('sumdig',safe(lambda a,b:(a//10)+(a%10)+(b//10)+(b%10))),
 ('l_plus_revr',safe(lambda a,b:a+(b%10*10+b//10))),('sq_l',safe(lambda a,b:a*a)),('sq_r',safe(lambda a,b:b*b)),
 ('avg',safe(lambda a,b:(a+b)//2)),('a_plus_2b',safe(lambda a,b:a+2*b)),('2a_plus_b',safe(lambda a,b:2*a+b)),
 ('mul_plus1',safe(lambda a,b:a*b+1)),('sub_floor10',safe(lambda a,b:abs(a-b)))]
EXT=BASE+EXTRA

class S:
    def __init__(s,ex,q,ops,budget=250000):
        s.ex=ex; s.q=q; s.ops=ops; s.map={}; s.used=set(); s.opa={}
        s.budget=budget; s.nodes=0; s.answers=Counter(); s.maxans=80; s.opdata=None; s.first_map=None
    def solve(s):
        s._p(0)
        if s.answers:
            best,_=s.answers.most_common(1)[0]
            return best
        return None
    def _vals(s,sy):
        if sy in s.map: return (s.map[sy],)
        return [d for d in range(10) if d not in s.used]
    def _as(s,sy,d):
        if sy in s.map: return False if s.map[sy]==d else None
        if d in s.used: return None
        s.map[sy]=d; s.used.add(d); return True
    def _un(s,sy,w):
        if w is True: s.used.discard(s.map[sy]); del s.map[sy]
    def _p(s,idx):
        if s.nodes>s.budget or len(s.answers)>=s.maxans: return
        s.nodes+=1
        if idx==len(s.ex): s._q(); return
        a0,a1,op,b0,b1,rs=s.ex[idx]
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
                        tryops=[s.opa[op]] if op in s.opa else range(len(s.ops))
                        for oi in tryops:
                            res=s.ops[oi][1](l,r)
                            if res is None: continue
                            rd=n2d(res)
                            if len(rd)!=len(rs): continue
                            ass=[]; ok=True
                            for c,dg in zip(rs,rd):
                                x=s._as(c,dg)
                                if x is None: ok=False;break
                                ass.append((c,x))
                            if ok:
                                new=op not in s.opa
                                if new: s.opa[op]=oi
                                s._p(idx+1)
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
        ql=s.map[q0]*10+s.map[q1]; qr=s.map[q3]*10+s.map[q4]
        opc=[s.opa[qop]] if qop in s.opa else range(len(s.ops))
        d2s={}
        for c,d in s.map.items():
            if d not in d2s: d2s[d]=c
        for oi in opc:
            res=s.ops[oi][1](ql,qr)
            if res is None: continue
            parts=[];ok=True
            for d in n2d(res):
                if d not in d2s: ok=False;break
                parts.append(d2s[d])
            if not ok: continue
            ans=''.join(parts)
            s.answers[ans]+=1
            if s.opdata is None:
                od=defaultdict(list)
                for (a0,a1,op,b0,b1,rs) in s.ex:
                    L=s.map[a0]*10+s.map[a1]; R=s.map[b0]*10+s.map[b1]
                    RES=int(''.join(str(s.map[c]) for c in rs))
                    od[op].append((L,R,RES))
                s.opdata=dict(od); s.first_map=dict(s.map)

def verify(stored,pred):
    import re
    stored=stored.strip();pred=pred.strip()
    if re.fullmatch(r'[01]+',stored): return pred==stored
    try: return math.isclose(float(stored),float(pred),rel_tol=1e-2,abs_tol=1e-5)
    except: return pred==stored

probs=load()
print('total:',len(probs),flush=True)
base_corr=ext_corr=base_any=ext_any=0
needs_extra=[]; extra_op_unlock=Counter(); still_unsolved=[]
for i,d in enumerate(probs):
    if i%100==0: print('...',i,flush=True)
    p=parse(d)
    if p is None: continue
    ex,q=p; gt=str(d['answer'])
    sb=S(ex,q,BASE,budget=300000); ab=sb.solve()
    if ab is not None:
        base_any+=1
        if verify(gt,ab): base_corr+=1
    se=S(ex,q,EXT,budget=300000); ae=se.solve()
    if ae is not None:
        ext_any+=1
        if verify(gt,ae): ext_corr+=1
        if ab is None and se.opdata:
            # which extra ops were needed
            used=set()
            for op,tr in se.opdata.items():
                for name,fn in EXT:
                    if all(fn(L,R)==RES for L,R,RES in tr):
                        used.add(name);break
            extra=used-set(n for n,_ in BASE)
            if extra:
                needs_extra.append((d['id'],sorted(extra)))
                for o in extra: extra_op_unlock[o]+=1
    else:
        still_unsolved.append(d['id'])
print('=== RESULTS ===',flush=True)
print('BASE 5-op: any-consistent=%d  correct=%d'%(base_any,base_corr),flush=True)
print('EXT  ops : any-consistent=%d  correct=%d'%(ext_any,ext_corr),flush=True)
print('newly solved by extra ops (base failed, ext found):',len(needs_extra),flush=True)
print('extra-op unlock counts:',extra_op_unlock.most_common(),flush=True)
print('still unsolved by EXT:',len(still_unsolved),flush=True)
