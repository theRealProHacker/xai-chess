import json, subprocess, collections
from concurrent.futures import ThreadPoolExecutor
C={c['id']:c for c in map(json.loads, open('annotate/data/cases.jsonl'))}
A=[json.loads(l) for l in open('annotate/data/attempts.jsonl')]
last={}
for a in A:
    k=a['case']
    if k not in last or a.get('round',1)>=last[k].get('round',1): last[k]=a
def run(a):
    c=C[a['case']]
    r=subprocess.run(['dsl_verifier/target/release/dsl_verifier','check'],input=json.dumps({'fen':c['fen'],'mov':c['mov'],'dsl':a['dsl']}),capture_output=True,text=True)
    o=json.loads(r.stdout.strip().splitlines()[-1]); return a, o.get('v','error'), o.get('why') or o.get('error')
with ThreadPoolExecutor(12) as ex: R=list(ex.map(run, last.values()))
print(collections.Counter((a['v'], v) for a,v,_ in R))
for a,v,w in R:
    if a['v']!=v: print(f"{a['case']:<14} r{a.get('round',1)} {a['v']} -> {v}: {(w or '')[:110]}")
