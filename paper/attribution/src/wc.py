import re,sys
t=open(sys.argv[1]).read()
t=t.split("## References")[0]
secs=re.split(r"\n(?=## )",t)
tot={'main':0,'app':0}
for s in secs:
    head=s.strip().splitlines()[0]
    body=[]; inmath=False
    for l in s.splitlines()[1:]:
        if l.startswith('$$'):
            if not l.rstrip().endswith('$$') or l.strip()=='$$': inmath=not inmath
            continue
        if inmath or l.startswith('|') or l.startswith('!['): continue
        body.append(l)
    w=len(re.sub(r'\$[^$]*\$','X',' '.join(body)).split())
    k='app' if 'Appendix' in head else ('main' if re.match(r'## \d',head) else 'other')
    if k in tot: tot[k]+=w
    print(f"{w:6d}  {head}")
print(tot)
