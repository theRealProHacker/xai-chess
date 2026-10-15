import re,sys
t=open(sys.argv[1]).read().split("## References")[0]
t=re.sub(r'<!--.*?-->','',t,flags=re.S)
tot={'main':0,'app':0}
for s in re.split(r"\n(?=## )",t):
    head=s.strip().splitlines()[0]; body=s.split('\n',1)[1] if '\n' in s else ''
    w=len(body.split()); wp=len(' '.join(l for l in body.splitlines() if not l.startswith('|')).split())
    k='app' if 'Appendix' in head else ('main' if re.match(r'## \d',head) else 'x')
    if k in tot: tot[k]+=w
    print(f"{w:6d} {wp:6d}  {head[:60]}")
print(tot)
