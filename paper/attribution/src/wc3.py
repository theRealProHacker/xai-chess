import re,sys
t=open(sys.argv[1]).read().split("## References")[0]
t=re.sub(r'<!--.*?-->','',t,flags=re.S)
tot={'main':0,'app':0}
for s in re.split(r"\n(?=## )",t):
    head=s.strip().splitlines()[0]; body=s.split('\n',1)[1] if '\n' in s else ''
    lines=[l for l in body.splitlines() if not re.fullmatch(r'\|[\s|:-]*',l.strip() or 'x')]
    toks=[w for w in ' '.join(lines).split() if not re.fullmatch(r'[|$]+|[-–]',w)]
    k='app' if 'Appendix' in head else ('main' if re.match(r'## \d',head) else 'x')
    if k in tot: tot[k]+=len(toks)
    print(f"{len(toks):6d}  {head[:60]}")
print(tot)
