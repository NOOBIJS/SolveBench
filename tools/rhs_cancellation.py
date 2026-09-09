import numpy as np, scipy.io as sio, scipy.sparse as sp, glob, os, json
rows=[]
for f in sorted(glob.glob('dataset_large/*/*.mtx')):
    name=os.path.basename(f)[:-4]
    try:
        A=sio.mmread(f)
        if not sp.issparse(A): A=sp.csr_matrix(A)
        A=A.tocsr().astype(float)
    except Exception as e:
        rows.append((name,None,'load_fail')); continue
    n=A.shape[0]
    if A.shape[0]!=A.shape[1]: rows.append((name,None,'nonsquare')); continue
    one=np.ones(n)
    b=A@one                       # exactly the harness RHS
    babs=abs(A)@one               # magnitude sum, no cancellation
    nb=np.linalg.norm(b); na=np.linalg.norm(babs)
    ratio = nb/na if na>0 else np.nan
    rows.append((name,float(ratio),'ok'))
ok=[r for r in rows if r[2]=='ok' and r[1]==r[1]]
ok.sort(key=lambda r:r[1])
print('matrices measured:', len(ok))
for t in [1e-8,1e-10,1e-12,1e-14]:
    print(f'  ratio < {t:.0e}: {sum(1 for r in ok if r[1]<t)}')
print('\n--- 10 worst ---')
for nm,r,_ in ok[:10]: print(f'  {nm:<22} {r:.3e}')
json.dump({n:r for n,r,_ in ok}, open('C:/Users/User/AppData/Local/Temp/claude/e--L-4-T-1-Study-Materials-Numerical-Analysis-Lab-Project/b308eec6-b765-49f4-ad79-d925f85fcdc2/scratchpad/cancel.json','w'))
