from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import contourpy
import numpy as np
import pandas as pd
from protocol import ROOT, THEORY_OUTPUT, CHECK_OUTPUT, GRID_STEP, MF_R_MIN, MF_R_MAX, MF_EXTRA_R, MF_K_MIN, MF_K_MAX, production_validate
import hashlib
import tempfile

REPO_ROOT = ROOT
OUT_DIR = THEORY_OUTPUT

def quadrature(r: float, n: int, span: float = 8.0):
    if r <= 0:
        raise ValueError("Stationary scan requires r > 0; this script focuses on r=3.8..6.")
    a = np.linspace(-span * r, span * r, n)
    density = np.exp(-0.5 * (a / r)**2) / (np.sqrt(2*np.pi) * r)
    w = density * np.gradient(a)
    w[[0, -1]] *= 0.5
    return a, w / w.sum()
def h1(B):
    B = np.asarray(B, dtype=float)
    q = np.minimum(B, 1.0)
    root = np.sqrt(np.maximum(1 - q ** 2, 0.0))
    small = 1j * q / (1.0 + root)
    big = np.sqrt(np.maximum(1.0 - np.maximum(B, 1e-14) ** -2, 0.0)) + 1j / np.maximum(B, 1e-14)
    return np.where(B >= 1.0, big, small)

def h2(B):
    B = np.asarray(B, dtype=float)
    q = np.minimum(B, 1.0)
    root = np.sqrt(np.maximum(1 - q ** 2, 0.0))
    small = -(q / (1.0 + root)) ** 2
    inv = 1.0 / np.maximum(B, 1e-14)
    big = 1.0 - 2.0 * inv ** 2 + 2j * np.sqrt(np.maximum(1.0 - inv ** 2, 0.0)) * inv
    return np.where(B >= 1.0, big, small)

def response(Z, a, w, block=128):
    z = np.atleast_1d(Z).astype(complex)
    out = np.empty(len(z), dtype=complex)
    for lo in range(0, len(z), block):
        hi = min(lo+block, len(z))
        W = a[None, :] + z[lo:hi, None]
        B = np.abs(W)
        out[lo:hi] = (w[None, :] * W / np.maximum(B, 1e-14) * h1(B)).sum(axis=1)
    return out
def response_one(z, a, w):
    return response(np.array([z]), a, w)[0]
def observables(Z, a, w):
    W = a + Z
    B = np.abs(W)
    u = W / np.maximum(B, 1e-14)
    R1 = abs((w*u*h1(B)).sum())
    R2 = abs((w*u**2*h2(B)).sum())
    D = float((w * np.where(B < 1, np.sqrt(np.maximum(1-B**2, 0)), 0.)).sum())
    fp = float((w * (B >= 1)).sum())
    return float(R1), float(R2), D, fp
def stationary_residual(z, kappa, a, w):
    zres = z - kappa * response_one(z, a, w)
    return np.array([zres.real, zres.imag], dtype=float)
def stationary_jacobian(z, kappa, a, w, h=2e-4):
    dI_dx = (response_one(z+h, a, w) - response_one(z-h, a, w))/(2*h)
    dI_dy = (response_one(z+1j*h, a, w) - response_one(z-1j*h, a, w))/(2*h)
    return np.eye(2) - kappa * np.array([[dI_dx.real, dI_dy.real],
                                         [dI_dx.imag, dI_dy.imag]])
def newton_system(v, fun, tol=3e-9, max_iter=24):
    v = np.asarray(v, dtype=float).copy()
    for iteration in range(max_iter):
        f = fun(v)
        err = np.linalg.norm(f, ord=np.inf)
        if err < tol:
            return v, float(err)
        J = np.empty((len(v), len(v)))
        for j in range(len(v)):
            h = 1e-4 * max(1., abs(v[j]))
            e = np.zeros(len(v)); e[j] = h
            J[:,j] = (fun(v+e) - fun(v-e))/(2*h)
        try:
            delta = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"Newton Jacobian singular near {v}") from exc
        accepted = False
        for back in range(17):
            factor = 2.0**(-back)
            test = v + factor*delta
            if len(test) == 3 and test[2] <= 0:
                continue
            ft = fun(test)
            if np.isfinite(ft).all() and np.linalg.norm(ft, ord=np.inf) < err:
                v = test; accepted = True; break
        if not accepted:
            raise RuntimeError(f"Newton line search failed: v={v}, residual={err:g}")
    raise RuntimeError(f"Newton failed to converge: v={v}, residual={np.linalg.norm(fun(v),ord=np.inf):g}")
def refine_root(z0, kappa, a, w):
    v, err = newton_system([z0.real, z0.imag],
        lambda x: stationary_residual(complex(x[0],x[1]), kappa, a, w))
    z = complex(*v)
    return z, err
def refine_fold(z0, k0, a, w):
    def conditions(v):
        z = complex(v[0],v[1]); k = float(v[2])
        f = stationary_residual(z, k, a, w)
        det = np.linalg.det(stationary_jacobian(z, k, a, w))
        return np.array([f[0], f[1], det])
    v, err = newton_system([z0.real, z0.imag, k0], conditions, tol=8e-8)
    z = complex(v[0],v[1]); k = float(v[2])
    if err > 8e-8 or not np.isfinite(k) or k <= 0:
        raise RuntimeError(f"Fold refinement failed (k={k}, residual={err:g})")
    return z, k, err
def pseudo_arclength_branch(contour, a, w, ds=0.06, max_steps=2500):
    cZ = contour['z']; cK = contour['k']
    j = max(1, len(cZ)//4)
    z, err = refine_root(cZ[j], float(cK[j]), a, w)
    start = np.array([z.real, z.imag, float(cK[j])])
    def jac23(v):
        zz=complex(v[0],v[1]); kk=float(v[2])
        J=stationary_jacobian(zz,kk,a,w)
        I=response_one(zz,a,w)
        return np.column_stack([J, -np.array([I.real,I.imag])])
    def tangent(v, reference=None):
        vh=np.linalg.svd(jac23(v),full_matrices=True)[2]
        t=vh[-1,:]
        if reference is not None and np.dot(t,reference)<0: t=-t
        return t/np.linalg.norm(t)
    direction=np.array([ (cZ[(j+1)%len(cZ)]-cZ[j]).real,
                         (cZ[(j+1)%len(cZ)]-cZ[j]).imag,
                         cK[(j+1)%len(cK)]-cK[j]])
    t0=tangent(start,direction)
    v=start.copy(); t=t0.copy()
    points=[start.copy()]
    approx=np.column_stack((cZ.real,cZ.imag,cK))
    perimeter=float(np.linalg.norm(np.diff(approx,axis=0),axis=1).sum())
    total=0.0
    for it in range(max_steps):
        predictor=v+ds*t
        vv=predictor.copy()
        corrected=False
        for tries in range(4):
            def f3(q):
                zz=complex(q[0],q[1]);kk=float(q[2])
                f=stationary_residual(zz,kk,a,w)
                return np.array([f[0],f[1],np.dot(q-predictor,t)])
            try:
                for n in range(14):
                    f=f3(vv)
                    if np.linalg.norm(f,ord=np.inf)<1e-9:
                        corrected=True; break
                    JJ=np.vstack((jac23(vv),t))
                    delta=np.linalg.solve(JJ,-f)
                    vv=vv+delta
                    if not np.isfinite(vv).all() or vv[2]<=0:
                        raise ArithmeticError('continuation step invalid')
            except (np.linalg.LinAlgError, ArithmeticError):
                pass
            if corrected: break
            ds*=0.5
            if ds<0.002: break
            predictor=v+ds*t
            vv=predictor.copy()
        if not corrected:
            raise RuntimeError(f'Pseudo-arclength corrector did not converge near {v}')
        total += float(np.linalg.norm(vv-v))
        v=vv
        t=tangent(v,t)
        points.append(v.copy())
        if total>0.7*perimeter and np.linalg.norm(v-start)<1.3*ds and np.dot(t,t0)>0.8:
            return np.asarray(points),total
    raise RuntimeError('Pseudo-arclength branch did not close within step budget; no final dataset exported')
def pinned_stability_data(Z, kappa, a, w):
    W = a + Z
    B = np.abs(W)
    if np.any(B <= 1.0):
        raise RuntimeError("Partially drifting stationary branch requires a separate kinetic stability analysis")
    lam = np.sqrt(B * B - 1.0)
    theta = np.angle(W) + np.arcsin(1.0 / B)
    c = np.cos(theta)
    s = np.sin(theta)
    return lam, c, s

def stability_matrix(sigma, kappa, lam, c, s, w):
    inv = w / (sigma + lam)
    cc = np.dot(inv, c * c)
    ss = np.dot(inv, s * s)
    cs = np.dot(inv, c * s)
    return kappa * np.array([[cc, cs], [cs, ss]], dtype=float)

def instability_eigenvalues(Z, kappa, a, w):
    lam, c, s = pinned_stability_data(Z, kappa, a, w)
    vals = np.linalg.eigvalsh(stability_matrix(0.0, kappa, lam, c, s, w))
    if np.min(np.abs(vals - 1.0)) < 1e-8:
        raise RuntimeError("Marginal stationary root: resolve stability at higher numerical precision")
    return vals

def unstable_count(Z, kappa, a, w):
    return int(np.count_nonzero(instability_eigenvalues(Z, kappa, a, w) > 1.0))

def leading_real_eigenvalue(Z, kappa, a, w):
    lam, c, s = pinned_stability_data(Z, kappa, a, w)
    def f(sig):
        return np.linalg.eigvalsh(stability_matrix(sig, kappa, lam, c, s, w))[-1] - 1.0
    if f(0.0) <= 0.0:
        return np.nan
    lo, hi = 0.0, max(1.0, float(kappa))
    for _ in range(30):
        if f(hi) < 0.0:
            break
        hi *= 2.0
    else:
        raise RuntimeError("Could not bracket a positive real stability eigenvalue")
    for _ in range(44):
        mid = (lo + hi) / 2.0
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)

def contour_branches(a, w, rho_max, radial_step, nphi):
    nrho = int(np.ceil((rho_max-0.02)/radial_step))+1
    rho = np.linspace(0.02,rho_max,nrho)
    phi = np.linspace(-np.pi/2, 5*np.pi/2, nphi)
    E = np.exp(1j*phi)
    img = np.empty((nrho,nphi))
    for j,p in enumerate(rho):
        img[j] = (response(p*E,a,w)*np.conj(E)).imag
    lines = contourpy.contour_generator(x=phi,y=rho,z=img).lines(0.)
    curves=[]
    for seg in lines:
        ph,rh = seg[:,0],seg[:,1]
        z = rh*np.exp(1j*ph)
        G = response(z,a,w)*np.exp(-1j*ph)
        good = np.where(G.real > 1e-6)[0]
        for piece in np.split(good,np.where(np.diff(good)>1)[0]+1):
            if len(piece)<4: continue
            p = np.mod(ph[piece],2*np.pi)
            if np.median(p) >= np.pi: continue
            zc=z[piece]; k=rh[piece]/G.real[piece]
            curves.append(dict(z=zc,k=np.asarray(k),closed=abs(zc[0]-zc[-1])<1e-8,
                               end_rho=max(abs(zc[0]),abs(zc[-1]))))
    return curves,nrho
def find_complete_branches(r,a,w,p):
    rho_max = p['rho_start']; attempts=[]
    while True:
        curves,nrho = contour_branches(a,w,rho_max,p['radial_step'],p['nphi'])
        attempts.append(dict(rho_max=rho_max, nrho=nrho, branches=len(curves),
                             open=sum(not c['closed'] for c in curves)))
        open_curves=[c for c in curves if not c['closed']]
        if not open_curves:
            return curves,rho_max,nrho,attempts
        boundary_open=all(c['end_rho']>rho_max-2*p['radial_step'] for c in open_curves)
        if not boundary_open:
            raise RuntimeError(f"At r={r}, non-closed curves not explained by radial boundary: {attempts}. "
                               "Check angular seams/contour resolution before trusting the scan.")
        if rho_max*1.5 > p['rho_hard_limit']:
            raise RuntimeError(f"At r={r}, open branch still touches rho={rho_max}. "
                               f"Reached safety bound {p['rho_hard_limit']}; NO partial results exported.")
        rho_max = min(p['rho_hard_limit'],rho_max*2)
def fold_indices(k):
    k = np.asarray(k[:-1])
    n = len(k)
    if n<8: raise RuntimeError('Closed contour too short to locate both folds')
    minima=[]; maxima=[]
    for j in range(n):
        near = k[np.arange(j-3,j+4)%n]
        if k[j]==near.min() and np.count_nonzero(near==k[j])==1:
            minima.append(j)
        if k[j]==near.max() and np.count_nonzero(near==k[j])==1:
            maxima.append(j)
    if len(minima)!=1 or len(maxima)!=1:
        raise RuntimeError(f"Expected 1 lower+upper fold, found {len(minima)} minima "
                           f"and {len(maxima)} maxima; inspect curve rather than silently dropping folds")
    return [('lower',minima[0]),('upper',maxima[0])]
def analyse_r(job):
    r,p=job
    a,w=quadrature(r,p['nq'])
    curves,rho_used,nrho,attempts=find_complete_branches(r,a,w,p)
    kgrid=np.round(np.arange(p['kappa_min'],p['kappa_max']+1e-9,p['dkappa']),6)
    pts, folds, roots=[],[],[]
    pseudo_rows=[]
    max_root_res=0.; max_fold_res=0.
    for b,c in enumerate(curves):
        zarr,k=c['z'],c['k']
        pseudo, length = pseudo_arclength_branch(c,a,w,ds=p['arc_step'])
        for j,(x,y,kk) in enumerate(pseudo):
            pseudo_rows.append(dict(r=r,branch=b,index=j,kappa=kk,ReZ=x,ImZ=y,
                                    R1=np.hypot(x,y)/kk,
                                    residual=np.linalg.norm(stationary_residual(complex(x,y),kk,a,w),ord=np.inf)))
        for kind,j in fold_indices(k):
            zfold,kfold,err=refine_fold(zarr[j],k[j],a,w)
            if kfold<p['kappa_min']-1e-8: raise RuntimeError(f'Invalid fold at k={kfold}')
            max_fold_res=max(max_fold_res,err)
            folds.append(dict(r=r,branch=b,kind=kind,kappa=kfold,
                             R1=abs(zfold)/kfold,Theta_deg=np.degrees(np.angle(zfold))%360,
                             ReZ=zfold.real,ImZ=zfold.imag,
                             fold_residual=err,amplitude_scan_limit=rho_used))
        nu=[unstable_count(z,float(kk),a,w) for z,kk in zip(zarr,k)]
        for j,(z,kk,nunst) in enumerate(zip(zarr,k,nu)):
            R1,R2,D,fp=observables(z,a,w)
            res=np.linalg.norm(stationary_residual(z,float(kk),a,w),ord=np.inf)
            pts.append(dict(r=r,branch=b,index=j,kappa=float(kk),R1=R1,R2=R2,
                            Theta_deg=np.degrees(np.angle(z))%360,ReZ=z.real,ImZ=z.imag,
                            D_MF=D,f_pinned=fp,n_unstable=int(nunst),residual=float(res)))
        for kk in kgrid[kgrid>0]:
            crossed=np.where((k[:-1]-kk)*(k[1:]-kk)<=0)[0]
            already=[]
            for j in crossed:
                if k[j]==k[j+1]: continue
                t=(kk-k[j])/(k[j+1]-k[j])
                guess=zarr[j]+t*(zarr[j+1]-zarr[j])
                z,err=refine_root(guess,float(kk),a,w)
                if any(abs(z-other)<1e-7 for other in already): continue
                already.append(z)
                max_root_res=max(max_root_res,err)
                R1,R2,D,fp=observables(z,a,w)
                roots.append(dict(r=r,kappa=float(kk),branch=b,R1=R1,R2=R2,
                                  Theta_deg=np.degrees(np.angle(z))%360,
                                  ReZ=z.real,ImZ=z.imag,D_MF=D,f_pinned=fp,
                                  n_unstable=unstable_count(z,float(kk),a,w),
                                  sigma_max=leading_real_eigenvalue(z,float(kk),a,w),
                                  residual=err))
    diagnostics=dict(r=r,branches=len(curves),folds=len(folds),roots=len(roots),
                     used_rho_max=rho_used,used_nrho=nrho,grid_nphi=p['nphi'],
                     max_root_residual=max_root_res,max_fold_residual=max_fold_res,
                     attempts=json.dumps(attempts))
    if curves and len(folds)!=2*len(curves):
        raise RuntimeError(f"r={r}: not every closed branch has TWO folds: {diagnostics}")
    if any(x['n_unstable'] < 0 for x in pts) or any(x['n_unstable'] < 0 for x in roots):
        raise RuntimeError(f"r={r}: an unclassified stationary root was encountered")
    diagnostics['all_stationary_roots_stability_classified'] = True
    return r,pts,folds,roots,diagnostics,pseudo_rows
def existence_map(roots_df, r_values, kgrid):
    if len(roots_df) and (roots_df['n_unstable'] < 0).any():
        raise RuntimeError("Cannot label unassessed branches as unstable in the existence map")
    grouped={key:g for key,g in roots_df.groupby(['r','kappa'])} if len(roots_df) else {}
    rows=[]
    for r in r_values:
        for k in kgrid:
            if k<=0: continue
            d=grouped.get((r,k),None)
            if d is None:
                rows.append(dict(r=r,kappa=k,n_roots=0,n_stable=0,
                                 R1_stable=np.nan,R2_stable=np.nan,Theta_stable=np.nan))
                continue
            st=d[d.n_unstable==0]
            if len(st):
                best=st.loc[st.R1.idxmax()]
                rows.append(dict(r=r,kappa=k,n_roots=len(d),n_stable=len(st),
                                 R1_stable=best.R1,R2_stable=best.R2,Theta_stable=best.Theta_deg))
            else:
                rows.append(dict(r=r,kappa=k,n_roots=len(d),n_stable=0,
                                 R1_stable=np.nan,R2_stable=np.nan,Theta_stable=np.nan))
    return pd.DataFrame(rows)
def parse_args():
    p=argparse.ArgumentParser(description='Validated stationary mean-field two-fold analysis')
    p.add_argument('--r-min',type=float,default=MF_R_MIN)
    p.add_argument('--r-max',type=float,default=MF_R_MAX)
    p.add_argument('--dr',type=float,default=GRID_STEP)
    p.add_argument('--r-values',type=float,nargs='+',default=None)
    p.add_argument('--kappa-min',type=float,default=MF_K_MIN)
    p.add_argument('--kappa-max',type=float,default=MF_K_MAX)
    p.add_argument('--dkappa',type=float,default=GRID_STEP)
    p.add_argument('--rho-start',type=float,default=9.)
    p.add_argument('--rho-hard-limit',type=float,default=40.)
    p.add_argument('--radial-step',type=float,default=(9.-0.02)/359)
    p.add_argument('--nphi',type=int,default=720)
    p.add_argument('--arc-step',type=float,default=0.06,help='Pseudo-arclength predictor step; adaptive reduction on correction failure')
    p.add_argument('--nq',type=int,default=3001)
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--check',action='store_true',help='Run one r point with reduced contour grid for a plumbing test')
    p.add_argument('--tag',default='',help='Optional dataset suffix; outputs stored under data/theory/final')
    return p.parse_args()
def main():
    args=parse_args()
    production_validate()
    if not np.isclose(args.dr, GRID_STEP) or not np.isclose(args.dkappa, GRID_STEP):
        raise ValueError('Theory parameter-grid spacing must equal production spacing')
    if args.kappa_max <= 0 or args.kappa_min<0 or args.r_min<=0 or args.radial_step<=0:
        raise ValueError('Require positive r and kappa domain and positive radial-step')
    if args.r_values is not None:
        rv=np.asarray(args.r_values,dtype=float)
    else:
        rv=np.unique(np.round(np.r_[np.arange(args.r_min,args.r_max+1e-9,args.dr),MF_EXTRA_R],6))
    if args.check:
        rv=np.array([4.5])
        args.nq=151
        args.nphi=120
        args.radial_step=0.12
        args.dkappa=GRID_STEP
    params=dict(kappa_min=args.kappa_min,kappa_max=args.kappa_max,dkappa=args.dkappa,
                rho_start=args.rho_start,rho_hard_limit=args.rho_hard_limit,
                radial_step=args.radial_step,nphi=args.nphi,nq=args.nq,arc_step=args.arc_step)
    kg=np.round(np.arange(args.kappa_min,args.kappa_max+1e-9,args.dkappa),6)
    global OUT_DIR
    if args.check:
        OUT_DIR = CHECK_OUTPUT / "mf_theory"
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    manifest = OUT_DIR / "manifest.json"
    token = hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name("protocol.py").read_bytes() + repr((rv.tolist(), params, args.check, args.tag)).encode()).hexdigest()
    if manifest.exists():
        if json.loads(manifest.read_text()).get("fingerprint") != token:
            raise RuntimeError("Existing mean-field folder has different settings or source; refusing overwrite")
    elif any(OUT_DIR.iterdir()):
        raise RuntimeError("Existing mean-field folder without manifest; refusing overwrite")
    else:
        manifest.write_text(json.dumps(dict(fingerprint=token, parameters=params, r_values=rv.tolist(), check=args.check, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
    chunks = OUT_DIR / "_chunks"
    chunks.mkdir(exist_ok=True)
    def name(r):
        return chunks / ("r_" + format(float(r), ".5f") + ".json")
    def atomic_save(path, result):
        fd, temp = tempfile.mkstemp(prefix="_partial_", suffix=".json", dir=chunks)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(result, f, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
    jobs=[(float(r),params) for r in rv if not name(r).exists()]
    print(f"Mean-field: {len(rv)-len(jobs)}/{len(rv)} checkpointed r slices, {len(jobs)} remaining",flush=True)
    if args.jobs==1:
        iterator=map(analyse_r,jobs)
        pool=None
    elif jobs:
        pool=mp.get_context('spawn').Pool(processes=args.jobs)
        iterator=pool.imap_unordered(analyse_r,jobs)
    else:
        pool=None
        iterator=iter(())
    try:
        for r,pts,folds,roots,diag,pseudo in iterator:
            atomic_save(name(r), [float(r),pts,folds,roots,diag,pseudo])
            print(f"r={r:.3f}: {len({x['branch'] for x in pts})} branch(es), folds={[(f['kind'],round(f['kappa'],5)) for f in folds]}",flush=True)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    missing=[float(r) for r in rv if not name(r).exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} theory chunks: {missing}")
    all_pts=[];all_folds=[];all_roots=[];all_pseudo=[];diags=[]
    for r in rv:
        rr,pts,folds,roots,diag,pseudo=json.loads(name(r).read_text())
        if not np.isclose(rr,r,rtol=0,atol=1e-9):
            raise RuntimeError("Theory checkpoint r mismatch")
        all_pts.extend(pts)
        all_folds.extend(folds)
        all_roots.extend(roots)
        all_pseudo.extend(pseudo)
        diags.append(diag)
    if len(diags)!=len(rv):
        raise RuntimeError("Missing mean-field disorder slices")
    suffix='_'+args.tag if args.tag else ('_check' if args.check else '')
    cols_br=['r','branch','index','kappa','R1','R2','Theta_deg','ReZ','ImZ','D_MF','f_pinned','n_unstable','residual']
    cols_fold=['r','branch','kind','kappa','R1','Theta_deg','ReZ','ImZ','fold_residual','amplitude_scan_limit']
    cols_roots=['r','kappa','branch','R1','R2','Theta_deg','ReZ','ImZ','D_MF','f_pinned','n_unstable','sigma_max','residual']
    b=pd.DataFrame(all_pts,columns=cols_br).sort_values(['r','branch','index'])
    f=pd.DataFrame(all_folds,columns=cols_fold).sort_values(['r','branch','kind'])
    roots=pd.DataFrame(all_roots,columns=cols_roots).sort_values(['r','kappa','branch','R1'])
    diag=pd.DataFrame(diags).sort_values('r')
    if (diag['folds']!=2*diag['branches']).any():
        raise RuntimeError('Validation failure: missing folds; no final data written.')
    if (len(b) and (b['n_unstable'] < 0).any()) or (len(roots) and (roots['n_unstable'] < 0).any()):
        raise RuntimeError('Incomplete stationary-root stability classification; no final data written.')
    b.to_csv(OUT_DIR/f'mf_branches{suffix}.csv',index=False)
    f.to_csv(OUT_DIR/f'mf_folds{suffix}.csv',index=False)
    roots.to_csv(OUT_DIR/f'mf_roots_on_grid{suffix}.csv',index=False)
    existence_map(roots,rv,kg).to_csv(OUT_DIR/f'mf_existence_map{suffix}.csv',index=False)
    diag.to_csv(OUT_DIR/f'mf_scan_diagnostics{suffix}.csv',index=False)
    pd.DataFrame(all_pseudo,columns=['r','branch','index','kappa','ReZ','ImZ','R1','residual']).sort_values(['r','branch','index']).to_csv(OUT_DIR/f'mf_pseudo_arclength{suffix}.csv',index=False)
    print(f'Validated {len(rv)} r values, {len(f)} folds, {len(roots)} root samples.')
    (OUT_DIR / 'COMPLETE.json').write_text(json.dumps(dict(r_count=len(rv),fold_count=len(f),root_count=len(roots),fingerprint=token), indent=2))
    print('Saved to',OUT_DIR)
if __name__=='__main__':
    mp.freeze_support()
    main()
