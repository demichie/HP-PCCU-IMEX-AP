from pathlib import Path
import sys, csv
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import hp_pccu_1d_core as hp
import hp_pccu_1d_blended_ap as ap
OUT=Path(__file__).resolve().parent/'results'; OUT.mkdir(exist_ok=True)

def reflecting(side,t,q_edge,p):
    q=np.array(q_edge,float,copy=True); q[1]*=-1.0; return q

def run_one(N):
    L=4.; dx=L/N; xf=np.linspace(0,L,N+1); x=.5*(xf[:-1]+xf[1:])
    Bf=0.10*np.sin(np.pi*xf/2)+0.035*np.sin(np.pi*xf+0.3)
    B=hp.continuous_bed_geometry(Bf); Bc=hp.bed_center_values(B)
    eta0=1+0.06*np.cos(np.pi*x/2)+0.02*np.cos(np.pi*x)
    h0=eta0-Bc; p=hp.SolverParams(cfl=.24); q=np.array([hp.pure_liquid_state(float(h),0,p) for h in h0])
    r=ap.NewtonianRheology(mu=500.); t=0.; outs=[.5,1.,2.]; k=0; saved={0.:q.copy()}; dts=[]; full=[]
    while t<2-1e-13:
        cap=2-t
        if k<len(outs): cap=min(cap,outs[k]-t)
        cache=ap.prepare_cache(q,B,dx,t,0,p,reflecting,r,q_is_sanitized=True)
        speed=float(np.max(np.maximum(np.abs(cache['a_minus']),np.abs(cache['a_plus']))))
        df=np.inf if speed<=1e-14 else .99*p.cfl*dx/speed
        q,d=ap.step_imex_euler(q,B,dx,t,cap,p,reflecting,r,True); t+=d['dt'];dts.append(d['dt']);full.append(min(df,cap))
        while k<len(outs) and t>=outs[k]-1e-11: saved[outs[k]]=q.copy();k+=1
    h=hp.primitive(q,p)[3]; eta=h+Bc; u=hp.primitive(q,p)[1]
    np.savez(OUT/f'relax_lake_N{N}.npz',x=x,B=Bc,eta=eta,u=u,**{f'eta_t{tt}':hp.primitive(qq,p)[3]+Bc for tt,qq in saved.items()})
    return dict(N=N,mean_dt=float(np.mean(dts)),mean_full_dt=float(np.mean(full)),Gmean=float(np.mean(dts)/np.mean(full)),mass=float(np.sum(h)*dx),eta_L2=float(np.sqrt(np.mean((eta-1.)**2))),umax=float(np.max(np.abs(u))))

if __name__=='__main__':
    rows=[run_one(N) for N in (20,40,80)]
    for i in range(2):
        a=np.load(OUT/f"relax_lake_N{rows[i]['N']}.npz")['eta']; b=np.load(OUT/f"relax_lake_N{rows[i+1]['N']}.npz")['eta']
        br=b.reshape(len(a),2).mean(1); rows[i]['selfdiff']=float(np.sum(np.abs(a-br))/np.sum(np.abs(br-1.)+1e-30))
    with open(OUT/'relaxation_lake_summary.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*[r.keys() for r in rows])));w.writeheader();w.writerows(rows)
    for r in rows: print(r)
