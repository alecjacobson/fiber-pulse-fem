#!/usr/bin/env python3
"""Publish numerical-method accuracy/cost comparisons and pulse diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert

COLORS={'leapfrog':'#177eab','rk4':'#219653','midpoint':'#d85c27','cn':'#934cba','theta':'#777777'}
def label(r):
    q=r['case'];m=q['method']
    if m=='theta':m+=f" {q.get('theta',.55):g}"
    if 'backend' in q:m+=' '+q['backend']
    return m

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path('results/integrators'));ap.add_argument('--out',type=Path,default=Path('artifacts/integrators'))
    ap.add_argument('--archive',type=Path,help='Regenerate figures from committed measurements and selected traces')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows=json.loads(a.archive.read_text())['rows'] if a.archive else []
    for path in ([] if a.archive else sorted(a.root.glob('*/metrics.json'))):
        r=json.loads(path.read_text());r['name']=path.parent.name
        if 'gpu_seconds' in r:rows.append(r)
    fine=[r for r in rows if r['case']['h']==.005 and r['case']['method']=='leapfrog']
    reference=min(fine,key=lambda r:r['dt_fs']) if fine else min(rows,key=lambda r:(r['case']['h'],r['dt_fs']))
    # Report both spatial+temporal error against a refined run and pure temporal
    # error against each mesh's small-step RK4 (or small-step leapfrog) control.
    for r in rows:
        h=r['case']['h']
        same=[x for x in rows if x['case']['h']==h and x['case']['method']=='rk4']
        if not same:same=[x for x in rows if x['case']['h']==h and x['case']['method']=='leapfrog']
        local=min(same,key=lambda x:x['dt_fs'])
        r['same_mesh_reference']=local['name']
        for key in ['modal_transmission','total_guided_transmission','delay_fs']:
            r[key+'_reference_error']=r[key]/reference[key]-1
            r[key+'_temporal_error']=r[key]/local[key]-1
        r['sample_interval_fs']=max(1,round(r['run']['sample_fs']/r['dt_fs']))*r['dt_fs']
        r['quality_score']=max(abs(r['beta_relative_error'])/.001,
            abs(r['modal_transmission_reference_error'])/.005,
            abs(r['total_guided_transmission_reference_error'])/.005,
            abs(r['delay_fs']-reference['delay_fs'])/.1)
        r['admissible']=r['quality_score']<=1 and r['status']=='ok' and r['tail_fraction']<1e-3
    tiers={}
    for tier,beta,power,arrival in [('preview',.01,.02,2.),('balanced',.002,.005,.4),('strict',.001,.005,.1)]:
        acceptable=[r for r in rows if r['status']=='ok' and r['tail_fraction']<1e-3
                    and abs(r['beta_relative_error'])<=beta
                    and abs(r['modal_transmission_reference_error'])<=power
                    and abs(r['total_guided_transmission_reference_error'])<=power
                    and abs(r['delay_fs']-reference['delay_fs'])<=arrival]
        winner=min(acceptable,key=lambda r:r['gpu_seconds']) if acceptable else None
        tiers[tier]=dict(beta_limit=beta,power_relative_limit=power,arrival_limit_fs=arrival,
                         winner=winner['name'] if winner else None)
    valid=[r for r in rows if r['admissible']]
    best=min(valid,key=lambda r:r['gpu_seconds']) if valid else None
    (a.out/'measurements.json').write_text(json.dumps(dict(reference=reference['name'],best=best['name'] if best else None,tiers=tiers,rows=rows),indent=2)+'\n')
    flat=[]
    for r in rows:
        row={k:v for k,v in r.items() if isinstance(v,(int,float,str,bool))}
        row.update({f'case_{k}':v for k,v in r['case'].items()});flat.append(row)
    keys=sorted(set().union(*(r.keys() for r in flat)))
    with (a.out/'measurements.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(flat)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    hs=sorted(set(r['case']['h'] for r in rows),reverse=True)
    fig,axs=plt.subplots(len(hs),3,figsize=(14,3.1*len(hs)),squeeze=False,layout='constrained')
    for axrow,h in zip(axs,hs):
        data=[r for r in rows if r['case']['h']==h and r['case']['method']!='theta']
        for lab in sorted(set(map(label,data))):
            group=sorted([r for r in data if label(r)==lab],key=lambda r:r['dt_fs'])
            # Tolerance controls at an identical dt are in the full table/frontier.
            group=[r for r in group if 'tolerance' not in r['case'] and 'maxiter' not in r['case']]
            if not group:continue
            x=[r['dt_fs'] for r in group];color=COLORS[group[0]['case']['method']]
            style='s--' if group[0]['case'].get('backend')=='cudss' else 'o-'
            for ax,y in zip(axrow,[[100*r['beta_relative_error'] for r in group],[-10*np.log10(max(r['total_guided_transmission'],1e-30)) for r in group],[r['delay_fs'] for r in group]]):
                ax.plot(x,y,style,label=lab,color=color,ms=4,lw=1);ax.set_xscale('log');ax.grid(alpha=.2)
        for ax,title,ylabel in zip(axrow,['Propagation constant','Guided insertion loss','Pulse arrival delay'],['beta error [%]','loss [dB]','delay [fs]']):
            ax.set(title=f'{h*1000:g} nm: {title}',xlabel='time step [fs]',ylabel=ylabel)
        axrow[0].legend(fontsize=7,ncol=2)
    fig.suptitle('Nondissipative methods and RK4; dissipative controls shown separately')
    fig.savefig(a.out/'time_step_trends.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    controls=[r for r in rows if r['case']['method']=='theta']
    ax.bar([f"{r['case']['h']*1000:g} nm, theta={r['case']['theta']:g}" for r in controls],
           [-10*np.log10(max(r['total_guided_transmission'],1e-30)) for r in controls],color='#777777')
    ax.set(ylabel='guided insertion loss [dB]',title='Stable does not imply accurate: dissipative controls at dt = 0.05 fs')
    fig.savefig(a.out/'damping_controls.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for h in hs:
        for method in COLORS:
            data=[r for r in rows if r['case']['h']==h and r['case']['method']==method]
            if not data:continue
            for ax,key,ylabel in zip(axs,['quality_score','total_guided_transmission_reference_error','delay_fs_reference_error'],['worst normalized error (≤1 passes)','guided transmission error [%]','arrival-delay error [%]']):
                ys=[max(abs(r[key])*(1 if key=='quality_score' else 100),1e-5) for r in data]
                ax.scatter([r['gpu_seconds'] for r in data],ys,c=COLORS[method],s=25+300*h,
                           marker='x' if method=='theta' else 'o',alpha=.7,label=f'{method} {h*1000:g} nm')
                ax.set(xscale='log',yscale='log',xlabel='trajectory [s]',ylabel=ylabel);ax.grid(alpha=.2)
    rejected=[r for r in rows if r['status']!='ok']
    for ax,key in zip(axs,['quality_score','total_guided_transmission_reference_error','delay_fs_reference_error']):
        if rejected:
            ax.scatter([r['gpu_seconds'] for r in rejected],
                       [max(abs(r[key])*(1 if key=='quality_score' else 100),1e-5) for r in rejected],
                       marker='x',color='black',s=65,label='solver gate failed')
    axs[0].axhline(1,color='black',ls=':',lw=1)
    if best:
        axs[0].annotate(f"fastest admissible\n{best['case']['method']}, {best['case']['h']*1000:g} nm",(best['gpu_seconds'],best['quality_score']),xytext=(-110,32),textcoords='offset points',fontsize=8,arrowprops=dict(arrowstyle='->',color=COLORS['leapfrog']))
        axs[0].scatter([best['gpu_seconds']],[best['quality_score']],marker='*',s=100,color=COLORS['leapfrog'],edgecolors='black',zorder=5)
    axs[2].legend(fontsize=6,ncol=2)
    fig.suptitle('Accuracy against analytic beta and refined spatial reference; time includes diagnostics')
    fig.savefig(a.out/'accuracy_cost.png',dpi=160);plt.close(fig)
    # All methods on the 20 nm mesh illustrate physical damping versus temporal dispersion.
    chosen=[]
    for method in ['leapfrog','rk4','midpoint','cn','theta']:
        group=[r for r in rows if r['case']['h']==.02 and r['case']['method']==method and r['case'].get('backend','cg')=='cg']
        if group:chosen.append(min(group,key=lambda r:abs(r['dt_fs']-.05)))
    fig,axs=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    for r in chosen:
        path=a.root/r['name']/'traces.npz'
        if not path.exists():path=(a.archive.parent if a.archive else a.out)/'traces'/f"{r['name']}.npz"
        p=np.load(path);tr=p['trace'];t=p['time_fs']
        neff=r['beta_per_um']/(1+r['beta_relative_error'])/(2*np.pi/.532)
        fw=.5*(tr[:,:,0]-tr[:,:,1]/neff);en=np.abs(hilbert(fw,axis=0))**2
        scale=en[:,3].max()
        axs[0].plot(t,en[:,9]/scale,label=f"{label(r)}, dt={r['dt_fs']:.3f} fs",color=COLORS[r['case']['method']])
        cumulative=np.cumsum(fw[:,9]**2)/np.sum(fw[:,3]**2)
        axs[1].plot(t,cumulative,color=COLORS[r['case']['method']])
    axs[0].set(xlabel='time [fs]',ylabel='output TE0 intensity / peak input',xlim=(70,160));axs[0].legend(fontsize=8)
    axs[1].set(xlabel='time [fs]',ylabel='accumulated transmitted TE0 energy',xlim=(60,240))
    for ax in axs:ax.grid(alpha=.2)
    fig.savefig(a.out/'pulse_comparison.png',dpi=160);plt.close(fig)
    print('Tiers:',json.dumps(tiers));print('Reference:',reference['name']);print('Best:',best['name'] if best else 'none')
    for r in sorted(rows,key=lambda r:(r['case']['h'],r['gpu_seconds'])):
        print(f"{r['name']}: {r['gpu_seconds']:.3f}s beta={r['beta_relative_error']:.5%} T0={r['modal_transmission']:.6f} Tg={r['total_guided_transmission']:.6f} delay={r['delay_fs']:.4f} score={r['quality_score']:.2f} {r['status']}")
if __name__=='__main__':main()
