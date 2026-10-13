#!/usr/bin/env python3
"""Reproduce speed plots and envelope animation from the archived measurements."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.signal import hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='artifacts/light_speed')
    p.add_argument('--no-animation',action='store_true')
    a = p.parse_args(); out = Path(a.out)
    report = json.loads((out/'measurements.json').read_text())
    c = report['c0_m_s']
    rows = report['homogeneous']
    guide = []
    for f in ['artifacts/resolution.json','artifacts/reference.json']:
        for m in json.loads(Path(f).read_text()):
            if m['mesh']['bend'] != 0: continue
            k0 = 2*np.pi/m['run']['wavelength_um']
            vp = c*k0/m['beta_per_um']
            vg = c/m['straight_group_index_measured']
            guide.append(dict(name=m['name'],h_um=m['mesh']['h'],
                expected_phase_speed_m_s=c*k0/m['beta_exact_per_um'],
                expected_group_speed_m_s=c/m['group_index'],
                phase_speed_m_s=vp,group_speed_m_s=vg,
                phase_relative_error=m['beta_exact_per_um']/m['beta_per_um']-1,
                group_relative_error=m['group_index']/m['straight_group_index_measured']-1,
                effective_index=m['beta_exact_per_um']/k0,group_index=m['group_index'],
                source_artifact=f,source_sha256=m['source_sha256']))
    guide.sort(key=lambda r:r['h_um'],reverse=True)
    (out/'straight_guide.json').write_text(json.dumps(guide,indent=2)+'\n')
    fig,axes = plt.subplots(2,2,figsize=(11,8),layout='constrained')
    colors = ['#2579b2','#dd8830','#339861']
    for n,col in zip(sorted(set(r['index'] for r in rows)),colors):
        rs = sorted([r for r in rows if r['index']==n],key=lambda r:r['h_um'],reverse=True)
        fine = rs[-1]
        z = np.load(out/f'n{n:g}_h{fine["h_um"]:g}.npz')
        x = z['x_um']; arrivals = np.array(fine['arrival_fs'])
        axes[0,0].plot(x,arrivals,'.',color=col,label=f'n = {n:g}')
        # Fit offset only: reference slope is fixed by c0/n.
        expected_slope=1e9*n/c
        offset=np.mean(arrivals-expected_slope*x)
        axes[0,0].plot(x,offset+expected_slope*x,'--',color=col,alpha=.7)
        for ax,key in [(axes[1,0],'phase_relative_error'),(axes[1,1],'group_relative_error')]:
            ax.loglog([r['h_um']*1000 for r in rs],[abs(r[key])*100 for r in rs],'.-',color=col,label=f'Uniform n={n:g}')
    z = np.load(out/'n1_h0.005.npz')
    axes[0,1].pcolormesh(z['x_um'],z['time_fs'],abs(hilbert(z['e'],axis=0)),shading='auto',cmap='magma',rasterized=True)
    axes[0,1].set(xlabel='Position (µm)',ylabel='Time (fs)',title='Vacuum: recorded pulse envelope (h = 5 nm)')
    axes[0,0].set(xlabel='Position (µm)',ylabel='Envelope arrival (fs)',title='Measured arrival; dashed slope = n/c₀ (h = 5 nm)')
    axes[0,0].legend()
    for ax,key,label in [(axes[1,0],'phase_relative_error','Phase speed'),(axes[1,1],'group_relative_error','Envelope speed')]:
        ax.loglog([r['h_um']*1000 for r in guide],[abs(r[key])*100 for r in guide],'s-',color='#914bb2',label='Straight slab TE₀')
        ax.set(xlabel='Mesh h (nm)',ylabel='Absolute relative error (%)',title=label+' convergence')
        ax.grid(True,which='both',alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out/'speed_check.png',dpi=170); plt.close(fig)
    if not a.no_animation:
        fig,ax = plt.subplots(figsize=(8,3.7),layout='constrained')
        fields=[];lines=[]
        for n,col in zip([1.,1.46,2.04],colors):
            z=np.load(out/f'n{n:g}_h0.005.npz')
            env=abs(hilbert(z['e'],axis=0));env/=env.max()
            fields.append((z['time_fs'],z['x_um'],env))
            line,=ax.plot(z['x_um'],env[0],color=col,lw=2,label=f'n = {n:g}')
            lines.append(line)
        label=ax.text(.02,.92,'',transform=ax.transAxes)
        ax.set(xlim=(12,16),ylim=(0,1.08),xlabel='Position (µm)',ylabel='Normalized envelope',
               title='Measured source-free propagation • 5 nm mesh')
        ax.legend(loc='upper right')
        def frame(t):
            for line,(ts,x,env) in zip(lines,fields):
                line.set_ydata([np.interp(t,ts,env[:,i],left=0,right=0) for i in range(len(x))])
            label.set_text(f't = {t:.1f} fs')
            return lines+[label]
        ani=FuncAnimation(fig,frame,frames=np.linspace(7,70,110),blit=True)
        ani.save(out/'envelopes.gif',writer=PillowWriter(fps=20));plt.close(fig)
    print('Homogeneous 5 nm results:')
    for r in rows:
        if r['h_um']==.005:print(r['index'],r['expected_speed_m_s'],r['phase_speed_m_s'],r['group_speed_m_s'])
    print('Finest straight slab:',guide[-1])


if __name__=='__main__':main()
