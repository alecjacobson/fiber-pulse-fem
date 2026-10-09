#!/usr/bin/env python3
"""Render saved GPU raster frames and study plots after simulation completion."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import imageio_ffmpeg
from scipy.signal import hilbert
from fiber_pulse import Z0

plt.rcParams.update({'figure.facecolor':'#101722','axes.facecolor':'#101722',
    'savefig.facecolor':'#101722','text.color':'#e3edf7','axes.labelcolor':'#e3edf7',
    'axes.edgecolor':'#778596','xtick.color':'#c4d1de','ytick.color':'#c4d1de',
    'grid.color':'#485261','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
CM_E = LinearSegmentedColormap.from_list('electric',['#33c5ff','#101722','#ffbe45'])


def animate(run,out):
    run,out = Path(run),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    data = np.load(run/'frames.npz')
    frames,times,extent = data['frames'],data['time_fs'],data['extent']
    mesh = np.load(run/'mesh.npz')
    tr = np.load(run/'traces.npz')
    m = json.loads((run/'metrics.json').read_text())
    fig,axs = plt.subplots(4,1,figsize=(12,9),dpi=100,gridspec_kw={'height_ratios':[1,1,1,0.7]},layout='constrained')
    images=[]
    maxima=[1,2.04*1e3/Z0,2.04*1e3/Z0]
    for ax,title,cmap,lim in zip(axs[:3],['Electric field $E_z$ [MV/m]','Magnetic magnitude $|H|$ [kA/m]',
                                        'Poynting magnitude $|S|$ [GW/m²]'],[CM_E,'viridis','magma'],maxima):
        im=ax.imshow(np.zeros(frames.shape[1:3]),extent=extent,origin='lower',cmap=cmap,
                     vmin=-lim if cmap==CM_E else 0,vmax=lim,aspect='equal')
        images.append(im)
        ol=mesh['outline']
        ax.plot(*np.vstack([ol,ol[:1]]).T,color='#b0c3d1',lw=0.5,alpha=0.8)
        ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
        ax.set_title(title,loc='left',fontsize=11);ax.set_ylabel('y [µm]')
        for x in [tr['ports'][3],tr['ports'][-1]]:
            ax.axvline(x,color='white',ls=':',lw=0.6,alpha=0.5)
        fig.colorbar(im,ax=ax,fraction=0.02,pad=0.02)
    axs[2].set_xlabel('x [µm]')
    fw=0.5*(tr['trace'][:,:,0]-tr['trace'][:,:,1]/(m['beta_exact_per_um']/(2*np.pi/0.532)))
    env=np.abs(hilbert(fw,axis=0))**2
    scale=env[:,3].max()
    axs[3].plot(tr['time_fs'],env[:,3]/scale,color='#33c5ff',label='input TE₀ envelope²')
    axs[3].plot(tr['time_fs'],env[:,-1]/scale,color='#ffbe45',label='output TE₀ envelope²')
    marker=axs[3].axvline(0,color='white',lw=1)
    axs[3].set(xlim=(0,times[-1]),xlabel='time [fs]',ylabel='relative intensity')
    axs[3].legend(loc='upper right',facecolor='#101722',labelcolor='white',frameon=False)
    title=fig.suptitle('',fontsize=14)
    fig.canvas.draw()
    width,height=fig.canvas.get_width_height()
    pipe=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
        '-s',f'{width}x{height}','-pix_fmt','rgba','-r','25','-i','-','-an','-vcodec','libx264','-crf','20',
        '-pix_fmt','yuv420p',str(out/'pulse.mp4')],stdin=subprocess.PIPE)
    snapshot=int(np.argmin(np.abs(times-70)))
    for i,(frame,t) in enumerate(zip(frames,times)):
        images[0].set_data(frame[:,:,0])
        images[1].set_data(frame[:,:,1]*1e3/Z0)
        images[2].set_data(frame[:,:,2]*1e3/Z0)
        marker.set_xdata([t,t])
        title.set_text(f'532 nm pulse · Si₃N₄ / SiO₂ Bézier slab · h = {1000*m["mesh"]["h"]:g} nm · t = {t:.1f} fs')
        fig.canvas.draw()
        pipe.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
        if i==snapshot:
            fig.savefig(out/'pulse_snapshot.png',dpi=150)
    pipe.stdin.close()
    if pipe.wait()!=0:
        raise RuntimeError('ffmpeg failed')
    plt.close(fig)
    # A small looping preview is convenient in repository Markdown.
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(out/'pulse.mp4'),
        '-vf','fps=12,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
        '-loop','0',str(out/'pulse.gif')],check=True)
    print('Rendered',out,flush=True)


def plots(root,out):
    root,out=Path(root),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((root/'resolution.json').read_text())
    ref=json.loads((root/'reference.json').read_text()) if (root/'reference.json').exists() else []
    allrows=rows+ref
    straight=sorted([r for r in allrows if r['mesh']['bend']==0],key=lambda r:r['mesh']['h'])
    bend=sorted([r for r in allrows if r['mesh']['bend']!=0],key=lambda r:r['mesh']['h'])
    rb=bend[0]
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    h=np.array([r['mesh']['h']*1000 for r in straight])
    axs[0,0].loglog(h,[abs(r['beta_relative_error'])*100 for r in straight],'o-',label='phase constant')
    axs[0,0].loglog(h,[abs(r['delay_relative_error'])*100 for r in straight],'s-',label='arrival delay')
    axs[0,0].set(ylabel='error against analytic slab [%]',title='Straight-guide accuracy');axs[0,0].legend()
    axs[0,1].semilogx(h,[r['modal_transmission'] for r in straight],'o-',label='TE₀ energy')
    axs[0,1].semilogx(h,[r['net_flux_transmission'] for r in straight],'s-',label='net flux')
    axs[0,1].axhline(1,color='white',lw=.7,ls=':')
    axs[0,1].set(ylabel='transmission ratio',title='Straight-guide conservation');axs[0,1].legend()
    bend=[r for r in bend if r['tail_fraction']<1e-3]
    hb=np.array([r['mesh']['h']*1000 for r in bend])
    axs[0,2].semilogx(hb,[r['modal_loss_db'] for r in bend],'o-',label='TE₀ insertion loss')
    axs[0,2].semilogx(hb,[-10*np.log10(r['total_guided_transmission']) for r in bend],'^-',label='all guided modes')
    axs[0,2].semilogx(hb,[-10*np.log10(r['net_flux_transmission']) for r in bend],'s-',label='net-flux loss')
    axs[0,2].set(ylabel='loss [dB]',title='Bent-guide loss trends');axs[0,2].legend()
    for key,label,marker in [('modal_transmission','TE₀ transmission','o'),('delay_fs','arrival delay','s'),
                              ('net_flux_transmission','net-flux transmission','^')]:
        errors=[abs(r[key]/rb[key]-1)*100 for r in bend[1:]]
        axs[1,0].loglog(hb[1:],np.maximum(errors,1e-6),marker+'-',label=label)
    axs[1,0].set(ylabel='difference from finest run [%]',title=f'Bend reference: {rb["mesh"]["h"]*1000:g} nm');axs[1,0].legend()
    axs[1,1].loglog(hb,[r['gpu_seconds'] for r in bend],'o-',label='GPU trajectory')
    axs[1,1].set(ylabel='seconds on NVIDIA L40',title='Trajectory cost (setup/output excluded)');axs[1,1].legend()
    axs[1,2].semilogx(hb,[r['delay_fs'] for r in bend],'o-',label='measured TE₀ centroid delay')
    axs[1,2].axhline(rb['geometric_delay_fs'],color='white',ls=':',label='geometric ng L/c estimate')
    axs[1,2].set(ylabel='input-to-output delay [fs]',title='Bend arrival time');axs[1,2].legend(fontsize=8)
    for ax in axs.ravel():
        ax.set_xlabel('near-guide spacing [nm]');ax.grid(alpha=.25)
    fig.suptitle('Mesh convergence · P1/P0 Maxwell FEM · coarse 80 nm bend has an unresolved tail',fontsize=14)
    fig.savefig(out/'convergence.png',dpi=160);plt.close(fig)
    if (root/'parameters.json').exists():
        pars=json.loads((root/'parameters.json').read_text())
        fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for ax,prefix,key,label in zip(axs.ravel(),['cfl_','clad_','halo_','absorb_'],
                         ['cfl','clad_ratio','halo','absorber_strength'],
                         ['CFL safety fraction','far/near spacing ratio','fine cladding halo [µm]','absorber strength']):
            rs=[r for r in pars if r['name'].startswith(prefix)]
            xs=[r['mesh'].get(key,r['run'].get(key)) for r in rs]
            ax.plot(xs,[abs(r['modal_transmission']/rb['modal_transmission']-1)*100 for r in rs],'o-',label='TE₀ T error vs reference')
            ax.plot(xs,[abs(r['delay_fs']/rb['delay_fs']-1)*100 for r in rs],'s-',label='arrival error vs reference')
            ax.set(xlabel=label,ylabel='difference [%]');ax.grid(alpha=.25)
            other=ax.twinx();other.plot(xs,[r['gpu_seconds'] for r in rs],'^:',color='#ffbe45');other.set_ylabel('GPU seconds',color='#ffbe45')
            ax.legend(loc='upper left',fontsize=8)
        fig.suptitle('Parameter study at 20 nm · dotted gold curves show runtime',fontsize=14)
        fig.savefig(out/'parameters.png',dpi=160);plt.close(fig)
    if (root/'optimized.json').exists():
        opt=json.loads((root/'optimized.json').read_text())
        fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
        for rs,label,color in [(bend,'baseline: clad ratio 2, unroll 1','#33c5ff'),
                               (opt,'optimized: clad ratio 4, unroll 16','#ffbe45')]:
            gpu=[r['gpu_seconds'] for r in rs]
            axs[0].loglog(gpu,[abs(r['beta_relative_error'])*100 for r in rs],'o-',label=label,color=color)
            compared=[r for r in rs if r is not rb]
            gpu_compared=[r['gpu_seconds'] for r in compared]
            axs[1].loglog(gpu_compared,[abs(r['total_guided_transmission']/rb['total_guided_transmission']-1)*100 for r in compared],'o-',color=color)
            axs[2].loglog(gpu_compared,[abs(r['delay_fs']/rb['delay_fs']-1)*100 for r in compared],'o-',color=color)
            for r in rs:
                if r['mesh']['h']<=0.02:
                    axs[0].annotate(f"{r['mesh']['h']*1000:g} nm",(r['gpu_seconds'],abs(r['beta_relative_error'])*100),
                                    xytext=(4,4),textcoords='offset points',fontsize=8)
        axs[0].set_ylabel('phase error vs analytic slab [%]')
        axs[1].set_ylabel('guided transmission difference [%]')
        axs[2].set_ylabel('arrival delay difference [%]')
        axs[0].legend(fontsize=8)
        for ax in axs:
            ax.set_xlabel('GPU trajectory [s]');ax.grid(alpha=.25)
        fig.suptitle('Accuracy vs execution time · bend differences use the 5 nm reference',fontsize=14)
        fig.savefig(out/'performance.png',dpi=160);plt.close(fig)
    if (root/'graph.json').exists():
        rs=json.loads((root/'graph.json').read_text())
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        labels=['no graph' if not r['run']['graph'] else str(r['run']['graph_unroll']) for r in rs]
        ax.bar(labels,[r['gpu_seconds'] for r in rs],color=['#8293a7']+['#33c5ff']*(len(rs)-1))
        ax.set(xlabel='steps unrolled per device WHILE iteration',ylabel='240 fs trajectory [s]',
               title='Graph batching at 20 nm; same physics kernels')
        fig.savefig(out/'graph_batching.png',dpi=160);plt.close(fig)
    print('Plots written',out,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run');ap.add_argument('--study');ap.add_argument('--out',default='artifacts')
    a=ap.parse_args()
    if a.run: animate(a.run,a.out)
    if a.study: plots(a.study,a.out)
