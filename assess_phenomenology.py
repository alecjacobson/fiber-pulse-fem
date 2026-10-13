#!/usr/bin/env python3
"""Analytic checks and plots for the archived bend-height controls.

Controls can be rerun with warp_waveguide.py --h .01 --clad-ratio 4 --cfl .98
--bend B --out results/phenomenology/bend_B, for B=0,.25,.5,1,1.5.
The additional B=.25 refinement uses --h .005 with the other settings unchanged.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import erfc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fiber_pulse import slab_mode, mode_profile, C0


def main():
    out=Path('artifacts/phenomenology')
    k=2*np.pi/.532;dk=k*1e-4
    modes=[]
    for i in range(3):
        m=slab_mode(k,.5,2.04,1.46,i)
        f=lambda y:float(mode_profile(np.array([y]),m)[0][0]**2)
        core=quad(f,-.25,.25)[0];clad=2*quad(f,.25,10)[0]
        # Hellmann-Feynman identity for this nondispersive TE eigenproblem.
        ng_energy=(2.04**2*core+1.46**2*clad)/(core+clad)/m['neff']
        ng_derivative=(slab_mode(k+dk,.5,2.04,1.46,i)['beta']-slab_mode(k-dk,.5,2.04,1.46,i)['beta'])/(2*dk)
        modes.append(dict(order=i,neff=m['neff'],beta_per_um=m['beta'],
                          group_index_energy=ng_energy,group_index_derivative=ng_derivative,
                          cladding_amplitude_decay_length_um=1/m['gamma']))
    cutoff=2*.5*np.sqrt(2.04**2-1.46**2)/3
    std_f=1/(2*np.pi*np.sqrt(2)*4)
    report=dict(modes=modes,V_half_width=k*.25*np.sqrt(2.04**2-1.46**2),
                min_bend_radius_um=3**2/(6*1.5),
                intensity_fwhm_fs=2*np.sqrt(np.log(2))*4,
                spectral_intensity_fwhm_THz=np.sqrt(np.log(2))/(np.pi*4)*1000,
                fourth_mode_cutoff_wavelength_um=cutoff,
                ideal_gaussian_energy_fraction_above_fourth_mode_cutoff=float(.5*erfc((C0/cutoff*1e-9-C0/.532*1e-9)/(np.sqrt(2)*std_f))),
                note='Spectral tail is for the ideal temporal Gaussian, not a measured fourth-mode power.',
                sources=[
                  'https://ocw.mit.edu/courses/6-974-fundamentals-of-photonics-quantum-electronics-spring-2006/bf3bdeb971819ddc9c68719d28bb2745_classnotes.pdf',
                  'https://math.mit.edu/~stevenj/18.369/spring16/',
                  'https://www.nature.com/articles/ncomms2232',
                  'https://optics.ansys.com/hc/en-us/articles/360042799933-Bent-waveguide-analysis',
                  'https://optics.ansys.com/hc/en-us/articles/360034902213-Mode-source-in-broadband-simulations',
                  'https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/'])
    (out/'analytic_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    rows=json.loads((out/'controls.json').read_text())
    x=np.array([r['mesh']['bend'] for r in rows])
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
    for j,col in enumerate(['#287ab0','#dc8035','#369765']):
        ax.plot(x,[100*r['mode_transmissions'][j]/sum(r['input_mode_energy_ratios']) for r in rows],'.-',color=col,label=f'TE{j}')
    ax.plot(x,[100*r['total_guided_transmission'] for r in rows],'k.-',label='All guided modes')
    f=json.loads((out/'refinement.json').read_text())
    ax.plot([.25],[100*f['modal_transmission']/sum(f['input_mode_energy_ratios'])],marker='s',ms=8,mfc='none',mec='#287ab0',ls='none',label='TE0, 5 nm refinement')
    ax.set(xlabel='Bézier bump height (µm)',ylabel='Forward energy / incident guided energy (%)',
           title='Mode conversion and radiation are different effects\n10 nm mesh; fixed 6 µm bend span',ylim=(-2,104))
    ax.grid(alpha=.2);ax.legend(ncol=2);fig.savefig(out/'bend_controls.png',dpi=170)


if __name__=='__main__': main()
