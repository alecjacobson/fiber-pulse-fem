#!/usr/bin/env python3
"""Export compact benchmark evidence and update the measured-results README section."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np


def report(root,out):
    root,out=Path(root),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    data={}
    for study in ['resolution','parameters','reference','optimized','graph']:
        data[study]=json.loads((root/f'{study}.json').read_text())
        shutil.copyfile(root/f'{study}.json',out/f'{study}.json')
    rows=[r for study in data.values() for r in study]
    columns=['name','h_nm','clad_ratio','cfl','graph_unroll','vertices','triangles','dt_fs','nsteps',
             'mesh_seconds','setup_seconds','gpu_seconds','beta_relative_error','modal_transmission',
             'total_guided_transmission','net_flux_transmission','delay_fs','tail_fraction']
    with (out/'measurements.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        for r in rows:
            flat={k:r.get(k,'') for k in columns}
            flat.update(h_nm=r['mesh']['h']*1000,clad_ratio=r['mesh']['clad_ratio'],cfl=r['run']['cfl'],
                        graph_unroll=r['run'].get('graph_unroll',1))
            writer.writerow(flat)
    combined=data['resolution']+data['reference']
    bend=sorted([r for r in combined if r['mesh']['bend']!=0],key=lambda r:-r['mesh']['h'])
    straight={r['mesh']['h']:r for r in combined if r['mesh']['bend']==0}
    ref=bend[-1]
    opt=sorted(data['optimized'],key=lambda r:-r['mesh']['h'])
    lines=['## Measured results (NVIDIA L40, Warp 1.15.0)',
           '',f'The committed study contains {len(rows)} runs. Raw configurations and measurements are in',
           '[artifacts/measurements.csv](artifacts/measurements.csv) and the adjacent JSON files.',
           'Timings are measurements on this host, not portable performance guarantees.','',
           '| h [nm] | phase error [%] | straight arrival error [%] | bend TE₀ T [%] | bend guided T [%] | bend delay [fs] | GPU [s] |',
           '|---:|---:|---:|---:|---:|---:|---:|']
    for r in bend:
        complete=r['tail_fraction']<1e-3
        vals=[f"{100*r['modal_transmission']:.3f}",f"{100*r['total_guided_transmission']:.3f}",f"{r['delay_fs']:.3f}"] if complete else ['—','—','—']
        lines.append(f"| {r['mesh']['h']*1000:g} | {abs(r['beta_relative_error'])*100:.4f} | "
                     f"{abs(straight[r['mesh']['h']]['delay_relative_error'])*100:.4f} | "+' | '.join(vals)+f" | {r['gpu_seconds']:.2f} |")
    lines += ['', 'The 80 nm bend still has a substantial late-time tail at 480 fs; its transmission',
              'and centroid are not accepted quantitative measurements. It is a preview setting.',
              'Finer runs have negligible late-window TE₀ energy. Small straight-guide modal',
              'transmission values above one reflect finite-element port sampling and the fixed',
              'carrier-admittance approximation; net flux is the stronger conservation check.', '',
              '![Resolution and loss trends](artifacts/convergence.png)', '',
              '### Recommended settings', '',
              'The following runs use a nominal 0.35 µm vertical cladding halo, CFL fraction 0.85,',
              'and 16 steps per device graph iteration. The distant-cladding spacing is 4h.', '',
              '| h [nm] | vertices | GPU [s] | mesh + setup + run [s] | phase error [%] | guided T difference vs 5 nm reference [%] | arrival difference [%] |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for r in opt:
        lines.append(f"| {r['mesh']['h']*1000:g} | {r['vertices']:,} | {r['gpu_seconds']:.2f} | "
                     f"{r['mesh_seconds']+r['setup_seconds']+r['wall_seconds']:.2f} | {abs(r['beta_relative_error'])*100:.4f} | "
                     f"{abs(r['total_guided_transmission']/ref['total_guided_transmission']-1)*100:.4f} | "
                     f"{abs(r['delay_fs']/ref['delay_fs']-1)*100:.4f} |")
    fine=opt[-1]
    lines += ['', 'Use **10 nm** for routine iteration and **5 nm** for the highest tested quality',
              'at a practical cost on this GPU. These are empirical choices within the tested',
              'P1/P0 scheme, not a global optimum over all FEM discretizations.', '',
              '```bash',
              'python warp_waveguide.py --h 0.01 --clad-ratio 4 --graph-unroll 16 --out results/balanced',
              'python warp_waveguide.py --h 0.005 --clad-ratio 4 --graph-unroll 16 --frames --out results/quality',
              'python render_warp.py --run results/quality --out artifacts',
              '```', '',
              'Mesh/setup/run totals exclude archive compression and rendering. Most reference and',
              'resolution timing records predate the faster polygon classification;',
              'older setup timers excluded the small host CFL prepass (see the JSON flag). Optimized-run totals',
              'include that prepass. Mesh vertices, connectivity, and material labels were verified',
              'bitwise identical before/after the classification optimization at 20 and 7 nm.', '',
              '![Accuracy and runtime](artifacts/performance.png)', '',
              '### What converged, and what the loss means', '',
              f"The finest baseline has β = {ref['beta_per_um']:.6f} rad/µm in the input lead, compared",
              f"with the analytic {ref['beta_exact_per_um']:.6f} rad/µm. Straight-guide phase error",
              'decreases approximately quadratically with spacing. Straight-guide pulse delay',
              f"at 5 nm differs from analytic n_g Δx/c₀ by {100*abs(straight[0.005]['delay_relative_error']):.3f}%.", '',
              f"For the bend, the 5 nm baseline transmits {100*ref['modal_transmission']:.3f}% in TE₀",
              f"({ref['modal_loss_db']:.3f} dB insertion loss), and {100*ref['total_guided_transmission']:.3f}%",
              f"in all guided modes ({-10*np.log10(ref['total_guided_transmission']):.3f} dB guided-power loss).",
              'TE₀/TE₁/TE₂ forward energy ratios are '+', '.join(f'{v:.6f}' for v in ref['mode_transmissions'])+'.',
              'Individual mode ratios use incident TE₀ energy; summed guided transmission uses',
              'all incident guided energy. The finite-bandwidth source has about 0.033% higher-mode',
              'content at the input, which is included in that normalization.',
              f"Net cross-section flux transmission is {100*ref['net_flux_transmission']:.3f}%, including radiation.",
              f"The TE₀ envelope centroid delay is {ref['delay_fs']:.3f} fs; the geometric n_g L/c₀",
              f"estimate is {ref['geometric_delay_fs']:.3f} fs. The difference from that geometric estimate",
              'is not itself a discretization error: the converged bend filters the pulse spectrum',
              'and changes propagation relative to the straight slab.', '',
              'At 20 nm, reducing CFL from 0.98 to 0.30 changes delay by only about 0.08 fs;',
              'spatial dispersion dominates. A fine evanescent halo matters. At that resolution,',
              'a far/near ratio of 4 preserves guided power reasonably well but changes total',
              'radiation flux noticeably; the recommended fine runs were checked separately.',
              'Changing sponge strength from 5 to 20 leaves guided transmission nearly unchanged',
              'while moving full-plane net flux by about 1.2 percentage points. Therefore the',
              'guided transmission is better established than the radiation/absorption balance.',
              'A true PML, domain-size convergence, and frequency-dependent port admittances would',
              'be useful next checks for more precise radiation or broadband measurements.', '',
              'Changing sigma from 4 to 8 fs is a change of physical input bandwidth, not a',
              'numerical refinement: TE₀ transmission changes while summed guided transmission',
              'is nearly unchanged in the 20 nm study.', '',
              'Graph batching has only a small timing effect here (about 2.02 s unbatched versus',
              '1.99 s with 16 steps). Its main benefit is keeping the complete trajectory independent',
              'of host scheduling. Most of the optimized speedup comes from the coarser distant cladding.', '',
              '![Parameter study](artifacts/parameters.png)',
              '![Graph batching](artifacts/graph_batching.png)', '',
              '### Animation and verification', '',
              '[MP4 animation](artifacts/pulse.mp4) · [snapshot](artifacts/pulse_snapshot.png) ·',
              '[captured CUDA graph](artifacts/trajectory.dot) · [animation-run metrics](artifacts/quality_metrics.json)',
              '', 'The [quality port traces](artifacts/quality_traces.npz) and',
              '[reference port traces](artifacts/reference_traces.npz) retain the quantitative data.', '',
              '![Pulse through the Bézier bend](artifacts/pulse.gif)', '',
              'The animation shows instantaneous E, magnetic magnitude, and Poynting magnitude,',
              'with the input/output TE₀ intensity envelopes below. Raster frames are sampled',
              'on device; video encoding happens after the simulation.', '',
              'Verification includes arbitrary/nonpreset mesh resolutions in both host meshers,',
              'conforming material interfaces, Warp FEM assembly against analytic triangle',
              'operators, full-graph trajectories against independent float64 NumPy/SciPy',
              'stepping, bitwise graph replay, and a lossless discrete-energy invariant.',
              'The comparison errors are about 1e-5; lossless energy drift is about 1e-7.',
              'See [artifacts/verification.txt](artifacts/verification.txt) for the executed checks.', '',
              'Reproduce additional tuning plots with:', '',
              '```bash', 'python sweep_waveguide.py --study graph',
              'python sweep_waveguide.py --study optimized',
              'python render_warp.py --study results/study --out artifacts',
              'python report_waveguide.py', '```', '']
    readme=Path('README.md')
    text=readme.read_text()
    start,end='<!-- warp-results-start -->','<!-- warp-results-end -->'
    block=start+'\n'+'\n'.join(lines)+'\n'+end+'\n\n'
    if start in text:
        a=text.index(start);b=text.index(end)+len(end)
        text=text[:a]+block.rstrip()+text[b:]
    else:
        text=text.replace('## Original implementation\n',block+'## Original implementation\n')
    readme.write_text(text)
    print('Exported',len(rows),'measurements to',out)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study',default='results/study');ap.add_argument('--out',default='artifacts')
    a=ap.parse_args();report(a.study,a.out)
