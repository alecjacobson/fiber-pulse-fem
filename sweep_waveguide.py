#!/usr/bin/env python3
"""Reproducible serial GPU resolution and parameter studies (no timing contention)."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import warp as wp
from mesh_waveguide import MeshConfig, build
from warp_waveguide import RunConfig, Simulator


def cases(study):
    if study=='resolution':
        for h in [0.08,0.06,0.04,0.03,0.02,0.015,0.01]:
            for bend in [0.,1.5]:
                yield f"{'straight' if bend==0 else 'bend'}_h{round(h*1000):03d}",MeshConfig(h=h,bend=bend),RunConfig(duration_fs=240)
    elif study=='parameters':
        for cfl in [0.3,0.6,0.85,0.98]:
            yield f'cfl_{cfl}',MeshConfig(h=0.02),RunConfig(cfl=cfl,duration_fs=240)
        for ratio in [1.,2.,3.,4.]:
            yield f'clad_{ratio}',MeshConfig(h=0.02,clad_ratio=ratio),RunConfig(duration_fs=240)
        for halo in [0.,0.15,0.35,0.6]:
            yield f'halo_{halo}',MeshConfig(h=0.02,halo=halo),RunConfig(duration_fs=240)
        for strength in [5.,10.,20.]:
            yield f'absorb_{strength}',MeshConfig(h=0.02),RunConfig(duration_fs=240,absorber_strength=strength)
        for sigma in [4.,8.]:
            yield f'sigma_{sigma}',MeshConfig(h=0.02),RunConfig(duration_fs=280,sigma_fs=sigma)
    elif study=='reference':
        for bend in [0.,1.5]:
            yield f"{'straight' if bend==0 else 'bend'}_h007",MeshConfig(h=0.007,bend=bend),RunConfig(duration_fs=240)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study',choices=['resolution','parameters','reference'],default='resolution')
    ap.add_argument('--out',default='results/study')
    ap.add_argument('--rerun',action='store_true')
    a = ap.parse_args()
    wp.init()
    root = Path(a.out)
    root.mkdir(parents=True,exist_ok=True)
    results = []
    for name,mc,rc in cases(a.study):
        path = root/name
        if (path/'metrics.json').exists() and not a.rerun:
            result = json.loads((path/'metrics.json').read_text())
        else:
            print('RUN',name,flush=True)
            mesh = build(mc)
            sim = Simulator(mesh,rc)
            sim.run()
            result = sim.save(path)
            del sim,mesh
            gc.collect()
        results.append(dict(name=name,**result))
        print(f"{name}: beta error {result['beta_relative_error']:.3%}, T0 {result['modal_transmission']:.5f}, "
              f"Tflux {result['net_flux_transmission']:.5f}, delay {result['delay_fs']:.3f} fs, GPU {result['gpu_seconds']:.3f}s",flush=True)
        (root/f'{a.study}.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    main()
