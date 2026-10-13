#!/usr/bin/env python3
"""Serial, source-keyed method/solver benchmark with independent process isolation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

FILES=['warp_waveguide.py','mesh_waveguide.py','analyze_waveguide.py','integrators.py','device_cg.py','cudss_backend.py','benchmark_integrators.py']
def hashes():return {p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest() for p in FILES}
def cases(study):
    hs=[.04,.02] if study=='screen' else [.01,.005]
    for h in hs:
        for cfl in ([.3,.85,.98] if study=='screen' else [.3,.85,.98]):
            yield dict(h=h,method='leapfrog',cfl=cfl)
        for frac in [.6,1.25]:yield dict(h=h,method='rk4',dt_fraction=frac)
        for dt in ([.01,.02,.05,.1] if study=='screen' else [.02,.05]):
            for backend in ['cg','cudss']:
                yield dict(h=h,method='midpoint',dt_fs=dt,backend=backend)
        if study=='screen':
            yield dict(h=h,method='cn',dt_fs=.05,backend='cg')
            for theta in [.55,1.]:yield dict(h=h,method='theta',theta=theta,dt_fs=.05,backend='cg')
        else:
            for tol,maxiter in [(1e-4,100),(1e-7,100),(1e-6,4)]:
                yield dict(h=h,method='midpoint',dt_fs=.02,backend='cg',tolerance=tol,maxiter=maxiter)

def name(q):return '_'.join(f'{k}-{v}' for k,v in q.items())
def run_case(q,out,repeats):
    import warp as wp
    from mesh_waveguide import MeshConfig,build,operators
    from warp_waveguide import Simulator,RunConfig,TIME_FS
    from integrators import TimeIntegrator,IntegratorConfig
    from analyze_waveguide import metrics
    from cuda.bindings import runtime as cu
    wp.init();wp.synchronize()
    status,free0,total=cu.cudaMemGetInfo()
    mesh=build(MeshConfig(h=q['h'],clad_ratio=4))
    run=RunConfig(cfl=q.get('cfl',.85),duration_fs=240,frames=q.get('frames',False))
    if q['method']=='leapfrog':s=Simulator(mesh,run)
    else:
        dt=q.get('dt_fs',q.get('dt_fraction',.85)*operators(mesh)[-1]*TIME_FS)
        cfg=IntegratorConfig(**{k:v for k,v in q.items() if k in asdict(IntegratorConfig()) and k!='dt_fs'},dt_fs=dt)
        s=TimeIntegrator(mesh,run,cfg)
    s.run();times=[s.gpu_seconds];walls=[s.wall_seconds]
    for _ in range(repeats-1):
        s.reset();start,end=wp.Event(enable_timing=True),wp.Event(enable_timing=True)
        wp.record_event(start);t=time.perf_counter();wp.capture_launch(s.graph);wp.record_event(end);wp.synchronize()
        times.append(wp.get_event_elapsed_time(start,end)/1000);walls.append(time.perf_counter()-t)
    status,free1,total=cu.cudaMemGetInfo()
    trace=s.trace.numpy();ts=(np.arange(len(trace))*s.sample_stride+.5)*s.dt*TIME_FS
    if not np.isfinite(trace).all():raise RuntimeError('Nonfinite trace')
    r=dict(case=q,source_sha256=hashes(),mesh=mesh['config'],run=asdict(run),vertices=len(s.e),triangles=len(s.hx),
        dt_fs=s.dt*TIME_FS,nsteps=s.nsteps,setup_seconds=s.setup_seconds,mesh_seconds=mesh['mesh_seconds'],
        gpu_seconds=float(np.median(times)),gpu_repeats=times,wall_seconds=float(np.median(walls)),
        capture_seconds=s.capture_seconds,device_memory_delta_bytes=int(free0-free1),device=wp.get_device().name,
        warp_version=wp.__version__,status='ok')
    r.update(metrics(ts,trace,s.ports,s.md,s.ng,mesh['center'],run,s.modes,s.pnorm.numpy()))
    if hasattr(s,'residual_stats'):
        r['max_sampled_relative_residual']=float(s.residual_stats.numpy()[0]);r['residual_samples']=int(s.residual_stats.numpy()[1])
        r['integration']=asdict(s.integration)
        if q.get('backend','cudss')=='cg':
            r['max_cg_iterations'],r['cg_unconverged_steps']=map(int,s.solver.stats.numpy())
        else:
            r.update(factor_seconds=s.solver.factor_seconds,factor_nnz=s.solver.factor_nnz,metadata_nodes_replaced=s.solver.patched_nodes)
    out.mkdir(parents=True,exist_ok=True)
    wp.capture_debug_dot_print(s.graph,str(out/'trajectory.dot'))
    dot=(out/'trajectory.dot').read_text()
    r['graph_has_host_transfer']=any(x in dot for x in ['HtoD','DtoH','HOST PAGEABLE','HOST PINNED'])
    if r['graph_has_host_transfer']:r['status']='host_transfer_in_graph'
    if r.get('cg_unconverged_steps',0):r['status']='solver_nonconvergence'
    if r.get('max_sampled_relative_residual',0)>2e-5:r['status']='large_solve_residual'
    (out/'metrics.json').write_text(json.dumps(r,indent=2)+'\n')
    np.savez_compressed(out/'traces.npz',time_fs=ts,trace=trace,ports=s.ports,mode_norms=s.pnorm.numpy())
    if run.frames:s.save(out/'animation')
    if hasattr(s,'close'):s.close()
    print(json.dumps(r),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--study',choices=['screen','fine'],default='screen')
    ap.add_argument('--out',type=Path,default=Path('results/integrators'));ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--case',help='JSON single case');ap.add_argument('--filter',default='')
    a=ap.parse_args()
    if a.case:return run_case(json.loads(a.case),a.out,a.repeats)
    a.out.mkdir(parents=True,exist_ok=True);results=[]
    for q in cases(a.study):
        n=name(q)
        if a.filter and a.filter not in n:continue
        p=a.out/n;p.mkdir(parents=True,exist_ok=True);cached=p/'metrics.json'
        r=json.loads(cached.read_text()) if cached.exists() else {}
        if r.get('source_sha256')!=hashes() or r.get('case')!=q or len(r.get('gpu_repeats',[]))!=a.repeats:
            print('RUN',n,flush=True)
            with (p/'run.log').open('w') as log:
                result=subprocess.run([sys.executable,__file__,'--case',json.dumps(q),'--out',str(p),'--repeats',str(a.repeats)],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                r=dict(case=q,status='failed',returncode=result.returncode,source_sha256=hashes())
                cached.write_text(json.dumps(r,indent=2))
            else:r=json.loads(cached.read_text())
        results.append(dict(name=n,**r));(a.out/f'{a.study}.json').write_text(json.dumps(results,indent=2)+'\n')
        print(n,r['status'],'GPU',r.get('gpu_seconds'),'beta',r.get('beta_relative_error'),'T',r.get('total_guided_transmission'),flush=True)
if __name__=='__main__':main()
