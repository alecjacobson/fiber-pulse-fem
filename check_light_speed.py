#!/usr/bin/env python3
"""Independent time-of-flight checks using production Warp FEM Maxwell kernels.

A source-free electric Gaussian packet splits into left/right traveling waves.
H starts at zero: no expected speed is imposed in either an injection or a
staggered initialization. The strip is uniform in y, with natural side walls.
Only x ends absorb. Readbacks occur after the complete captured trajectory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.signal import hilbert
import warp as wp
import warp.fem as fem
from warp.sparse import bsr_transposed
from fiber_pulse import C0
from warp_waveguide import (Simulator, RunConfig, TIME_FS, mass_form, volume_form,
                            derivative_form, advance_h, advance_e, increment)


@wp.kernel
def monitor(e: wp.array(dtype=float), old: wp.array(dtype=float),
            ids: wp.array(dtype=int), weights: wp.array(dtype=float),
            step: wp.array(dtype=int), stride: int, trace: wp.array2d(dtype=float)):
    i = wp.tid()
    if step[0] % stride == 0:
        j = ids[i]
        w = weights[i]
        trace[step[0]//stride,i] = 0.5*((1.0-w)*(e[j]+old[j])+w*(e[j+1]+old[j+1]))


class HomogeneousPulse(Simulator):
    """Reuse production integrands, time kernels and full-trajectory graph runner."""
    def __init__(self, h=0.02, index=1.0, cfl=0.85, wavelength_um=0.532):
        if h <= 0 or index <= 0 or wavelength_um <= 0 or not 0 < cfl <= 1:
            raise ValueError("Positive h, index, wavelength and CFL in (0,1] required")
        self.setup_start = time.perf_counter()
        self.config = RunConfig(cfl=cfl, duration_fs=12*index*TIME_FS,
                                wavelength_um=wavelength_um, sample_fs=0.06)
        self.device = self.config.device
        self.index, self.wavelength_um = index,wavelength_um
        nx,ny = round(24/h),max(2,round(0.16/h))
        self.h = 24/nx
        dy = 0.16/ny
        xx,yy = np.meshgrid(np.linspace(0,24,nx+1),np.linspace(0,0.16,ny+1))
        v = np.column_stack((xx.ravel(),yy.ravel()))
        a = (np.arange(ny)[:,None]*(nx+1)+np.arange(nx)).ravel()
        f = np.concatenate((np.column_stack((a,a+1,a+nx+2)),
                            np.column_stack((a,a+nx+2,a+nx+1)))).astype(np.int32)
        self.mesh = {'v':v,'f':f}
        # Gershgorin bound for lumped P1 on this rectangular triangulation.
        # Include the two corners with only one incident triangle (3/2 factor).
        self.dt = cfl*index/np.sqrt(1.5*(1/self.h**2+1/dy**2))
        self.nsteps = int(np.ceil(self.config.duration_fs/(self.dt*TIME_FS)))
        self.sample_stride = max(1,round(0.06/(self.dt*TIME_FS)))
        self.ports = np.arange(12.,16.001,0.08)
        # Uniform initial E, H=0; exact continuum evolution is two equal packets.
        e0 = np.exp(-0.5*((v[:,0]-8)/0.8)**2)*np.cos(2*np.pi*index/wavelength_um*(v[:,0]-8))
        def damping(p):
            depth = np.maximum(np.maximum(1.5-p[:,0],p[:,0]-22.5),0)
            return 10/1.5*(depth/1.5)**3
        with wp.ScopedDevice(self.device):
            arr = lambda x,dtype=float: wp.array(np.asarray(x),dtype=dtype)
            self.geo = fem.Trimesh2D(tri_vertex_indices=arr(f,int),positions=arr(v,wp.vec2))
            se = fem.make_polynomial_space(self.geo,degree=1)
            sh = fem.make_polynomial_space(self.geo,degree=0)
            domain = fem.Cells(self.geo)
            et,ht = fem.make_test(se,domain=domain),fem.make_test(sh,domain=domain)
            eu = fem.make_trial(se,domain=domain)
            eps = wp.full(len(f),index**2,dtype=float)
            self.mass = fem.integrate(mass_form,fields={'v':et},values={'eps':eps},output_dtype=float)
            self.area = fem.integrate(volume_form,fields={'v':ht},output_dtype=float)
            self.bx = fem.integrate(derivative_form,fields={'u':eu,'v':ht},values={'axis':0},output_dtype=float)
            self.by = fem.integrate(derivative_form,fields={'u':eu,'v':ht},values={'axis':1},output_dtype=float)
            self.tx,self.ty = bsr_transposed(self.bx),bsr_transposed(self.by)
            self.initial = arr(e0)
            self.e,self.old = wp.zeros(len(v)),wp.zeros(len(v))
            self.hx,self.hy = wp.zeros(len(f)),wp.zeros(len(f))
            self.ae,self.ah = arr(damping(v)),arr(damping(v[f].mean(axis=1)))
            self.se,self.sh = wp.zeros(len(v),dtype=wp.vec2),wp.zeros(len(f),dtype=wp.vec2)
            ix = np.floor(self.ports/self.h).astype(int)
            self.ids = arr(ix+(ny//2)*(nx+1),int)
            self.weights = arr(self.ports/self.h-ix)
            self.trace = wp.zeros(((self.nsteps-1)//self.sample_stride+1,len(self.ports)))
            self.step,self.condition = wp.zeros(1,dtype=int),wp.ones(1,dtype=int)
            self.frames,self.graph = None,None
            self.reset()

    def reset(self):
        super().reset()
        wp.copy(self.e,self.initial)

    def one_step(self):
        common = [self.step,self.dt,1.,1.,1.,1.]  # sources are identically zero
        wp.launch(advance_h,len(self.mesh['f']),inputs=[self.e,self.hx,self.hy,
            self.bx.offsets,self.bx.columns,self.by.offsets,self.by.columns,
            self.bx.values,self.by.values,self.area,self.ah,self.sh,*common],device=self.device)
        wp.launch(advance_e,len(self.mesh['v']),inputs=[self.e,self.old,self.hx,self.hy,
            self.tx.offsets,self.tx.columns,self.ty.offsets,self.ty.columns,
            self.tx.values,self.ty.values,self.mass,self.ae,self.se,*common],device=self.device)
        wp.launch(monitor,len(self.ports),inputs=[self.e,self.old,self.ids,self.weights,
            self.step,self.sample_stride,self.trace],device=self.device)
        wp.launch(increment,1,inputs=[self.step,self.condition,self.nsteps],device=self.device)

    def measure(self, trace):
        t = (np.arange(len(trace))*self.sample_stride+0.5)*self.dt*TIME_FS
        return measure_speed(t,self.ports,trace,self.wavelength_um),t


def measure_speed(t_fs, x_um, trace, wavelength_um):
    """Fit observed arrival times and spatial phase; no theoretical phase anchor."""
    env2 = np.abs(hilbert(trace,axis=0))**2
    arrival = np.sum(t_fs[:,None]*env2,axis=0)/np.sum(env2,axis=0)
    slope,intercept = np.polyfit(x_um,arrival,1)
    # Fourier transfer at the vacuum carrier frequency, not a prescribed source.
    omega_fs = 2*np.pi*C0/wavelength_um*1e-9
    phasor = np.sum(trace*np.exp(-1j*omega_fs*t_fs[:,None]),axis=0)
    jumps = np.angle(phasor[1:]/phasor[:-1])
    if np.max(np.abs(jumps)) > 0.95*np.pi:
        raise ValueError('Spatial probes too far apart for independent phase unwrapping')
    phase = np.unwrap(np.angle(phasor))
    pslope,pintercept = np.polyfit(x_um,phase,1)
    return dict(group_speed_m_s=float(1e9/slope),phase_speed_m_s=float(omega_fs/(-pslope)*1e9),
                arrival_slope_fs_per_um=float(slope),arrival_intercept_fs=float(intercept),
                arrival_fit_rms_fs=float(np.std(arrival-slope*x_um-intercept)),
                beta_per_um=float(-pslope),phase_fit_rms_rad=float(np.std(phase-pslope*x_um-pintercept)),
                max_phase_jump_rad=float(np.max(np.abs(jumps))),arrival_fs=arrival.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='artifacts/light_speed')
    p.add_argument('--h',type=float,nargs='+',default=[0.04,0.02,0.01,0.005])
    p.add_argument('--index',type=float,nargs='+',default=[1.,1.46,2.04])
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True,exist_ok=True)
    records = []
    for n in a.index:
        for h in a.h:
            sim = HomogeneousPulse(h,n)
            trace = sim.run()
            m,t = sim.measure(trace)
            m.update(h_um=sim.h,index=n,expected_speed_m_s=C0/n,
                     group_relative_error=m['group_speed_m_s']/(C0/n)-1,
                     phase_relative_error=m['phase_speed_m_s']/(C0/n)-1,
                     dt_fs=sim.dt*TIME_FS,nsteps=sim.nsteps,gpu_seconds=sim.gpu_seconds,
                     vertices=len(sim.mesh['v']),triangles=len(sim.mesh['f']),
                     method='leapfrog',cfl=sim.config.cfl,
                     sample_interval_fs=sim.sample_stride*sim.dt*TIME_FS,
                     duration_fs=sim.nsteps*sim.dt*TIME_FS)
            records.append(m)
            name=f'n{n:g}_h{h:g}'
            np.savez_compressed(out/f'{name}.npz',time_fs=t,x_um=sim.ports,e=trace)
            wp.capture_debug_dot_print(sim.graph,str(out/f'{name}.dot'))
            print(name, {k:m[k] for k in ['group_relative_error','phase_relative_error','arrival_fit_rms_fs']},flush=True)
    report = dict(c0_m_s=C0,time_unit_fs=TIME_FS,warp_version=wp.__version__,
                  device=wp.get_device().name,
                  geometry=dict(length_um=24,height_um=.16,initial_center_um=8,
                                initial_envelope_sigma_um=.8,initial_magnetic_field=0,
                                probe_start_um=12,probe_end_um=16,probe_spacing_um=.08,
                                absorber_width_um=1.5,absorber_strength=10),
                  source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                                 ['check_light_speed.py','warp_waveguide.py','fiber_pulse.py']},
                  homogeneous=records)
    (out/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
