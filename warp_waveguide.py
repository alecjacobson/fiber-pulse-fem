#!/usr/bin/env python3
"""Device-resident P1/P0 Maxwell FEM, assembled by warp.fem.

Internal units: x / 1um, tau = c0*t / 1um, e = E/E0, h = Z0*H/E0.
All stepping, source evaluation, diagnostics and frame sampling run in Warp.
Only setup and final readback use NumPy/SciPy. The CUDA conditional graph
contains the ENTIRE trajectory, including a device-side step counter.
"""
import argparse
from dataclasses import dataclass, asdict
import json
import hashlib
from pathlib import Path
import time
import numpy as np
import warp as wp
import warp.fem as fem
from warp.sparse import bsr_transposed
from fiber_pulse import C0, Z0, slab_mode, mode_profile
from mesh_waveguide import MeshConfig, build, operators

wp.set_module_options({'enable_backward': False})
TIME_FS = 1e9/C0
E0 = 1e6
SOURCE_HASHES = {p: hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
                 for p in ['warp_waveguide.py','mesh_waveguide.py','analyze_waveguide.py']}


@fem.integrand
def mass_form(s: fem.Sample, v: fem.Field, eps: wp.array(dtype=float)):
    return eps[s.element_index]*v(s)


@fem.integrand
def volume_form(s: fem.Sample, v: fem.Field):
    return v(s)


@fem.integrand
def derivative_form(s: fem.Sample, u: fem.Field, v: fem.Field, axis: int):
    return fem.grad(u,s)[axis]*v(s)


@wp.func
def pulse(t: float, x: float, omega: float, beta: float, vg: float, sigma: float):
    tau = t-4.5*sigma-x/vg
    return wp.exp(-0.5*tau*tau/(sigma*sigma))*wp.cos(omega*(t-4.5*sigma)-beta*x)


@wp.kernel
def advance_h(e: wp.array(dtype=float), hx: wp.array(dtype=float), hy: wp.array(dtype=float),
              offsets: wp.array(dtype=int), cols: wp.array(dtype=int),
              yoffsets: wp.array(dtype=int), ycols: wp.array(dtype=int),
              bx: wp.array(dtype=float), by: wp.array(dtype=float),
              area: wp.array(dtype=float), alpha: wp.array(dtype=float), src: wp.array(dtype=wp.vec2),
              step: wp.array(dtype=int), dt: float, omega: float, beta: float, vg: float, sigma: float):
    i = wp.tid()
    gx = float(0.0)
    gy = float(0.0)
    for j in range(offsets[i], offsets[i+1]):
        gx += bx[j]*e[cols[j]]
    for j in range(yoffsets[i], yoffsets[i+1]):
        gy += by[j]*e[ycols[j]]
    a = 0.5*alpha[i]*dt
    d = dt/(1.0+a)
    source = src[i][0]*pulse(float(step[0])*dt,src[i][1],omega,beta,vg,sigma)
    hx[i] = (1.0-a)/(1.0+a)*hx[i]-d*gy/area[i]
    hy[i] = (1.0-a)/(1.0+a)*hy[i]+d*(gx/area[i]-source)


@wp.kernel
def advance_e(e: wp.array(dtype=float), old: wp.array(dtype=float),
              hx: wp.array(dtype=float), hy: wp.array(dtype=float),
              offsets: wp.array(dtype=int), cols: wp.array(dtype=int),
              yoffsets: wp.array(dtype=int), ycols: wp.array(dtype=int),
              bx: wp.array(dtype=float), by: wp.array(dtype=float),
              mass: wp.array(dtype=float), alpha: wp.array(dtype=float), src: wp.array(dtype=wp.vec2),
              step: wp.array(dtype=int), dt: float, omega: float, beta: float, vg: float, sigma: float):
    i = wp.tid()
    r = float(0.0)
    for j in range(offsets[i],offsets[i+1]):
        r -= bx[j]*hy[cols[j]]
    for j in range(yoffsets[i],yoffsets[i+1]):
        r += by[j]*hx[ycols[j]]
    a = 0.5*alpha[i]*dt
    source = src[i][0]*(beta/omega)*pulse((float(step[0])+0.5)*dt,src[i][1],omega,beta,vg,sigma)
    old[i] = e[i]
    e[i] = (1.0-a)/(1.0+a)*e[i]+dt/(mass[i]*(1.0+a))*(r+source)


@wp.kernel
def sample_ports(e: wp.array(dtype=float), old: wp.array(dtype=float), hy: wp.array(dtype=float),
                 f: wp.array(dtype=wp.vec3i), ids: wp.array(dtype=int), weights: wp.array(dtype=wp.vec3),
                 qw: wp.array(dtype=wp.vec3), offsets: wp.array(dtype=int),
                 norm: wp.array(dtype=float), step: wp.array(dtype=int), stride: int,
                 trace: wp.array3d(dtype=float)):
    port = wp.tid()
    if step[0] % stride == 0:
        se = wp.float64(0.0)
        sh = wp.float64(0.0)
        sf = wp.float64(0.0)
        sw = wp.float64(0.0)
        for j in range(offsets[port],offsets[port+1]):
            fi = ids[j]
            tri = f[fi]
            w = weights[j]
            z = 0.5*(w[0]*(e[tri[0]]+old[tri[0]])+w[1]*(e[tri[1]]+old[tri[1]])+w[2]*(e[tri[2]]+old[tri[2]]))
            h = hy[fi]
            se += wp.float64(qw[j][0]*z)
            sh += wp.float64(qw[j][0]*h)
            sf -= wp.float64(qw[j][1]*z*h)
            sw -= wp.float64(qw[j][2]*z*h)
        k = step[0]//stride
        trace[k,port,0] = float(se)/norm[port]
        trace[k,port,1] = float(sh)/norm[port]
        trace[k,port,2] = float(sf)
        trace[k,port,3] = float(sw)


@wp.kernel
def sample_frames(e: wp.array(dtype=float), old: wp.array(dtype=float),
                  hx: wp.array(dtype=float), hy: wp.array(dtype=float), f: wp.array(dtype=wp.vec3i),
                  ids: wp.array(dtype=int), weights: wp.array(dtype=wp.vec3),
                  step: wp.array(dtype=int), stride: int, frames: wp.array3d(dtype=float)):
    i = wp.tid()
    if step[0] % stride == 0:
        fi = ids[i]
        tri = f[fi]
        w = weights[i]
        z = 0.5*(w[0]*(e[tri[0]]+old[tri[0]])+w[1]*(e[tri[1]]+old[tri[1]])+w[2]*(e[tri[2]]+old[tri[2]]))
        h = wp.sqrt(hx[fi]*hx[fi]+hy[fi]*hy[fi])
        k = step[0]//stride
        frames[k,i,0] = z
        frames[k,i,1] = h
        frames[k,i,2] = wp.abs(z)*h


@wp.kernel
def increment(step: wp.array(dtype=int), condition: wp.array(dtype=int), nsteps: int):
    step[0] += 1
    condition[0] = int(step[0]<nsteps)


@dataclass
class RunConfig:
    cfl: float = 0.85
    duration_fs: float = 240.0
    sigma_fs: float = 4.0
    wavelength_um: float = 0.532
    source_x: float = 2.5
    source_width: float = 0.08
    absorber_um: float = 1.5
    absorber_strength: float = 10.0
    sample_fs: float = 0.12
    frame_fs: float = 1.2
    frames: bool = False
    graph: bool = True
    graph_unroll: int = 1
    device: str = 'cuda:0'


def mode_data(c, run):
    k = 2*np.pi/run.wavelength_um
    md = slab_mode(k,c.width,2.04,1.46,0)
    dk = k*1e-4
    ng = (slab_mode(k+dk,c.width,2.04,1.46,0)['beta']-slab_mode(k-dk,c.width,2.04,1.46,0)['beta'])/(2*dk)
    return md,ng


def absorber(p, c, r):
    w = r.absorber_um
    depth = np.maximum(np.column_stack((w-p[:,0],p[:,0]-(c.length-w),w-p[:,1],p[:,1]-(c.height-w))),0)
    return r.absorber_strength/w*np.sum((depth/w)**3,axis=1)


def barycentric(v, f, ids, p):
    tri = v[f[ids]]
    b = tri[:,1]-tri[:,0]
    c = tri[:,2]-tri[:,0]
    d = p-tri[:,0]
    det = np.cross(b,c)
    w1 = np.cross(d,c)/det
    w2 = np.cross(b,d)/det
    return np.column_stack((1-w1-w2,w1,w2))


def port_quadrature(mesh, ports, md):
    """Three-point Gauss rule on exact triangle/vertical-line intersections."""
    v,f = mesh['v'],mesh['f']
    tri = v[f]
    ids,weights,qw,norm,offsets = [],[],[],[],[0]
    gauss,gw = np.polynomial.legendre.leggauss(3)
    modes = md if isinstance(md,list) else [md]*len(ports)
    for x,mode in zip(ports,modes):
        sel = np.flatnonzero((tri[:,:,0].min(axis=1)<x)&(tri[:,:,0].max(axis=1)>x))
        t = tri[sel]
        lo = np.full(len(sel),np.inf)
        hi = np.full(len(sel),-np.inf)
        for a,b in [(0,1),(1,2),(2,0)]:
            dx = t[:,b,0]-t[:,a,0]
            s = np.divide(x-t[:,a,0],dx,out=np.full_like(dx,np.nan),where=dx!=0)
            y = t[:,a,1]+s*(t[:,b,1]-t[:,a,1])
            ok = (s>=0)&(s<=1)
            lo = np.minimum(lo,np.where(ok,y,np.inf))
            hi = np.maximum(hi,np.where(ok,y,-np.inf))
        ys = ((lo+hi)[:,None]/2+(hi-lo)[:,None]/2*gauss).ravel()
        linew = ((hi-lo)[:,None]/2*gw).ravel()
        ti = np.repeat(sel,3)
        q = ys-mesh['config']['y0']
        psi,_ = mode_profile(q,mode)
        ids.extend(ti)
        weights.extend(barycentric(v,f,ti,np.column_stack((np.full(len(ys),x),ys))))
        qw.extend(np.column_stack((linew*psi,linew,linew*(np.abs(q)<0.9))))
        norm.append(np.sum(linew*psi**2))
        offsets.append(len(ids))
    return ids,weights,qw,offsets,norm


class Simulator:
    def __init__(self, mesh, config=RunConfig()):
        self.setup_start = time.perf_counter()
        self.mesh, self.config = mesh,config
        c = MeshConfig(**mesh['config'])
        r = config
        if not (0 < r.cfl <= 1):
            raise ValueError('CFL fraction must be in (0,1]')
        if r.graph_unroll < 1:
            raise ValueError('graph_unroll must be a positive integer')
        if r.graph and (not wp.get_device(r.device).is_cuda or not wp.is_conditional_graph_supported()):
            raise RuntimeError('Full device loop requires CUDA conditional graphs (CUDA 12.4+); use --no-graph only for validation')
        self.md,self.ng = mode_data(c,r)
        self.omega = 2*np.pi/r.wavelength_um
        area,grad,mass,vol,dtlimit = operators(mesh)
        self.dt = r.cfl*dtlimit
        self.dtlimit = dtlimit
        self.nsteps = int(np.ceil(r.duration_fs/(self.dt*TIME_FS)))
        self.nsteps = ((self.nsteps+r.graph_unroll-1)//r.graph_unroll)*r.graph_unroll
        self.sample_stride = max(1,round(r.sample_fs/(self.dt*TIME_FS)))
        self.frame_stride = max(1,round(r.frame_fs/(self.dt*TIME_FS)))
        self.ports = np.r_[np.arange(4.0,5.61,0.2)+1e-6,14.5+1e-6]
        self.modes = [slab_mode(self.omega,c.width,2.04,1.46,i)
                      for i in range(int(np.ceil(2*self.md['V']/np.pi)))]
        self.device = r.device
        with wp.ScopedDevice(r.device):
            arr = lambda a,dtype=float: wp.array(np.asarray(a),dtype=dtype)
            self.f = arr(mesh['f'],wp.vec3i)
            self.geo = fem.Trimesh2D(tri_vertex_indices=arr(mesh['f'],int),positions=arr(mesh['v'],wp.vec2))
            self.space_e = fem.make_polynomial_space(self.geo,degree=1)
            self.space_h = fem.make_polynomial_space(self.geo,degree=0)
            domain = fem.Cells(self.geo)
            et = fem.make_test(self.space_e,domain=domain)
            ht = fem.make_test(self.space_h,domain=domain)
            eu = fem.make_trial(self.space_e,domain=domain)
            eps = arr(np.where(mesh['core'],2.04**2,1.46**2))
            self.mass = fem.integrate(mass_form,fields={'v':et},values={'eps':eps},output_dtype=float)
            self.area = fem.integrate(volume_form,fields={'v':ht},output_dtype=float)
            self.bx = fem.integrate(derivative_form,fields={'u':eu,'v':ht},values={'axis':0},output_dtype=float)
            self.by = fem.integrate(derivative_form,fields={'u':eu,'v':ht},values={'axis':1},output_dtype=float)
            self.tx,self.ty = bsr_transposed(self.bx),bsr_transposed(self.by)
            self.e = wp.zeros(len(mesh['v']))
            self.old = wp.zeros_like(self.e)
            self.hx = wp.zeros(len(mesh['f']))
            self.hy = wp.zeros_like(self.hx)
            self.step = wp.zeros(1,dtype=int)
            self.condition = wp.ones(1,dtype=int)
            self.ae = arr(absorber(mesh['v'],c,r))
            centers = mesh['v'][mesh['f']].mean(axis=1)
            self.ah = arr(absorber(centers,c,r))
            def source(p):
                psi,_ = mode_profile(p[:,1]-c.y0,self.md)
                x = p[:,0]-r.source_x
                g = np.exp(-0.5*(x/r.source_width)**2)/(r.source_width*np.sqrt(2*np.pi))
                return np.column_stack((psi*g,x))
            se = source(mesh['v'])
            se[:,0] *= vol
            self.se,self.sh = arr(se,wp.vec2),arr(source(centers),wp.vec2)
            extra_ports = list(self.ports[[3,-1]])*(len(self.modes)-1)
            port_modes = [self.md]*len(self.ports)+[md for md in self.modes[1:] for _ in range(2)]
            ids,weights,qw,offsets,norm = port_quadrature(mesh,list(self.ports)+extra_ports,port_modes)
            self.pids,self.pweights,self.pqw = arr(ids,int),arr(weights,wp.vec3),arr(qw,wp.vec3)
            self.poffsets,self.pnorm = arr(offsets,int),arr(norm)
            self.trace = wp.zeros(((self.nsteps-1)//self.sample_stride+1,len(norm),4))
            self.frames = None
            if r.frames:
                import matplotlib.tri as mtri
                self.extent = [0.05,c.length-0.05,c.y0-1.15,c.y0+c.bend+1.15]
                x = np.linspace(*self.extent[:2],640)
                y = np.linspace(*self.extent[2:],144)
                xx,yy = np.meshgrid(x,y)
                p = np.column_stack((xx.ravel(),yy.ravel()))
                ids = mtri.Triangulation(*mesh['v'].T,mesh['f']).get_trifinder()(*p.T)
                if np.any(ids<0):
                    raise RuntimeError('Frame samples outside mesh')
                self.rids,self.rweights = arr(ids,int),arr(barycentric(mesh['v'],mesh['f'],ids,p),wp.vec3)
                self.frame_shape = (len(y),len(x))
                self.frames = wp.zeros(((self.nsteps-1)//self.frame_stride+1,len(ids),3))
        self.graph = None

    def one_step(self):
        r = self.config
        common = [self.step,self.dt,self.omega,self.md['beta'],1/self.ng,r.sigma_fs/TIME_FS]
        wp.launch(advance_h,len(self.mesh['f']),inputs=[self.e,self.hx,self.hy,self.bx.offsets,self.bx.columns,self.by.offsets,self.by.columns,
                  self.bx.values,self.by.values,self.area,self.ah,self.sh,*common],device=self.device)
        wp.launch(advance_e,len(self.mesh['v']),inputs=[self.e,self.old,self.hx,self.hy,self.tx.offsets,self.tx.columns,self.ty.offsets,self.ty.columns,
                  self.tx.values,self.ty.values,self.mass,self.ae,self.se,*common],device=self.device)
        wp.launch(sample_ports,len(self.pnorm),inputs=[self.e,self.old,self.hy,self.f,self.pids,self.pweights,
                  self.pqw,self.poffsets,self.pnorm,self.step,self.sample_stride,self.trace],device=self.device)
        if self.frames is not None:
            wp.launch(sample_frames,len(self.rids),inputs=[self.e,self.old,self.hx,self.hy,self.f,self.rids,
                      self.rweights,self.step,self.frame_stride,self.frames],device=self.device)
        wp.launch(increment,1,inputs=[self.step,self.condition,self.nsteps],device=self.device)

    def reset(self):
        for a in [self.e,self.old,self.hx,self.hy,self.step,self.trace]:
            a.zero_()
        self.condition.fill_(1)
        if self.frames is not None:
            self.frames.zero_()

    def graph_body(self):
        # Static unrolling reduces conditional-node overhead while the outer
        # loop and its termination remain entirely device controlled.
        for _ in range(self.config.graph_unroll):
            self.one_step()

    def run(self):
        with wp.ScopedDevice(self.device):
            # Compile/allocate before capture. Reset all warm-up side effects.
            self.one_step()
            self.reset()
            wp.synchronize_device(self.device)
            t = time.perf_counter()
            if self.config.graph:
                with wp.ScopedCapture(device=self.device) as capture:
                    wp.capture_while(self.condition,self.graph_body)
                self.graph = capture.graph
            self.capture_seconds = time.perf_counter()-t
            self.setup_seconds = time.perf_counter()-self.setup_start
            # Exactly one graph launch and one completion sync for the trajectory.
            start = wp.Event(enable_timing=True)
            end = wp.Event(enable_timing=True)
            wp.record_event(start)
            t = time.perf_counter()
            if self.config.graph:
                wp.capture_launch(self.graph)
            else:
                for _ in range(self.nsteps):
                    self.one_step()
            wp.record_event(end)
            wp.synchronize_device(self.device)
            self.wall_seconds = time.perf_counter()-t
            self.gpu_seconds = wp.get_event_elapsed_time(start,end)/1000
        return self.trace.numpy()

    def save(self,out):
        from analyze_waveguide import metrics
        out = Path(out)
        out.mkdir(parents=True,exist_ok=True)
        trace = self.trace.numpy()
        if not np.isfinite(trace).all() or not np.isfinite(self.e.numpy()).all():
            raise RuntimeError('Non-finite fields or diagnostics; refusing to report physical metrics')
        ts = (np.arange(len(trace))*self.sample_stride+0.5)*self.dt*TIME_FS
        result = dict(mesh=self.mesh['config'],run=asdict(self.config),vertices=len(self.mesh['v']),
                      triangles=len(self.mesh['f']),dt_fs=self.dt*TIME_FS,nsteps=self.nsteps,
                      conservative_dt_limit_fs=self.dtlimit*TIME_FS,
                      mesh_seconds=self.mesh['mesh_seconds'],setup_seconds=self.setup_seconds,
                      capture_seconds=self.capture_seconds,wall_seconds=self.wall_seconds,gpu_seconds=self.gpu_seconds,
                      device=str(wp.get_device(self.device)),warp_version=wp.__version__,
                      device_name=wp.get_device(self.device).name,
                      source_sha256=SOURCE_HASHES,
                      max_final_e=float(np.max(np.abs(self.e.numpy()))),
                      beta_exact_per_um=self.md['beta'],group_index=self.ng)
        result.update(metrics(ts,trace,self.ports,self.md,self.ng,self.mesh['center'],self.config,
                              self.modes,self.pnorm.numpy()))
        (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        if self.graph is not None:
            wp.capture_debug_dot_print(self.graph,str(out/'trajectory.dot'))
        np.savez_compressed(out/'traces.npz',time_fs=ts,trace=trace[:,:len(self.ports)],ports=self.ports,
                            higher_modes=trace[:,len(self.ports):],mode_norms=self.pnorm.numpy())
        np.savez_compressed(out/'mesh.npz',v=self.mesh['v'],f=self.mesh['f'],core=self.mesh['core'],
                            outline=self.mesh['outline'],center=self.mesh['center'],
                            config_json=json.dumps(self.mesh['config']))
        if self.frames is not None:
            # Float16 is only an archive/display format; all simulation and
            # quantitative monitor values retain their original precision.
            np.savez_compressed(out/'frames.npz',frames=self.frames.numpy().reshape((-1,*self.frame_shape,3)).astype(np.float16),
                                time_fs=(np.arange(self.frames.shape[0])*self.frame_stride+0.5)*self.dt*TIME_FS,
                                extent=self.extent)
        print(json.dumps(result,indent=2),flush=True)
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--h',type=float,default=0.04,help='near-guide mesh spacing in um')
    ap.add_argument('--clad-ratio',type=float,default=2)
    ap.add_argument('--halo',type=float,default=0.35)
    ap.add_argument('--bend',type=float,default=1.5)
    ap.add_argument('--cfl',type=float,default=0.85)
    ap.add_argument('--duration',type=float,default=240)
    ap.add_argument('--sigma',type=float,default=4)
    ap.add_argument('--absorber-strength',type=float,default=10)
    ap.add_argument('--frames',action='store_true')
    ap.add_argument('--no-graph',action='store_true')
    ap.add_argument('--graph-unroll',type=int,default=1)
    ap.add_argument('--mesh-from',type=Path,help='reuse a saved mesh.npz; overrides mesh flags')
    ap.add_argument('--out',default='results/run')
    args = ap.parse_args()
    wp.init()
    if args.mesh_from:
        with np.load(args.mesh_from,allow_pickle=False) as data:
            mesh = {k:data[k] for k in ['v','f','core','outline','center']}
            mesh.update(config=json.loads(str(data['config_json'])),mesh_seconds=0.0)
    else:
        mesh = build(MeshConfig(h=args.h,clad_ratio=args.clad_ratio,halo=args.halo,bend=args.bend))
    print(f"Mesh: {len(mesh['v'])} vertices, {len(mesh['f'])} triangles in {mesh['mesh_seconds']:.2f}s",flush=True)
    sim = Simulator(mesh,RunConfig(cfl=args.cfl,duration_fs=args.duration,sigma_fs=args.sigma,
                                 frames=args.frames,graph=not args.no_graph,graph_unroll=args.graph_unroll,
                                 absorber_strength=args.absorber_strength))
    sim.run()
    sim.save(args.out)


if __name__=='__main__':
    main()
