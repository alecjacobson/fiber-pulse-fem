"""Graph-capturable time integrators for the identical Warp FEM Maxwell system.

Host sparse products prepare a constant SPD electric Schur complement. Every
source, RHS, solve, field update and diagnostic during propagation is on device.
"""
from dataclasses import dataclass, asdict
import numpy as np
import warp as wp
from scipy import sparse
from warp.sparse import bsr_from_triplets, bsr_mv
from device_cg import DeviceCG, TiledDot
from warp_waveguide import Simulator, RunConfig, TIME_FS, pulse, sample_ports, sample_frames, increment

@wp.func
def forcing(t: float, x: float, dt: float, theta: float, midpoint: int,
            omega: float, beta: float, vg: float, sigma: float):
    if midpoint != 0:
        return pulse(t+0.5*dt,x,omega,beta,vg,sigma)
    return (1.0-theta)*pulse(t,x,omega,beta,vg,sigma)+theta*pulse(t+dt,x,omega,beta,vg,sigma)

@wp.kernel
def magnetic(e: wp.array(dtype=float), hx: wp.array(dtype=float), hy: wp.array(dtype=float),
             ox: wp.array(dtype=int), cx: wp.array(dtype=int), vx: wp.array(dtype=float),
             oy: wp.array(dtype=int), cy: wp.array(dtype=int), vy: wp.array(dtype=float),
             area: wp.array(dtype=float), alpha: wp.array(dtype=float), src: wp.array(dtype=wp.vec2),
             step: wp.array(dtype=int), dt: float, theta: float, midpoint: int, stage: float,
             omega: float, beta: float, vg: float, sigma: float,
             outx: wp.array(dtype=float), outy: wp.array(dtype=float), kind: int):
    i=wp.tid();gx=float(0.0);gy=float(0.0)
    for j in range(ox[i],ox[i+1]): gx+=vx[j]*e[cx[j]]
    for j in range(oy[i],oy[i+1]): gy+=vy[j]*e[cy[j]]
    t=(float(step[0])+stage)*dt
    source=src[i][0]*forcing(t,src[i][1],dt,theta,midpoint,omega,beta,vg,sigma)
    dx=-gy/area[i];dy=gx/area[i]
    if kind == 0: # ordinary first-order RHS
        outx[i]=dx-alpha[i]*hx[i]
        outy[i]=dy-alpha[i]*hy[i]-source
    elif kind == 1: # Hbar independent of unknown E1
        den=1.0+theta*dt*alpha[i]
        outx[i]=(hx[i]+theta*dt*(1.0-theta)*dx)/den
        outy[i]=(hy[i]+theta*dt*((1.0-theta)*dy-source))/den
    else: # recover H1 from solved Etheta
        den=1.0+theta*dt*alpha[i]
        outx[i]=((1.0-(1.0-theta)*dt*alpha[i])*hx[i]+dt*dx)/den
        outy[i]=((1.0-(1.0-theta)*dt*alpha[i])*hy[i]+dt*(dy-source))/den

@wp.kernel
def electric(e: wp.array(dtype=float), hx: wp.array(dtype=float), hy: wp.array(dtype=float),
             ox: wp.array(dtype=int), cx: wp.array(dtype=int), vx: wp.array(dtype=float),
             oy: wp.array(dtype=int), cy: wp.array(dtype=int), vy: wp.array(dtype=float),
             mass: wp.array(dtype=float), alpha: wp.array(dtype=float), src: wp.array(dtype=wp.vec2),
             step: wp.array(dtype=int), dt: float, theta: float, midpoint: int, stage: float,
             omega: float, beta: float, vg: float, sigma: float,
             scale: wp.array(dtype=float), out: wp.array(dtype=float), kind: int):
    i=wp.tid();r=float(0.0)
    for j in range(ox[i],ox[i+1]):r-=vx[j]*hy[cx[j]]
    for j in range(oy[i],oy[i+1]):r+=vy[j]*hx[cy[j]]
    source=src[i][0]*(beta/omega)*forcing((float(step[0])+stage)*dt,src[i][1],dt,theta,midpoint,omega,beta,vg,sigma)
    if kind == 0:out[i]=(r+source)/mass[i]-alpha[i]*e[i]
    else:out[i]=(mass[i]*(1.0-(1.0-theta)*dt*alpha[i])*e[i]+dt*(r+source))/scale[i]

@wp.kernel
def solved(e: wp.array(dtype=float), old: wp.array(dtype=float), u: wp.array(dtype=float),
           scale: wp.array(dtype=float), etheta: wp.array(dtype=float), theta: float):
    i=wp.tid();old[i]=e[i];e[i]=u[i]/scale[i];etheta[i]=(1.0-theta)*old[i]+theta*e[i]

@wp.kernel
def finish_h(h: wp.array(dtype=float), new: wp.array(dtype=float), mid: wp.array(dtype=float)):
    i=wp.tid();mid[i]=0.5*(h[i]+new[i]);h[i]=new[i]

@wp.kernel
def rk_combine(base: wp.array(dtype=float), deriv: wp.array(dtype=float), acc: wp.array(dtype=float),
               stage: wp.array(dtype=float), dt: float, weight: float, next_weight: float, first: int):
    i=wp.tid()
    if first != 0:acc[i]=base[i]
    acc[i]+=weight*dt*deriv[i]
    stage[i]=base[i]+next_weight*dt*deriv[i]

@wp.kernel
def diagnostic_condition(step: wp.array(dtype=int), flag: wp.array(dtype=int)):
    flag[0]=int(step[0]%32==0)

@wp.kernel
def subtract(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    i=wp.tid();a[i]-=b[i]

@wp.kernel
def diagnostic_max(rr: wp.array(dtype=float), bb: wp.array(dtype=float), stats: wp.array(dtype=float)):
    if bb[0]>1.e-20:
        stats[0]=wp.max(stats[0],wp.sqrt(rr[0]/bb[0]))
        stats[1]+=1.0


def host_csr(b):
    return sparse.csr_matrix((b.values.numpy().ravel()[:b.nnz],b.columns.numpy()[:b.nnz],b.offsets.numpy()),shape=(b.nrow,b.ncol))

@dataclass
class IntegratorConfig:
    method: str = 'midpoint'
    backend: str = 'cudss'
    dt_fs: float = 0.02
    theta: float = 0.55
    tolerance: float = 1e-6
    maxiter: int = 100

class TimeIntegrator(Simulator):
    def __init__(self, mesh, run=None, integration=None):
        super().__init__(mesh,run or RunConfig())
        self.integration=integration or IntegratorConfig();q=self.integration
        if q.method not in ('rk4','midpoint','cn','theta'):raise ValueError(q.method)
        if q.dt_fs<=0:raise ValueError('dt must be positive')
        self.dt=q.dt_fs/TIME_FS
        if q.method=='rk4' and self.dt>1.4*self.dtlimit:raise ValueError('RK4 step exceeds conservative stability bound')
        self.theta=0.5 if q.method in ('midpoint','cn') else q.theta
        self.nsteps=int(np.ceil(self.config.duration_fs/q.dt_fs))
        u=self.config.graph_unroll;self.nsteps=((self.nsteps+u-1)//u)*u
        self.sample_stride=max(1,round(self.config.sample_fs/q.dt_fs));self.frame_stride=max(1,round(self.config.frame_fs/q.dt_fs))
        with wp.ScopedDevice(self.device):
            self.trace=wp.zeros(((self.nsteps-1)//self.sample_stride+1,len(self.pnorm),4))
            if self.frames is not None:self.frames=wp.zeros(((self.nsteps-1)//self.frame_stride+1,len(self.rids),3))
            self.hxm=wp.zeros_like(self.hx);self.hym=wp.zeros_like(self.hy)
            self.stage=[wp.zeros_like(a) for a in (self.e,self.hx,self.hy)]
            self.deriv=[wp.zeros_like(a) for a in self.stage];self.acc=[wp.zeros_like(a) for a in self.stage]
            self.scale=wp.ones_like(self.e)
            self.solver=None
            if q.method!='rk4':
                bx,by=host_csr(self.bx),host_csr(self.by)
                den=self.area.numpy()*(1+self.theta*self.dt*self.ah.numpy())
                w=sparse.diags(1/den)
                a=sparse.diags(self.mass.numpy()*(1+self.theta*self.dt*self.ae.numpy()))+(self.theta*self.dt)**2*(bx.T@w@bx+by.T@w@by)
                scale=np.sqrt(a.diagonal());d=sparse.diags(1/scale)
                self.matrix=(d@a@d).astype(np.float32).tocsr();self.matrix.sort_indices()
                self.scale=wp.array(scale.astype(np.float32))
                self.rhs=wp.zeros_like(self.e);self.solution=wp.zeros_like(self.e)
                self.residual_work=wp.zeros_like(self.e);self.residual_stats=wp.zeros(2)
                self.diag_flag=wp.zeros(1,dtype=int)
                self.diag_dot=TiledDot(max_length=len(self.e),device=wp.get_device(self.device),scalar_type=wp.float32,max_column_count=2)
                coo=self.matrix.tocoo()
                self.a=bsr_from_triplets(len(scale),len(scale),wp.array(coo.row,dtype=int),wp.array(coo.col,dtype=int),wp.array(coo.data))
                if q.backend=='cudss':
                    from cudss_backend import CuDSS
                    self.solver=CuDSS(self.matrix,self.rhs,self.solution,self.device)
                elif q.backend=='cg':
                    self.solver=DeviceCG(self.a,self.rhs,self.solution,tol=q.tolerance,maxiter=q.maxiter)
                else:raise ValueError(q.backend)

    def magnetic(self,e,hx,hy,ox,oy,kind,stage=0.):
        q=self.integration;th=0. if kind==0 else self.theta
        wp.launch(magnetic,len(self.hx),inputs=[e,hx,hy,self.bx.offsets,self.bx.columns,self.bx.values,
            self.by.offsets,self.by.columns,self.by.values,self.area,self.ah,self.sh,self.step,self.dt,
            th,int(q.method=='midpoint' and kind!=0),stage,self.omega,self.md['beta'],1/self.ng,self.config.sigma_fs/TIME_FS,ox,oy,kind],device=self.device)
    def electric(self,e,hx,hy,out,kind,stage=0.):
        q=self.integration;th=0. if kind==0 else self.theta
        wp.launch(electric,len(self.e),inputs=[e,hx,hy,self.tx.offsets,self.tx.columns,self.tx.values,
            self.ty.offsets,self.ty.columns,self.ty.values,self.mass,self.ae,self.se,self.step,self.dt,
            th,int(q.method=='midpoint' and kind!=0),stage,self.omega,self.md['beta'],1/self.ng,self.config.sigma_fs/TIME_FS,self.scale,out,kind],device=self.device)
    def one_step(self):
        if self.integration.method=='rk4':
            bases=[self.e,self.hx,self.hy]
            wp.copy(self.old,self.e)
            for k,(c,w,a) in enumerate(zip([0.,.5,.5,1.],[1/6,1/3,1/3,1/6],[.5,.5,1.,0.])):
                state=bases if k==0 else self.stage
                self.magnetic(*state,*self.deriv[1:],0,c);self.electric(*state,self.deriv[0],0,c)
                for b,d,acc,s in zip(bases,self.deriv,self.acc,self.stage):
                    wp.launch(rk_combine,len(b),inputs=[b,d,acc,s,self.dt,w,a,int(k==0)],device=self.device)
            wp.copy(self.e,self.acc[0])
            newh=self.acc[1:]
        else:
            self.magnetic(self.e,self.hx,self.hy,self.stage[1],self.stage[2],1)
            self.electric(self.e,self.stage[1],self.stage[2],self.rhs,1)
            self.solver()
            wp.launch(diagnostic_condition,1,inputs=[self.step,self.diag_flag],device=self.device)
            if wp.get_device(self.device).is_capturing:wp.capture_if(self.diag_flag,self.diagnose)
            else:self.diagnose()
            wp.launch(solved,len(self.e),inputs=[self.e,self.old,self.solution,self.scale,self.stage[0],self.theta],device=self.device)
            self.magnetic(self.stage[0],self.hx,self.hy,self.deriv[1],self.deriv[2],2)
            newh=self.deriv[1:]
        for h,new,mid in zip([self.hx,self.hy],newh,[self.hxm,self.hym]):
            wp.launch(finish_h,len(h),inputs=[h,new,mid],device=self.device)
        wp.launch(sample_ports,len(self.pnorm),inputs=[self.e,self.old,self.hym,self.f,self.pids,self.pweights,self.pqw,self.poffsets,self.pnorm,self.step,self.sample_stride,self.trace],device=self.device)
        if self.frames is not None:
            wp.launch(sample_frames,len(self.rids),inputs=[self.e,self.old,self.hxm,self.hym,self.f,self.rids,self.rweights,self.step,self.frame_stride,self.frames],device=self.device)
        wp.launch(increment,1,inputs=[self.step,self.condition,self.nsteps],device=self.device)
    def diagnose(self):
        bsr_mv(self.a,self.solution,self.residual_work)
        wp.launch(subtract,len(self.e),inputs=[self.residual_work,self.rhs],device=self.device)
        self.diag_dot.compute(self.residual_work,self.residual_work,col_offset=0)
        self.diag_dot.compute(self.rhs,self.rhs,col_offset=1)
        wp.launch(diagnostic_max,1,inputs=[self.diag_dot.col(0),self.diag_dot.col(1),self.residual_stats],device=self.device)
    def reset(self):
        super().reset()
        if hasattr(self,'solution'):
            self.solution.zero_();self.residual_stats.zero_()
        if isinstance(self.solver,DeviceCG):self.solver.stats.zero_()
    def close(self):
        if self.integration.backend=='cudss' and self.solver is not None:self.solver.close()
