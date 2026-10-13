"""Preallocated conjugate gradient with a device-controlled stopping condition."""
import warp as wp
from warp.sparse import bsr_mv
from warp._src.optim.linear import TiledDot  # pinned Warp 1.15 reduction workspace

@wp.kernel
def init_r(b: wp.array(dtype=float), ax: wp.array(dtype=float), r: wp.array(dtype=float), p: wp.array(dtype=float)):
    i=wp.tid();r[i]=b[i]-ax[i];p[i]=r[i]
@wp.kernel
def update_xr(x: wp.array(dtype=float), r: wp.array(dtype=float), p: wp.array(dtype=float), ap: wp.array(dtype=float), rr: wp.array(dtype=float), pap: wp.array(dtype=float)):
    i=wp.tid();a=rr[0]/wp.max(pap[0],1.e-35);x[i]+=a*p[i];r[i]-=a*ap[i]
@wp.kernel
def update_p(p: wp.array(dtype=float), r: wp.array(dtype=float), rr: wp.array(dtype=float), newrr: wp.array(dtype=float)):
    i=wp.tid();p[i]=r[i]+newrr[0]/wp.max(rr[0],1.e-35)*p[i]
@wp.kernel
def condition(rr: wp.array(dtype=float), bb: wp.array(dtype=float), tol: float, maxiter: int,
              count: wp.array(dtype=int), active: wp.array(dtype=int), stats: wp.array(dtype=int), first: int):
    if first != 0:count[0]=0
    else:count[0]+=1
    unfinished=rr[0]>wp.max(tol*tol*bb[0],1.e-30)
    active[0]=int(unfinished and count[0]<maxiter)
    if active[0]==0:
        wp.atomic_max(stats,0,count[0])
        wp.atomic_add(stats,1,int(unfinished))

class DeviceCG:
    def __init__(self,a,b,x,tol=1e-6,maxiter=100):
        self.a,self.b,self.x,self.tol,self.maxiter=a,b,x,tol,maxiter
        self.r,self.p,self.ap=[wp.zeros_like(b) for _ in range(3)]
        self.dot=TiledDot(max_length=b.size,device=b.device,scalar_type=wp.float32,max_column_count=1)
        self.rr,self.bb,self.pap,self.newrr=[wp.zeros(1,device=b.device) for _ in range(4)]
        self.count=wp.zeros(1,dtype=int,device=b.device);self.active=wp.zeros_like(self.count)
        self.stats=wp.zeros(2,dtype=int,device=b.device)
    def inner(self,a,b,out):
        self.dot.compute(a,b);wp.copy(out,self.dot.col(0))
    def check(self,first):
        wp.launch(condition,1,inputs=[self.newrr,self.bb,self.tol,self.maxiter,self.count,self.active,self.stats,first],device=self.b.device)
    def iteration(self):
        wp.copy(self.rr,self.newrr)
        bsr_mv(self.a,self.p,self.ap)
        self.inner(self.p,self.ap,self.pap)
        wp.launch(update_xr,len(self.b),inputs=[self.x,self.r,self.p,self.ap,self.rr,self.pap],device=self.b.device)
        self.inner(self.r,self.r,self.newrr)
        wp.launch(update_p,len(self.b),inputs=[self.p,self.r,self.rr,self.newrr],device=self.b.device)
        self.check(0)
    def __call__(self):
        self.inner(self.b,self.b,self.bb);bsr_mv(self.a,self.x,self.ap)
        wp.launch(init_r,len(self.b),inputs=[self.b,self.ap,self.r,self.p],device=self.b.device)
        self.inner(self.r,self.r,self.newrr);self.check(1)
        if self.b.device.is_capturing:wp.capture_while(self.active,self.iteration)
        else:
            with wp.ScopedCapture(device=self.b.device) as c:wp.capture_while(self.active,self.iteration)
            wp.capture_launch(c.graph)
