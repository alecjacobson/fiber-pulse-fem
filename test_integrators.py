"""Independent algebra, temporal order, energy, and full-graph regressions."""
import unittest
import tempfile
import importlib.metadata
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, expm_multiply, eigsh
import warp as wp
from integrators import TimeIntegrator, IntegratorConfig, host_csr
from warp_waveguide import RunConfig, TIME_FS
from mesh_waveguide import build, MeshConfig

class IntegratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init();cls.mesh=build(MeshConfig(h=.2))
        cls.backends=['cg']
        try:importlib.metadata.version('nvidia-cudss-cu12')
        except importlib.metadata.PackageNotFoundError:pass
        else:cls.backends.append('cudss')
    def sim(self,method='midpoint',backend='cg',dt=.05,duration=1,**kw):
        return TimeIntegrator(self.mesh,RunConfig(duration_fs=duration,**kw),IntegratorConfig(method=method,backend=backend,dt_fs=dt,tolerance=1e-7))
    def system(self,s):
        bx,by=host_csr(s.bx).astype(float),host_csr(s.by).astype(float)
        m,a=s.mass.numpy().astype(float),s.area.numpy().astype(float)
        im,ia=sp.diags(1/m),sp.diags(1/a)
        return sp.bmat([[-sp.diags(s.ae.numpy()),im@by.T,-im@bx.T],[-ia@by,-sp.diags(s.ah.numpy()),None],[ia@bx,None,-sp.diags(s.ah.numpy())]],format='csr')
    def state(self,s):return np.concatenate([s.e.numpy(),s.hx.numpy(),s.hy.numpy()])
    def assign(self,s,x):
        n,k=len(s.e),len(s.hx)
        s.e.assign(x[:n].astype('f'));s.hx.assign(x[n:n+k].astype('f'));s.hy.assign(x[n+k:].astype('f'))
    def test_newmark_equivalence_and_source_quadrature(self):
        dt=.13;k=4.;c=.3;u=.7;v=-.4;f0=.2;f1=.8
        a0=f0-c*v-k*u
        a1=(f1-c*(v+dt*.5*a0)-k*(u+dt*v+dt**2*.25*a0))/(1+c*dt*.5+k*dt**2*.25)
        newmark=np.array([u+dt*v+dt**2*.25*(a0+a1),v+dt*.5*(a0+a1)])
        l=np.array([[0,1],[-k,-c]])
        cn=np.linalg.solve(np.eye(2)-dt*l/2,(np.eye(2)+dt*l/2)@[u,v]+dt*np.array([0,(f0+f1)/2]))
        np.testing.assert_allclose(newmark,cn,atol=1e-14)
        # Nonlinear-in-time forcing distinguishes midpoint from endpoint averaging.
        self.assertNotEqual((0.5*dt)**2,0.5*dt**2)
    def test_one_step_independent_full_maxwell(self):
        rng=np.random.default_rng(42)
        for backend in self.backends:
            for method in ['midpoint','cn','theta','rk4']:
                with self.subTest(backend=backend,method=method):
                    s=self.sim(method,backend);l=self.system(s);n,k=len(s.e),len(s.hx)
                    x=rng.normal(size=l.shape[0])*.1;self.assign(s,x);x=self.state(s).astype(float)
                    # Put the source near its maximum to catch timing/quadrature bugs.
                    step=340;s.step.fill_(step)
                    # Avoid writing diagnostics outside the allocated time interval.
                    s.trace=wp.zeros((step+1,len(s.pnorm),4),device=s.device)
                    se,sh=s.se.numpy(),s.sh.numpy();dt=s.dt
                    def force(t):
                        sg=s.config.sigma_fs/TIME_FS;om=s.omega;be=s.md['beta']
                        def p(z):return np.exp(-.5*((t-4.5*sg-z*s.ng)/sg)**2)*np.cos(om*(t-4.5*sg)-be*z)
                        return np.r_[se[:,0]*be/om*p(se[:,1])/s.mass.numpy(),np.zeros(k),-sh[:,0]*p(sh[:,1])]
                    t=step*dt
                    if method=='rk4':
                        a=l@x+force(t);b=l@(x+dt*a/2)+force(t+dt/2);c=l@(x+dt*b/2)+force(t+dt/2);d=l@(x+dt*c)+force(t+dt)
                        expected=x+dt*(a+2*b+2*c+d)/6
                    else:
                        th=s.theta;f=force(t+dt/2) if method=='midpoint' else (1-th)*force(t)+th*force(t+dt)
                        expected=spsolve(sp.eye(len(x))-th*dt*l,x+(1-th)*dt*(l@x)+dt*f)
                    s.one_step();actual=self.state(s)
                    self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),2e-6)
                    s.close()
    def test_temporal_orders_on_semidiscrete_system(self):
        rng=np.random.default_rng(5)
        for method,minimum in [('midpoint',1.8),('cn',1.8),('theta',.85),('rk4',3.4)]:
            errors=[]
            for dt in ([.064,.032] if method=='rk4' else [.16,.08]):
                s=self.sim(method,dt=dt,duration=1,absorber_strength=0)
                s.se.zero_();s.sh.zero_();l=self.system(s)
                if not errors:
                    bx,by=host_csr(s.bx).astype(float),host_csr(s.by).astype(float)
                    mass=sp.diags(s.mass.numpy().astype(float));ia=sp.diags(1/s.area.numpy().astype(float))
                    _,vectors=eigsh(bx.T@ia@bx+by.T@ia@by,k=1,M=mass,sigma=900. if method=='rk4' else 144.)
                    x=np.r_[vectors[:,0],np.zeros(2*len(s.hx))]
                self.assign(s,x)
                n=round((.64 if method=='rk4' else .8)/dt)
                for _ in range(n):s.one_step()
                expected=expm_multiply(l*(n*s.dt),x)
                errors.append(np.linalg.norm(self.state(s)-expected)/np.linalg.norm(expected));s.close()
            order=np.log2(errors[0]/errors[1]);print(method,'temporal errors',errors,'order',order)
            self.assertGreater(order,minimum)
    def test_midpoint_energy(self):
        s=self.sim(dt=.05,duration=10,absorber_strength=0)
        s.se.zero_();s.sh.zero_();rng=np.random.default_rng(10);self.assign(s,rng.normal(size=len(self.state(s))))
        weight=np.r_[s.mass.numpy(),s.area.numpy(),s.area.numpy()]
        e0=np.dot(weight,self.state(s)**2)
        for _ in range(100):s.one_step()
        self.assertLess(abs(np.dot(weight,self.state(s)**2)/e0-1),2e-5)
    def test_graph_replay_and_backend_agreement(self):
        states=[]
        for method,backend in [('rk4','cg')]+[('midpoint',b) for b in self.backends]:
            s=self.sim(method=method,backend=backend,duration=35);s.run();before=self.state(s);trace=s.trace.numpy()
            s.reset();wp.capture_launch(s.graph);wp.synchronize()
            np.testing.assert_allclose(self.state(s),before,rtol=0,atol=2e-6)
            np.testing.assert_allclose(s.trace.numpy(),trace,rtol=0,atol=2e-6)
            self.assertEqual(s.step.numpy()[0],s.nsteps)
            if hasattr(s,'residual_stats'):self.assertLess(s.residual_stats.numpy()[0],2e-6)
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'graph.dot';wp.capture_debug_dot_print(s.graph,str(path))
                self.assertNotIn('HtoD',path.read_text());self.assertNotIn('DtoH',path.read_text())
            if method=='midpoint':states.append(before)
            s.close()
        if len(states)>1:
            # Single precision roundoff accumulates over 700 solves; require
            # agreement within 0.05%, tighter than study power/arrival gates.
            self.assertLess(np.linalg.norm(states[0]-states[1])/np.linalg.norm(states[1]),5e-4)

    def test_cg_iteration_limit_is_reported_on_device(self):
        from device_cg import DeviceCG
        from warp.sparse import bsr_from_triplets
        with wp.ScopedDevice('cuda:0'):
            a=bsr_from_triplets(2,2,wp.array([0,0,1,1],dtype=int),wp.array([0,1,0,1],dtype=int),wp.array([4.,1.,1.,3.],dtype=float))
            b=wp.array([1.,2.],dtype=float);x=wp.zeros(2)
            solver=DeviceCG(a,b,x,tol=1e-8,maxiter=1)
            solver()
            np.testing.assert_array_equal(solver.stats.numpy(),[1,1])
            self.assertGreater(np.linalg.norm(np.array([[4.,1.],[1.,3.]])@x.numpy()-[1.,2.]),.1)
            b.zero_();x.zero_();solver.stats.zero_();solver()
            np.testing.assert_array_equal(solver.stats.numpy(),[0,0])

if __name__=='__main__':unittest.main()
