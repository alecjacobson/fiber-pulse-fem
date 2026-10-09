"""Numerical verification; run with python -m unittest -v test_waveguide."""
import unittest
import numpy as np
import scipy.sparse as sp
import warp as wp
from mesh_waveguide import MeshConfig, build, operators
from warp_waveguide import Simulator, RunConfig, TIME_FS


class WaveguideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.mesh = build(MeshConfig(h=0.12,clad_ratio=2,bend=1.5))

    def test_mesh_resolution_and_conformity(self):
        from mesh_waveguide import geometry
        for h in [0.12,0.071,0.04]:
            m = build(MeshConfig(h=h))
            v,f = m['v'],m['f']
            area = np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])/2
            self.assertTrue(np.all(area>0))
            self.assertAlmostEqual(area.sum(),18*8,places=9)
            self.assertLess(area.max(),0.5*(2*h)**2*(1+1e-8))
            # Interface edges have one core and one cladding neighbor. Their
            # midpoint must lie on the original PSLG to near roundoff.
            import igl
            edges = np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
            flags = np.tile(m['core'],3).astype(int)
            unique,inv,count = np.unique(edges,axis=0,return_inverse=True,return_counts=True)
            adjacent = np.bincount(inv,flags)
            interface = unique[(count==2)&(adjacent==1)]
            outline,_ = geometry(MeshConfig(h=h))
            oe = np.column_stack((np.arange(len(outline)),np.roll(np.arange(len(outline)),-1)))
            distance = igl.point_mesh_squared_distance(v[interface].mean(axis=1),outline,oe)[0]
            self.assertLess(np.max(distance),1e-16)

    @unittest.skipUnless(wp.is_cuda_available(), 'CUDA required')
    def test_fem_assembly_and_full_graph_against_numpy(self):
        r = RunConfig(duration_fs=35)
        sim = Simulator(self.mesh,r)
        area,grad,mass,vol,_ = operators(self.mesh)
        f = self.mesh['f']
        nt,nv = len(f),len(self.mesh['v'])
        bx = sp.csr_matrix(((area[:,None]*grad[:,:,0]).ravel(),
                            (np.repeat(np.arange(nt),3),f.ravel())),shape=(nt,nv))
        by = sp.csr_matrix(((area[:,None]*grad[:,:,1]).ravel(),
                            (np.repeat(np.arange(nt),3),f.ravel())),shape=(nt,nv))
        np.testing.assert_allclose(sim.mass.numpy(),mass,rtol=2e-4)
        np.testing.assert_allclose(sim.area.numpy(),area,rtol=2e-4)
        def host_bsr(b):
            n = b.offsets.numpy()[-1]
            return sp.csr_matrix((b.values.numpy()[:n],b.columns.numpy()[:n],b.offsets.numpy()),shape=(b.nrow,b.ncol))
        for actual,expected in [(host_bsr(sim.bx),bx),(host_bsr(sim.by),by)]:
            self.assertLess(sp.linalg.norm(actual-expected)/sp.linalg.norm(expected),2e-5)
        sim.run()
        e = np.zeros(nv)
        hx,hy = np.zeros(nt),np.zeros(nt)
        ae,ah = sim.ae.numpy(),sim.ah.numpy()
        se,sh = sim.se.numpy(),sim.sh.numpy()
        dt,om,be,vg,sg = sim.dt,sim.omega,sim.md['beta'],1/sim.ng,r.sigma_fs/TIME_FS
        def pulse(t,x):
            return np.exp(-0.5*((t-4.5*sg-x/vg)/sg)**2)*np.cos(om*(t-4.5*sg)-be*x)
        for n in range(sim.nsteps):
            hx = (1-ah*dt/2)/(1+ah*dt/2)*hx-dt/(1+ah*dt/2)*(by@e)/area
            hy = (1-ah*dt/2)/(1+ah*dt/2)*hy+dt/(1+ah*dt/2)*((bx@e)/area-sh[:,0]*pulse(n*dt,sh[:,1]))
            e = (1-ae*dt/2)/(1+ae*dt/2)*e+dt/(mass*(1+ae*dt/2))*(by.T@hx-bx.T@hy+se[:,0]*be/om*pulse((n+0.5)*dt,se[:,1]))
        for a,b in [(sim.e.numpy(),e),(sim.hx.numpy(),hx),(sim.hy.numpy(),hy)]:
            error = np.linalg.norm(a-b)/np.linalg.norm(b)
            print('graph vs independent float64 NumPy relative error:',error)
            self.assertLess(error,3e-4)
        self.assertEqual(sim.step.numpy()[0],sim.nsteps)
        self.assertEqual(sim.condition.numpy()[0],0)
        # Replay after reset must reproduce the complete trajectory exactly.
        before = sim.e.numpy().copy()
        sim.reset()
        wp.capture_launch(sim.graph)
        wp.synchronize_device(sim.device)
        np.testing.assert_array_equal(sim.e.numpy(),before)
        self.assertTrue(np.isfinite(sim.trace.numpy()).all())

    @unittest.skipUnless(wp.is_cuda_available(), 'CUDA required')
    def test_lossless_discrete_energy(self):
        sim = Simulator(self.mesh,RunConfig(duration_fs=1))
        sim.ae.zero_(); sim.ah.zero_(); sim.se.zero_(); sim.sh.zero_()
        v = self.mesh['v']
        init = np.exp(-((v[:,0]-8)**2+(v[:,1]-4)**2))*np.cos(5*v[:,0])
        sim.e.assign(init.astype(np.float32))
        wp.synchronize()
        mass,area = sim.mass.numpy(),sim.area.numpy()
        energies = []
        for _ in range(120):
            e = sim.e.numpy()
            hm = np.column_stack((sim.hx.numpy(),sim.hy.numpy()))
            sim.one_step()
            hp = np.column_stack((sim.hx.numpy(),sim.hy.numpy()))
            energies.append(0.5*np.dot(mass,e**2)+0.5*np.sum(area[:,None]*hm*hp))
            # Sampling buffers only cover duration_fs: rewind diagnostic index.
            sim.step.zero_()
        drift = np.ptp(energies)/np.mean(energies)
        print('lossless modified-energy relative drift:',drift)
        self.assertLess(drift,5e-6)


if __name__=='__main__':
    unittest.main()
