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
            a,b = v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]
            area = (a[:,0]*b[:,1]-a[:,1]*b[:,0])/2
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
            expected = np.abs(igl.winding_number(outline,oe,v[f].mean(axis=1)))>0.5
            np.testing.assert_array_equal(m['core'],expected)

    def test_original_host_mesher_accepts_nonpreset_resolution(self):
        from fiber_pulse import channel_outline, build_mesh
        lx = 25e-6
        _,_,outline = channel_outline(lx,0.5e-6)
        counts = []
        for h in [0.08e-6,0.047e-6]:
            v,f,core = build_mesh(outline,lx,h,2*h)
            self.assertTrue(core.any())
            a,b = v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]
            area = np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])/2
            self.assertAlmostEqual(area.sum(),1.,places=10)
            self.assertLessEqual(area.max(),0.5*(2*h/lx)**2*(1+1e-8))
            counts.append(len(f))
        self.assertGreater(counts[1],counts[0])

    def test_physical_metrics_on_analytic_pulses(self):
        from analyze_waveguide import metrics
        from warp_waveguide import mode_data
        from fiber_pulse import slab_mode
        c,r = MeshConfig(bend=0),RunConfig()
        md,ng = mode_data(c,r)
        omega = 2*np.pi/r.wavelength_um
        modes = [slab_mode(omega,c.width,2.04,1.46,i) for i in range(3)]
        ports = np.r_[np.arange(4,5.61,0.2),14.5]
        t = np.arange(0,170,0.06)
        arrival = 20+ports*ng*TIME_FS
        signal = np.exp(-0.5*((t[:,None]-arrival)/4)**2)*np.cos(omega*t[:,None]/TIME_FS-md['beta']*ports)
        trace = np.zeros((len(t),14,4))
        trace[:,:10,0] = signal
        trace[:,9,0] *= np.sqrt(0.6)
        trace[:,:10,1] = -md['neff']*trace[:,:10,0]
        trace[:,10,0] = 0.1*signal[:,3]
        trace[:,11,0] = np.sqrt(0.2*md['neff']/modes[1]['neff'])*signal[:,9]
        trace[:,10:12,1] = -modes[1]['neff']*trace[:,10:12,0]
        trace[:,:10,2] = md['neff']*trace[:,:10,0]**2
        trace[:,3,2] += modes[1]['neff']*trace[:,10,0]**2
        trace[:,9,2] += modes[1]['neff']*trace[:,11,0]**2
        trace[:,:,3] = trace[:,:,2]
        result = metrics(t,trace,ports,md,ng,np.array([[0,3],[18,3]]),r,modes,np.ones(14))
        self.assertLess(abs(result['beta_relative_error']),1e-9)
        self.assertAlmostEqual(result['modal_transmission'],0.6,places=9)
        expected = 0.8/(1+0.01*modes[1]['neff']/md['neff'])
        self.assertAlmostEqual(result['total_guided_transmission'],expected,places=9)
        self.assertAlmostEqual(result['net_flux_transmission'],expected,places=9)
        self.assertLess(abs(result['delay_relative_error']),1e-8)

    @unittest.skipUnless(wp.is_cuda_available(), 'CUDA required')
    def test_fem_assembly_and_full_graph_against_numpy(self):
        r = RunConfig(duration_fs=35,graph_unroll=4)
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
