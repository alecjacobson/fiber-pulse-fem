"""Physical units and independent source-free propagation regression gates."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import warp as wp
from fiber_pulse import C0
from warp_waveguide import TIME_FS
from check_light_speed import HomogeneousPulse, measure_speed


class LightSpeedTests(unittest.TestCase):
    def test_si_units_and_flight_time_estimator(self):
        self.assertEqual(C0,299792458.0)
        self.assertAlmostEqual(TIME_FS,3.3356409519815204,places=14)
        # Use literal SI speed, independent of simulator constants and mode data.
        for n in [1.,1.46,2.04]:
            v = 299792458.0/n
            x = np.arange(0,4.001,0.08)
            t = np.arange(0,100,.04)
            arrival = 20+1e9*x/v
            omega = 2*np.pi*299792458.0/0.532*1e-9
            e = np.exp(-0.5*((t[:,None]-arrival)/3)**2)*np.cos(omega*(t[:,None]-arrival))
            m = measure_speed(t,x,e,.532)
            self.assertLess(abs(m['group_speed_m_s']/v-1),1e-8)
            self.assertLess(abs(m['phase_speed_m_s']/v-1),1e-8)

    @unittest.skipUnless(wp.is_cuda_available(),'CUDA required')
    def test_source_free_speed_convergence_and_graph_replay(self):
        for n in [1.,2.04]:
            errors = []
            for h in [.02,.01]:
                s = HomogeneousPulse(h,n)
                trace = s.run()
                m,_ = s.measure(trace)
                expected = 299792458.0/n
                errors.append(abs(m['group_speed_m_s']/expected-1))
                self.assertLess(abs(m['phase_speed_m_s']/expected-1),.009)
                self.assertLess(m['arrival_fit_rms_fs'],.002)
                self.assertTrue(np.isfinite(trace).all())
                self.assertEqual(s.step.numpy()[0],s.nsteps)
            self.assertLess(errors[-1],.006)
            self.assertGreater(errors[0]/errors[1],3.6)
            self.assertLess(errors[0]/errors[1],4.5)
            s.reset(); wp.capture_launch(s.graph); wp.synchronize()
            np.testing.assert_array_equal(s.trace.numpy(),trace)
            with tempfile.TemporaryDirectory() as td:
                path = Path(td)/'graph.dot'
                wp.capture_debug_dot_print(s.graph,str(path))
                graph = path.read_text()
                self.assertIn('WHILE',graph)
                self.assertNotIn('HtoD',graph)
                self.assertNotIn('DtoH',graph)


if __name__=='__main__':
    unittest.main()
