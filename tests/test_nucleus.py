import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class NucleusRegressionTests(unittest.TestCase):
    def test_default_five_models(self):
        from crmf.finite_nucleus.nucleus import solve_nucleus
        from crmf.finite_nucleus.state import NucleusConfig
        cfg = NucleusConfig.from_file(ROOT/'tests/fixtures/dirhb.dat')
        reference = np.loadtxt(ROOT/'tests/fixtures/Ca_results.txt', skiprows=1, usecols=range(1,7))
        for i, model in enumerate(('new1','new2','new3','new4','new5')):
            with self.subTest(model=model):
                state = solve_nucleus(cfg, model)
                actual = [state.ea, state.radius_charge,
                          state.radius_neutron-state.radius_proton,
                          state.els_2d_p, state.els_3p_n, state.els_2f_n]
                np.testing.assert_allclose(actual, reference[i], atol=2e-5, rtol=2e-6)
                self.assertTrue(state.converged)
                self.assertLess(state.si, cfg.tolerance)
                np.testing.assert_allclose(state.xn[1:], [28,20,48], atol=1e-6, rtol=0)

    def test_nonconvergence_is_an_error(self):
        from crmf.finite_nucleus.nucleus import solve_nucleus
        from crmf.finite_nucleus.state import NucleusConfig
        with self.assertRaisesRegex(RuntimeError, 'converge'):
            solve_nucleus(NucleusConfig(n0f=4,n0b=6,max_iterations=1), 'new1')


if __name__ == '__main__':
    unittest.main()
