import unittest
from pathlib import Path


class StateTests(unittest.TestCase):
    def test_input_and_state(self):
        from crmf.finite_nucleus.state import NucleusConfig, new_state
        cfg = NucleusConfig.from_file(Path(__file__).resolve().parents[1] / 'dirhb.dat')
        self.assertEqual((cfg.symbol, cfg.mass_number, cfg.n0f, cfg.n0b), ('Ca', 48, 20, 30))
        s = new_state(cfg)
        self.assertEqual((s.nneu, s.npro), (28, 20))
        self.assertEqual(s.npr.tolist(), [0, 28, 20])
        self.assertEqual(s.de.shape, (s.nhhx + 1, s.nb2x + 1))
        self.assertEqual(s.rnl.shape, (s.nrx + 1, s.nlx + 1, s.ngh + 1))
        self.assertEqual(s.del_[1:].tolist(), [1.0, 1.0])

    def test_reject_impossible_nucleus(self):
        from crmf.finite_nucleus.state import NucleusConfig
        with self.assertRaises(ValueError):
            NucleusConfig(symbol='Ca', mass_number=10)

    def test_reject_noninteger_and_nonfinite_input(self):
        from crmf.finite_nucleus.state import NucleusConfig
        for overrides in ({'mass_number':48.5}, {'n0f':4.5}, {'max_iterations':3.5},
                          {'tolerance':float('nan')}, {'constraint':float('inf')},
                          {'gap_neutron':float('nan')}, {'gap_proton':-1.}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                NucleusConfig(**overrides)

    def test_constraint_step_must_be_finite(self):
        from crmf.finite_nucleus.nucleus import constrained_monopole
        with self.assertRaises(ValueError):
            constrained_monopole(step=float('nan'))


if __name__ == '__main__':
    unittest.main()
