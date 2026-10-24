from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.io import FortranFile

from crmf.finite_nucleus.state import NucleusConfig, new_state, apply_config
from crmf.finite_nucleus import finite_setup as setup
from crmf.finite_nucleus.io import initialize_pairing, read_potentials, write_restart

FIXTURES = Path(__file__).parent/'fixtures'


def state_from_input():
    s = new_state(NucleusConfig.from_file(FIXTURES/'dirhb.dat'))
    setup.default(s)
    apply_config(s,s.config)
    setup.forces(s,'new5')
    setup.base(s)
    return s


class RestartTests(unittest.TestCase):
    def test_read_original_fortran_restart(self):
        s = state_from_input()
        s.inink = 0
        initialize_pairing(s,FIXTURES)
        read_potentials(s,FIXTURES)
        self.assertTrue(np.isfinite(s.de).all())
        self.assertGreater(np.max(np.abs(s.de)),0)
        self.assertGreater(np.max(np.abs(s.vps)),1)

    def test_write_record_lengths_and_roundtrip(self):
        s = state_from_input()
        initialize_pairing(s)
        rng = np.random.default_rng(123)
        s.ala[1:] = [-7.,-12.]
        s.vps[:,1:] = rng.normal(size=(s.ngh+1,2))
        s.vms[:,1:] = rng.normal(size=(s.ngh+1,2))
        with tempfile.TemporaryDirectory() as folder:
            write_restart(s,folder)
            with FortranFile(Path(folder)/'dirhb.del','r') as f:
                self.assertEqual(f.read_ints(np.int32)[0],s.mv)
                self.assertEqual(len(f.read_reals(np.float64)),s.id[1,1])
            restored = state_from_input()
            restored.inink = 0
            initialize_pairing(restored,folder)
            read_potentials(restored,folder)
            for name in ('de','ala','vps','vms'):
                np.testing.assert_array_equal(getattr(s,name),getattr(restored,name))


if __name__=='__main__':
    unittest.main()
