"""Physical invariants of the independent Python finite-nucleus setup."""
import math
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from crmf.finite_nucleus import finite_setup as fs


def fixture(n0f=6, n0b=10, ngh=30):
    """Minimal padded COMMON-like state, independent of the application parser."""
    s = SimpleNamespace(n0f=n0f, n0b=n0b, n0fx=20, nrx=11, nlx=21,
                        ndx=11, nfx=11, ngx=11, nox=19, nbx=42,
                        ntx=482, nkx=251, nfgx=121, mvx=946, ngh=ngh,
                        igfv=100, amas=48.0, nama=48, npr=np.array([0,28,20]))
    floats = {'ff':(ngh+1,5,3), 'ala':(3,), 'tz':(3,), 'del_':(3,),
              'spk':(3,), 'spk0':(3,), 'ggmes':(5,), 'ddmes':(5,),
              'a_m':(5,), 'b_m':(5,), 'c_m':(5,), 'd_m':(5,), 'ames':(5,),
              'gl':(3,), 'xh':(ngh+1,), 'ph':(ngh+1,), 'wh':(ngh+1,),
              'rb':(ngh+1,), 'wdcor':(ngh+1,), 'cou':(ngh+1,),
              'vps':(ngh+1,3), 'vms':(ngh+1,3),
              'rnl':(12,22,ngh+1), 'rnl1':(12,22,ngh+1),
              'rnb':(20,ngh+1), 'sp':(122,43), 'ggc':(ngh+1,ngh+1),
              'gg':(ngh+1,ngh+1,5)}
    ints = {'lmes':(5,), 'kb':(43,), 'mb':(43,), 'nrbl':(43,5),
            'ia':(43,3), 'id':(43,3), 'ijb':(43,), 'ilb':(43,3),
            'ipb':(43,), 'ikb':(43,), 'nr':(483,), 'nl':(483,),
            'nj':(483,), 'kk':(483,), 'np':(483,), 'ipos':(43,),
            'nib':(947,), 'nni':(3,947)}
    for name, shape in floats.items():
        setattr(s, name, np.zeros(shape))
    for name, shape in ints.items():
        setattr(s, name, np.zeros(shape, dtype=np.int64))
    s.tb = np.full(43, '', dtype='U8')
    s.tt = np.full(483, '', dtype='U8')
    fs.default(s)
    return s


class FiniteSetupTests(unittest.TestCase):
    def test_half_line_quadrature_moments(self):
        s = fixture()
        fs.gaush(s)
        self.assertTrue(np.all(np.diff(s.xh) > 0))
        for power in (0, 2, 4, 10, 20):
            actual = np.sum(s.ph[1:] * s.xh[1:]**power)
            exact = math.gamma((power+1)/2)/2
            self.assertAlmostEqual(actual/exact, 1.0, places=12)

    def test_oscillator_completeness_and_orthogonality(self):
        s = fixture()
        fs.forces(s, 'new1')
        fs.gaush(s)
        fs.prep(s)
        fs.base(s)
        fs.gaupol(s)
        expected_states = sum((q+1)*(q+2) for q in range(s.n0f+1))
        self.assertEqual(sum(s.mb[1:s.nb+1]*s.id[1:s.nb+1,1]), expected_states)
        self.assertEqual(s.mv, sum(n*(n+1)//2 for n in s.id[1:s.nb+1,1]))
        for l in range(s.nlm+1):
            r = s.rnl[1:s.nrm+1,l,1:]
            np.testing.assert_allclose(r @ r.T, np.eye(s.nrm), atol=3e-13)
        b = s.rnb[1:s.no+1,1:] * np.sqrt(s.wh[1:]*s.xh[1:]**2)
        np.testing.assert_allclose(b @ b.T, np.eye(s.no), atol=3e-13)

    def test_force_units_and_aliases(self):
        s = fixture()
        fs.forces(s, 'new1')
        self.assertAlmostEqual(s.amu*s.hbc, 939.0)
        self.assertAlmostEqual(s.ames[1]*s.hbc, 491.5)
        self.assertAlmostEqual(s.gsig, 10.2143)
        self.assertAlmostEqual(s.ggmes[3], -(5.5/(980/s.hbc+1e-10))**2)
        self.assertEqual(s.icm, 1)
        self.assertEqual(s.icou, 2)
        self.assertAlmostEqual(s.z_ds, .053, places=6)
        fs.forces(s, 'DD-PC1')
        self.assertEqual(s.ipc, 1)
        self.assertAlmostEqual(s.a_m[1], -10.0462)
        self.assertAlmostEqual(s.ddmes[1], -.8149)
        self.assertEqual(s.z_ds, 0)

    def test_full_setup_matches_independent_fortran_reference(self):
        # This binary fixture was produced once by the original setup
        # routines. Tests themselves require neither source nor compiler.
        from crmf.finite_nucleus.state import new_state, NucleusConfig, apply_config
        from crmf.finite_nucleus.pairing import singd
        s = new_state(NucleusConfig(force='new1'))
        fs.default(s)
        apply_config(s, s.config)
        for name in ('forces','gaush','prep','start','base','gaupol','singf'):
            getattr(fs, name)(s)
        singd(s)
        fs.greecou(s)
        fs.greemes(s)
        actual = {
            'scalars':np.array([s.hom,s.hb0,s.b0,s.nb,s.nt,s.no,s.mv]),
            'masses':np.array([s.amu,*s.ames[1:]]),
            'couplings':np.array([s.gsig,s.gome,s.gdel,s.grho]),
            'nonlinear':np.array([s.z_bs,s.z_cs,s.z_cw,s.z_lv,s.z_ds]),
        }
        for name in ('ggmes','a_m','b_m','c_m','d_m','kb','mb','ipos'):
            actual[name] = getattr(s,name)[1:]
        for name in ('ia','id','nni','sp'):
            actual[name] = getattr(s,name)[1:,1:]
        for name in ('xh','ph','wh','rb','wdcor','cou','ggc'):
            actual[name] = getattr(s,name)
        for name in ('vps','vms'):
            actual[name] = getattr(s,name)[:,1:]
        for name in ('rnl','rnl1','rnb','vn'):
            actual[name] = getattr(s,name)[1:]
        actual['gg'] = s.gg[1:,1:,1:]
        fixture_path = Path(__file__).parent/'fixtures'/'reference_setup.npz'
        with np.load(fixture_path) as reference:
            for name in reference.files:
                expected = reference[name]
                with self.subTest(array=name):
                    np.testing.assert_allclose(actual[name], expected,
                        rtol=2e-11, atol=max(1e-13, 1e-12*np.max(np.abs(expected))))

    def test_all_source_force_constants_match_fortran(self):
        fixture_path = Path(__file__).parent/'fixtures'/'reference_forces.json'
        reference = json.loads(fixture_path.read_text())
        self.assertEqual(set(reference), set(fs.FORCE_NAMES)-{'EOS'})
        for name, expected in reference.items():
            with self.subTest(force=name):
                s = fixture()
                fs.forces(s, name)
                actual = [s.amu,*s.ames[1:],s.gsig,s.gome,s.gdel,s.grho,
                          *s.ggmes[1:],*s.lmes[1:],*s.ddmes[1:],
                          *s.a_m[1:],*s.b_m[1:],*s.c_m[1:],*s.d_m[1:],
                          s.dsat,s.z_bs,s.z_cs,s.z_cw,s.z_lv,s.z_ds,
                          s.icm,s.icou,s.ipc,s.idd]
                np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)

    def test_eos_parameter_mapping_and_missing_parameters(self):
        from crmf.nuclear_matter.eos import RMFParameters
        s = fixture()
        p = RMFParameters()
        fs.forces(s, 'EOS', p)
        self.assertAlmostEqual(s.amsig*s.hbc, p.xms*1000)
        self.assertAlmostEqual(s.amdel*s.hbc, p.xmd*1000)
        self.assertEqual(s.gsig, p.g_s)
        self.assertAlmostEqual(s.z_bs, p.b_s*939/s.hbc)
        self.assertEqual(s.z_ds, p.c_ds)
        self.assertEqual((s.icm,s.icou,s.ipc), (2,2,0))
        with self.assertRaisesRegex(ValueError, 'requires eos_parameters'):
            fs.forces(s, 'EOS')
        with self.assertRaisesRegex(ValueError, 'Missing EOS parameters'):
            fs.forces(s, 'EOS', {})

    def test_minimal_basis_and_neutron_only_setup(self):
        s = fixture(n0f=1, n0b=0)
        fs.forces(s, 'DD-ME2')
        s.itx = 1
        fs.gaush(s)
        fs.prep(s)
        fs.start(s)
        fs.base(s)
        fs.gaupol(s)
        fs.singf(s)
        fs.greecou(s)
        fs.greemes(s)
        self.assertEqual(s.icou, 0)
        self.assertEqual(s.no, 1)
        np.testing.assert_array_equal(s.vps[:,1], s.vps[:,2])
        np.testing.assert_array_equal(s.vms[:,1], s.vms[:,2])
        self.assertTrue(np.all(np.isfinite(s.gg)))


if __name__ == '__main__':
    unittest.main()
