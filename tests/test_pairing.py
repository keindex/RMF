import unittest


class PairingTests(unittest.TestCase):
    def test_ground_state_transformation(self):
        from crmf.finite_nucleus.pairing import talmos
        self.assertAlmostEqual(talmos(0, 0, 0, 0, 0, 0, 0, 0, 0), 1.0, places=12)

    def test_forbidden_energy_and_triangle(self):
        from crmf.finite_nucleus.pairing import talmos
        self.assertEqual(talmos(0, 0, 0, 0, 1, 0, 0, 0, 0), 0.0)
        self.assertEqual(talmos(0, 0, 0, 0, 0, 0, 0, 0, 1), 0.0)

    def test_six_j(self):
        from crmf.finite_nucleus.pairing import racah0
        self.assertAlmostEqual(racah0(1, 1, 1, 1, 1, 1), 1/6, places=12)


if __name__ == '__main__':
    unittest.main()
