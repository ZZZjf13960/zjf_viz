import unittest
import numpy as np
import os
import matplotlib.pyplot as plt
import zjf_viz as zviz

class TestEEG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        zviz.set_theme()

    def tearDown(self):
        plt.close('all')
        if os.path.exists('test_topoplot.png'):
            os.remove('test_topoplot.png')

    def test_topoplot_dict(self):
        # Create dummy data for topoplot
        data = {
            'Fz': 1.0,
            'Cz': 2.0,
            'Pz': 1.5,
            'Oz': 0.5,
            'T7': -1.0,
            'T8': -1.0
        }
        ax = zviz.topoplot(data, title="Test Topoplot Dict", save_path='test_topoplot.png')
        self.assertIsNotNone(ax)
        self.assertTrue(os.path.exists('test_topoplot.png'))

    def test_topoplot_array(self):
        data = np.random.randn(5)
        names = ['Fz', 'Cz', 'Pz', 'T7', 'T8']
        ax = zviz.topoplot(data, names=names, title="Test Topoplot Array")
        self.assertIsNotNone(ax)

    def test_topoplot_invalid(self):
        data = np.random.randn(5)
        # Should raise ValueError because names not provided
        with self.assertRaises(ValueError):
            zviz.topoplot(data)

    def test_topoplot_custom_montage(self):
        # We need more points for cubic interpolation (at least 3 non-collinear, but 4 is safer for triangulation issues)
        # Or switch to linear. But let's just provide enough points.
        data = np.array([1, 2, 3, 4])
        # A triangle + center
        montage = [
            (-0.5, 0.5),
            (0.5, 0.5),
            (0, -0.5),
            (0, 0)
        ]
        ax = zviz.topoplot(data, montage=montage, names=['A', 'B', 'C', 'D'], title="Custom Montage")
        self.assertIsNotNone(ax)

    def test_plot_montage(self):
        ax = zviz.plot_montage(title="Standard Montage")
        self.assertIsNotNone(ax)

    def test_plot_connectivity(self):
        # Need more non-collinear points for interpolation in topoplot
        names = ['Fz', 'Cz', 'Pz', 'T7', 'T8']
        con = np.zeros((5, 5))
        con[0, 1] = 1 # Fz-Cz
        con[3, 4] = 0.5 # T7-T8

        ax = zviz.plot_connectivity(con, names, title="Connectivity")
        self.assertIsNotNone(ax)

if __name__ == '__main__':
    unittest.main()
