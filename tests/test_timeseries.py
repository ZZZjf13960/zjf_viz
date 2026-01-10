import unittest
import numpy as np
import os
import matplotlib.pyplot as plt
import zjf_viz as zviz

class TestTimeseries(unittest.TestCase):
    def tearDown(self):
        plt.close('all')
        if os.path.exists('test_stacked.png'):
            os.remove('test_stacked.png')

    def test_stacked_signals(self):
        n_samples = 100
        n_channels = 3
        signals = np.random.randn(n_samples, n_channels)

        ax = zviz.stacked_signals(signals, sampling_rate=100, title="Test Stacked", save_path='test_stacked.png')
        self.assertIsNotNone(ax)
        self.assertTrue(os.path.exists('test_stacked.png'))

        # Check title
        self.assertEqual(ax.get_title(), "Test Stacked")

        # Check number of lines plotted (should be n_channels)
        self.assertEqual(len(ax.lines), n_channels)

if __name__ == '__main__':
    unittest.main()
