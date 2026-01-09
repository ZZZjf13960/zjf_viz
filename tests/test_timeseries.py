import unittest
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import zjf_viz as zviz

class TestTimeseries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create dummy data for time series
        np.random.seed(42)
        n_samples = 100
        n_channels = 5
        cls.data_array = np.random.randn(n_samples, n_channels)
        cls.data_df = pd.DataFrame(
            cls.data_array,
            columns=[f"Channel_{i}" for i in range(n_channels)]
        )
        zviz.set_theme()

    def tearDown(self):
        plt.close('all')
        if os.path.exists('test_ts_plot.png'):
            os.remove('test_ts_plot.png')

    def test_stacked_signals_array(self):
        ax = zviz.stacked_signals(
            self.data_array,
            sampling_rate=100,
            title="Test Array",
            save_path='test_ts_plot.png'
        )
        self.assertIsNotNone(ax)
        self.assertTrue(os.path.exists('test_ts_plot.png'))
        # Check title
        self.assertEqual(ax.get_title(), "Test Array")
        # Check number of lines (should be equal to n_channels)
        self.assertEqual(len(ax.lines), self.data_array.shape[1])

    def test_stacked_signals_df(self):
        ax = zviz.stacked_signals(
            self.data_df,
            title="Test DataFrame"
        )
        self.assertIsNotNone(ax)
        # Check y-tick labels
        yticks = [item.get_text() for item in ax.get_yticklabels()]
        self.assertEqual(yticks, self.data_df.columns.tolist())

    def test_stacked_signals_custom_offset(self):
        offset = 10
        ax = zviz.stacked_signals(
            self.data_array,
            offset=offset
        )
        # Check if ticks are spaced by offset
        yticks = ax.get_yticks()
        # The spacing should be equal to offset (reversed because we plot top down)
        # y_pos = (n_channels - 1 - i) * offset
        # So diffs should be -offset
        diffs = np.diff(yticks)
        # Since yticks might be sorted by matplotlib, let's check unique diffs
        unique_diffs = np.unique(np.round(np.abs(diffs), 5))
        self.assertTrue(np.allclose(unique_diffs, [offset]))

    def test_stacked_signals_labels(self):
        labels = ["A", "B", "C", "D", "E"]
        ax = zviz.stacked_signals(
            self.data_array,
            labels=labels
        )
        yticks = [item.get_text() for item in ax.get_yticklabels()]
        self.assertEqual(yticks, labels)

    def test_time_frequency(self):
        # Generate a chirp signal
        fs = 1000
        t = np.linspace(0, 1, fs)
        # Chirp from 10Hz to 100Hz
        signal = np.sin(2 * np.pi * 10 * t + 2 * np.pi * 90 * t**2 / 2)

        ax = zviz.time_frequency(signal, fs, title="Test Spectrogram")
        self.assertIsNotNone(ax)

    def test_psd(self):
        fs = 1000
        t = np.linspace(0, 1, fs)
        signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

        ax = zviz.psd(signal, fs, title="Test PSD")
        self.assertIsNotNone(ax)

    def test_butterfly(self):
        ax = zviz.butterfly(self.data_array, title="Butterfly Plot")
        self.assertIsNotNone(ax)

    def test_erp_image(self):
        # Epochs x Time
        data = np.random.randn(20, 100)
        ax = zviz.erp_image(data, sampling_rate=100, title="ERP Image")
        self.assertIsNotNone(ax)

if __name__ == '__main__':
    unittest.main()
