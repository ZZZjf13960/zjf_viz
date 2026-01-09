import unittest
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import zjf_viz as zviz
from zjf_viz.timeseries import stacked_signals

class TestTimeseries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        zviz.set_theme()

    def setUp(self):
        np.random.seed(42)
        n_samples = 100
        n_channels = 3
        self.data_array = np.random.randn(n_samples, n_channels)
        self.data_df = pd.DataFrame(
            self.data_array,
            columns=['Ch1', 'Ch2', 'Ch3'],
            index=pd.date_range('2021-01-01', periods=n_samples, freq='10ms')
        )

    def tearDown(self):
        plt.close('all')
        if os.path.exists('test_stacked.png'):
            os.remove('test_stacked.png')

    def test_stacked_signals_array(self):
        ax = stacked_signals(self.data_array, title="Test Stacked Array")
        self.assertIsNotNone(ax)

    def test_stacked_signals_df(self):
        ax = stacked_signals(self.data_df, title="Test Stacked DF")
        self.assertIsNotNone(ax)

    def test_save_plot(self):
        stacked_signals(self.data_array, save_path='test_stacked.png')
        self.assertTrue(os.path.exists('test_stacked.png'))

    def test_custom_labels(self):
        labels = ['A', 'B', 'C']
        ax = stacked_signals(self.data_array, labels=labels)
        # Check y-tick labels
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
