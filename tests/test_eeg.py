import unittest
import numpy as np
import os
import matplotlib.pyplot as plt
import mne
from zjf_viz.eeg import plot_topomap, plot_sensors, create_info

class TestEEG(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.n_channels = 19
        self.ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                        'T3', 'C3', 'Cz', 'C4', 'T4',
                        'T5', 'P3', 'Pz', 'P4', 'T6',
                        'O1', 'O2'] # Standard 10-20 channels
        self.data = np.random.rand(self.n_channels)
        self.info = create_info(self.ch_names)

    def tearDown(self):
        plt.close('all')
        if os.path.exists('test_topomap.png'):
            os.remove('test_topomap.png')
        if os.path.exists('test_sensors.png'):
            os.remove('test_sensors.png')

    def test_create_info(self):
        self.assertIsInstance(self.info, mne.Info)
        self.assertEqual(len(self.info['ch_names']), self.n_channels)

    def test_plot_topomap_with_names(self):
        ax = plot_topomap(self.data, ch_names=self.ch_names, title="Test Topomap")
        self.assertIsNotNone(ax)

    def test_plot_topomap_with_info(self):
        ax = plot_topomap(self.data, info=self.info, title="Test Topomap Info")
        self.assertIsNotNone(ax)

    def test_plot_sensors(self):
        ax = plot_sensors(info=self.info, title="Test Sensors")
        # plot_sensors creates a figure or axes. Our wrapper returns something (ax).
        self.assertIsNotNone(ax)

    def test_save_plot(self):
        plot_topomap(self.data, info=self.info, save_path='test_topomap.png')
        self.assertTrue(os.path.exists('test_topomap.png'))

if __name__ == '__main__':
    unittest.main()
