import unittest
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import zjf_viz as zviz

class TestZJFViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create dummy data
        np.random.seed(42)
        cls.df = pd.DataFrame({
            'x': np.random.randn(50),
            'y': np.random.randn(50) + 1,
            'category': np.random.choice(['A', 'B'], 50)
        })
        zviz.set_theme()

    def tearDown(self):
        plt.close('all')
        if os.path.exists('test_plot.png'):
            os.remove('test_plot.png')

    def test_scatter(self):
        ax = zviz.scatter(self.df, x='x', y='y', hue='category', title="Test Scatter", save_path='test_plot.png')
        self.assertIsNotNone(ax)
        self.assertTrue(os.path.exists('test_plot.png'))

    def test_line(self):
        ax = zviz.line(self.df, x='x', y='y', hue='category', title="Test Line")
        self.assertIsNotNone(ax)

    def test_bar(self):
        ax = zviz.bar(self.df, x='category', y='y', title="Test Bar")
        self.assertIsNotNone(ax)

    def test_box(self):
        ax = zviz.box(self.df, x='category', y='y', title="Test Box")
        self.assertIsNotNone(ax)

    def test_hist(self):
        ax = zviz.hist(self.df, x='x', hue='category', title="Test Hist")
        self.assertIsNotNone(ax)

if __name__ == '__main__':
    unittest.main()
