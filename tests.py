import unittest
import pandas as pd
import numpy as np # Added for np.testing.assert_allclose
from main import load_gdp_data, prepare_training_data, train_synthetic_control_model, generate_synthetic_control


class TestDataProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load data once for all tests."""
        cls.gdp_file = "gdp-per-capita-maddison.csv"
        try:
            cls.gdp_df = load_gdp_data(cls.gdp_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load data for tests: {e}")

    def test_load_gdp_data(self):
        """
        Tests if the GDP data is loaded into a non-empty DataFrame.
        """
        self.assertIsNotNone(self.gdp_df)
        self.assertIsInstance(self.gdp_df, pd.DataFrame)
        self.assertFalse(self.gdp_df.empty)
        required_columns = ['Entity', 'Code', 'Year', 'GDPPerCapita']
        for col in required_columns:
            self.assertIn(col, self.gdp_df.columns)

    def test_prepare_training_data(self):
        """
        Tests the data preparation function with Italy as the treated unit.
        """
        start_year = 1960
        treatment_year = 1990
        # User-specified change: Italy as treated, Germany/France/Spain as controls
        output_country = "Italy"
        input_countries = ["Germany", "France", "Spain"]
        
        try:
            X_train, y_train, years_train = prepare_training_data(
                self.gdp_df, start_year, treatment_year, input_countries, output_country
            )
            
            # Pre-treatment period is 30 years (1960 to 1989)
            expected_rows = 30
            self.assertEqual(len(years_train), expected_rows)
            self.assertEqual(years_train[0], start_year)
            self.assertEqual(years_train[-1], treatment_year - 1)
            
            # X_train should have 3 columns (for the 3 input countries)
            self.assertEqual(X_train.shape, (expected_rows, 3))
            
            # y_train should be a vector with 30 elements
            self.assertEqual(y_train.shape, (expected_rows,))
            
        except ValueError as e:
            # This test might fail if the data for these specific countries/years isn't in the CSV.
            # We fail the test explicitly if a ValueError is raised, as it indicates a data preparation issue.
            self.fail(f"prepare_training_data raised a ValueError unexpectedly: {e}")

    def test_train_synthetic_control_model(self):
        """
        Tests the synthetic control model training with a simple, predictable dataset.
        """
        # Create a simple dataset where y is the average of X's columns
        # Expected weights: [0.5, 0.5]
        X_test = np.array([
            [10, 20],
            [12, 22],
            [14, 24]
        ])
        y_test = np.array([15, 17, 19]) # (10+20)/2 = 15, (12+22)/2 = 17, (14+24)/2 = 19

        expected_weights = np.array([0.5, 0.5])
        
        try:
            actual_weights = train_synthetic_control_model(X_test, y_test)
            
            # Check if weights are close to expected (floating point comparison)
            np.testing.assert_allclose(actual_weights, expected_weights, atol=1e-6)
            
            # Verify sum of weights is 1
            self.assertAlmostEqual(np.sum(actual_weights), 1.0)
            
            # Verify weights are non-negative
            self.assertTrue(np.all(actual_weights >= 0))

        except Exception as e:
            self.fail(f"train_synthetic_control_model raised an exception unexpectedly: {e}")

    def test_generate_synthetic_control(self):
        """
        Tests the synthetic data generation with known weights and data.
        """
        # Create a simple DataFrame for testing
        test_data = {
            'Year': [2000, 2000, 2001, 2001],
            'Entity': ['A', 'B', 'A', 'B'],
            'GDPPerCapita': [100, 200, 110, 210]
        }
        test_df = pd.DataFrame(test_data)
        
        start_year = 2000
        end_year = 2001
        input_countries = ['A', 'B']
        weights = np.array([0.4, 0.6])

        # Expected output:
        # Year 2000: 100*0.4 + 200*0.6 = 40 + 120 = 160
        # Year 2001: 110*0.4 + 210*0.6 = 44 + 126 = 170
        expected_synthetic_y = np.array([160, 170])
        expected_years = [2000, 2001]
        
        try:
            synthetic_y, years = generate_synthetic_control(
                test_df, start_year, end_year, input_countries, weights
            )
            
            np.testing.assert_allclose(synthetic_y, expected_synthetic_y)
            self.assertEqual(years, expected_years)

        except Exception as e:
            self.fail(f"generate_synthetic_control raised an exception unexpectedly: {e}")


if __name__ == '__main__':
    unittest.main()
