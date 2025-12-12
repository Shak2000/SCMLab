import pandas as pd
import numpy as np
from scipy.optimize import minimize


def load_gdp_data(file_path: str) -> pd.DataFrame:
    """
    Loads GDP per capita data from a CSV file into a Pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        # Ensure required columns exist
        required_columns = ['Entity', 'Code', 'Year', 'GDPPerCapita']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV is missing one or more required columns: {required_columns}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error loading GDP data: {e}")
        raise


def prepare_training_data(
    df: pd.DataFrame,
    start_year: int,
    treatment_year: int,
    input_countries: list[str],
    output_country: str
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Prepares training data for the synthetic control model.

    Args:
        df: DataFrame containing GDP per capita data.
        start_year: The starting year for the pre-treatment period.
        treatment_year: The year the intervention occurred (exclusive).
        input_countries: A list of country names to use as input (control) units.
        output_country: The name of the treated country (output unit).

    Returns:
        A tuple containing:
        - X: NumPy array of input country GDPs for the pre-treatment period.
        - y: NumPy array of output country GDP for the pre-treatment period.
        - years: List of years included in the training data.
    """
    pre_treatment_years = range(start_year, treatment_year)

    # Filter data for relevant years and countries
    countries_of_interest = input_countries + [output_country]
    filtered_df = df[
        (df['Year'].isin(pre_treatment_years)) &
        (df['Entity'].isin(countries_of_interest))
    ].pivot_table(index='Year', columns='Entity', values='GDPPerCapita')

    # Ensure all required countries and years are present
    missing_countries = [c for c in countries_of_interest if c not in filtered_df.columns]
    if missing_countries:
        raise ValueError(f"Missing data for countries in the specified pre-treatment period: {missing_countries}")
    
    missing_years = [y for y in pre_treatment_years if y not in filtered_df.index]
    if missing_years:
        raise ValueError(f"Missing data for years in the specified pre-treatment period: {missing_years}")

    # Separate features (X) and target (y)
    X = filtered_df[input_countries].values
    y = filtered_df[output_country].values
    
    return X, y, list(filtered_df.index)


def train_synthetic_control_model(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Trains the synthetic control model to find optimal weights.

    Args:
        X: NumPy array of input country GDPs for the pre-treatment period
           (n_years, n_input_countries).
        y: NumPy array of output country GDP for the pre-treatment period
           (n_years,).

    Returns:
        NumPy array of optimal weights (n_input_countries,).
    """
    n_input_countries = X.shape[1]

    # Objective function: Minimize the squared difference between X @ w and y
    # f(w) = sum((X @ w - y)**2)
    def objective(weights):
        return np.sum((X @ weights - y)**2)

    # Constraints:
    # 1. Weights must be non-negative (w_i >= 0)
    # 2. Weights must sum to 1 (sum(w_i) = 1)
    
    # Bounds for weights (non-negative)
    bounds = [(0., 1.) for _ in range(n_input_countries)] # Weights between 0 and 1

    # Linear equality constraint: sum(w_i) - 1 = 0
    # This is of the form: A_eq @ w - b_eq = 0
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Initial guess for weights (equal distribution)
    initial_weights = np.array([1. / n_input_countries] * n_input_countries)

    # Perform the optimization
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP', # Sequential Least SQuares Programming
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return result.x


def generate_synthetic_control(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    input_countries: list[str],
    weights: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    """
    Generates the synthetic control data for the entire period.

    Args:
        df: DataFrame containing GDP per capita data.
        start_year: The starting year for the entire period.
        end_year: The ending year for the entire period.
        input_countries: A list of country names used as input (control) units.
        weights: The optimal weights from the trained model.

    Returns:
        A tuple containing:
        - synthetic_y: NumPy array of the generated synthetic control data.
        - years: List of years included in the data.
    """
    full_period_years = range(start_year, end_year + 1)

    # Filter data for relevant years and countries
    filtered_df = df[
        (df['Year'].isin(full_period_years)) &
        (df['Entity'].isin(input_countries))
    ].pivot_table(index='Year', columns='Entity', values='GDPPerCapita')

    # Ensure all required countries and years are present
    missing_countries = [c for c in input_countries if c not in filtered_df.columns]
    if missing_countries:
        raise ValueError(f"Missing data for countries in the specified period: {missing_countries}")

    missing_years = [y for y in full_period_years if y not in filtered_df.index]
    if missing_years:
        raise ValueError(f"Missing data for years in the specified period: {missing_years}")
        
    # Reorder columns to match the order of weights
    filtered_df = filtered_df[input_countries]

    # Get the data for the input countries
    X_full = filtered_df.values

    # Generate the synthetic control
    synthetic_y = X_full @ weights

    return synthetic_y, list(filtered_df.index)
