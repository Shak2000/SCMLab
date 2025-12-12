import pandas as pd
import numpy as np
from scipy.optimize import minimize


def load_data(file_path: str, metric_column: str) -> pd.DataFrame:
    """
    Loads data from a CSV file into a Pandas DataFrame and validates required columns.
    """
    try:
        df = pd.read_csv(file_path)
        required_columns = ['Entity', 'Year', metric_column]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV is missing one or more required columns: {required_columns}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")


def prepare_training_data(
    df: pd.DataFrame,
    start_year: int,
    treatment_year: int,
    input_countries: list[str],
    output_country: str,
    metric_column: str
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Prepares training data for the synthetic control model.
    """
    pre_treatment_years = range(start_year, treatment_year)

    countries_of_interest = input_countries + [output_country]
    filtered_df = df[
        (df['Year'].isin(pre_treatment_years)) &
        (df['Entity'].isin(countries_of_interest))
    ].pivot_table(index='Year', columns='Entity', values=metric_column)

    missing_countries = [c for c in countries_of_interest if c not in filtered_df.columns]
    if missing_countries:
        raise ValueError(f"Missing data for countries in the specified pre-treatment period: {missing_countries}")
    
    missing_years = [y for y in pre_treatment_years if y not in filtered_df.index]
    if missing_years:
        raise ValueError(f"Missing data for years in the specified pre-treatment period: {missing_years}")

    X = filtered_df[input_countries].values
    y = filtered_df[output_country].values
    
    return X, y, list(filtered_df.index)


def train_synthetic_control_model(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Trains the synthetic control model to find optimal weights.
    """
    n_input_countries = X.shape[1]

    def objective(weights):
        return np.sum((X @ weights - y)**2)

    bounds = [(0., 1.) for _ in range(n_input_countries)]
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    initial_weights = np.array([1. / n_input_countries] * n_input_countries)

    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
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
    weights: np.ndarray,
    metric_column: str
) -> tuple[np.ndarray, list[int]]:
    """
    Generates the synthetic control data for the entire period.
    """
    full_period_years = range(start_year, end_year + 1)

    filtered_df = df[
        (df['Year'].isin(full_period_years)) &
        (df['Entity'].isin(input_countries))
    ].pivot_table(index='Year', columns='Entity', values=metric_column)

    missing_countries = [c for c in input_countries if c not in filtered_df.columns]
    if missing_countries:
        raise ValueError(f"Missing data for countries in the specified period: {missing_countries}")

    missing_years = [y for y in full_period_years if y not in filtered_df.index]
    if missing_years:
        raise ValueError(f"Missing data for years in the specified period: {missing_years}")
        
    filtered_df = filtered_df[input_countries]
    X_full = filtered_df.values
    synthetic_y = X_full @ weights

    return synthetic_y, list(filtered_df.index)
