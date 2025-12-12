import pandas as pd
import numpy as np

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

