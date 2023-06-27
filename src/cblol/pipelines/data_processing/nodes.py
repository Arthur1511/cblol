import pandas as pd


def _remove_na_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for teams.

    Args:
        df (pd.DataFrame): raw data

    Returns:
        Preprocessed data, without na columns.
    """   
    return df.dropna(axis=1, how='all')

def _to_category(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for teams.

    Args:
        df (pd.DataFrame): raw data

    Returns:
        Preprocessed data, with column transformed to category.
    """ 
    cat_cols = list(df.select_dtypes(include=['object']).columns.append(pd.Index(['teamname', 'patch'])))
    for c in cat_cols:
        df[c] = df[c].astype('category')

    
    return df

def _seconds_to_min(x: pd.Series) -> pd.Series:
    return (x/60).round(2)

def preprocess_teams(teams: pd.DataFrame) -> pd.DataFrame:
  
    teams = _remove_na_cols(teams)
    teams['gamelength'] = _seconds_to_min(teams['gamelength'])
    teams = _to_category(teams)


    return teams



def create_model_input_table(teams: pd.DataFrame) -> pd.DataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        Model input table.

    """
    return teams.drop(columns=["Unnamed: 0", "gameid"])
