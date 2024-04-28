"""Recode ACS PUMS data to match the format of the model-generated data.
"""

import logging
import numpy as np
import pandas as pd
from folktables import ACSDataSource


def PUMS_to_model_generated(dataset):
    """ Recode the PUMS data such that it has identical format as the model-generated datasets """

    default_var_of_interest = [
        'SEX', 'AGEP', 'HISP', 'RAC1P', 'NATIVITY', 'CIT',
        'SCH', 'SCHL', 'LANX', 'ENG',
        'HICOV', 'DEAR', 'DEYE',
        'MAR', 'FER', 'GCL',
        'MIL',
        'WRK', 'ESR', 'JWTRNS', 'WKL', 'WKWN', 'WKHP', 'COW', 'PINCP', 'ST', 'PWGTP']

    # Check that all variables of interest exist in dataset
    var_of_interest = [var for var in default_var_of_interest if var in dataset.columns]
    if len(var_of_interest) != len(default_var_of_interest):
        logging.error(
            f"Some variables are missing in the dataset. Missing variables: "
            f"{set(default_var_of_interest) - set(var_of_interest)}"
        )

    # Keep only the variables of interest
    dataset = dataset[var_of_interest]

    def discretize_col(col_name, bins, labels, new_col_name=None):
        if col_name not in dataset.columns:
            logging.warning(f"Column {col_name} not found in dataset")
        else:
            dataset[new_col_name or col_name] = pd.cut(dataset[col_name], bins, labels=labels)

    # Recode relevant variables
    discretize_col("AGEP", [-1, 4, 15, 30, 40, 50, 64, 99], labels=np.arange(7) + 1, new_col_name="AGER")
    discretize_col("HISP", [-1, 1, 24], labels=[2, 1], new_col_name="HISPR")
    discretize_col("RAC1P", [-1, 1, 2, 5, 6, 8, 9], labels=np.arange(6) + 1, new_col_name="RAC1PR")
    discretize_col("SCHL", [-1, 1, 15, 17, 21, 24], labels=np.arange(5) + 1, new_col_name="SCHLR")
    discretize_col("WKHP", [-1, 9, 19, 34, 44, 59, 98], labels=np.arange(6) + 1, new_col_name="WKHPR")
    dataset['COWR'] = dataset['COW']
    discretize_col("PINCP", [-19999, 0, 12490, 52000, 120000, 4209995], labels=np.arange(5) + 1, new_col_name="PINCPR")
    discretize_col("WKWN", [-1, 13, 26, 39, 47, 52], labels=np.arange(5) + 1)

    dataset.drop(['AGEP', 'HISP', 'RAC1P', 'SCHL', 'WKHP', 'COW', 'PINCP'], axis=1, inplace=True)

    # Now process the nans
    below_5yo = dataset['AGER'] == 1
    dataset.loc[below_5yo, 'SCH'] = np.nan
    dataset.loc[below_5yo, 'SCHLR'] = np.nan
    dataset.loc[below_5yo, 'LANX'] = np.nan

    not_speak_other_language = dataset['LANX'] == 2
    dataset.loc[not_speak_other_language, 'ENG'] = np.nan

    below_15yo = dataset['AGER'] <= 2
    dataset.loc[below_15yo, 'MAR'] = np.nan
    dataset.loc[below_15yo, 'FER'] = np.nan

    above_15yo = dataset['AGER'] >= 3
    dataset.loc[above_15yo, 'MIL'].replace(to_replace=np.nan, value=4, inplace=True)

    return dataset


def download_and_recode_PUMS(save_path, cache_dir='data', survey_year='2019'):
    print('Loading ACS...')
    data_source = ACSDataSource(
        survey_year=survey_year,
        horizon='1-Year', survey='person',
        root_dir=cache_dir,
    )
    data = data_source.get_data(download=True)

    print('Converting...')
    data = PUMS_to_model_generated(data)

    data.to_csv(save_path, index=False)

    print('Done')


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--cache-dir', type=str, default='data')
    parser.add_argument('--survey-year', type=int, default=2019)
    args = parser.parse_args()

    download_and_recode_PUMS(
        Path(args.save_dir) / f'pums_{args.survey_year}.csv',
        cache_dir=args.cache_dir,
        survey_year=args.survey_year,
    )
