import json
import os
import re
from datetime import datetime
from functools import lru_cache
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

from src.main.python.data_processors.data_processor import DataProcessor


# noinspection PyPackageRequirements
class RulToyotaDatasetDataProcessor(DataProcessor):
    """
    The Toyota dataset implementation
    """

    def __init__(self, data_directory):
        """
        Initializes the DataProcessor class.

        :param data_directory: The directory containing the data files for model training.
        :type data_directory: str
        """
        self.data_directory = data_directory

    def preprocess_data(self) -> Tuple[Pipeline, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Preprocesses the data by splitting it into training and testing sets and applying a preprocessing pipeline.

        :return: The preprocessing pipeline and the split training and testing data (X_train, y_train, X_test, y_test).
        :rtype: tuple
        """
        battery_filenames = self.__list_battery_files(self.data_directory)

        # Split battery data filenames into training and testing sets
        train, test = train_test_split(battery_filenames, test_size=0.2, random_state=42)

        # Define and fit preprocessing pipeline on training data
        preprocessing_pipeline = Pipeline([
            ('read_and_parse_files', FunctionTransformer(func=self.read_and_parse_multiple_files)),
            ('prefit_preprocessing', FunctionTransformer(func=self.__preprocess_data_before_fitting)),
            ('label_encoder', LabelEncoderPipelineFriendly(categories=['expired', 'short_lifespan', 'medium_lifespan',
                                                                       'long_lifespan', 'very_long_lifespan']))
        ])

        # Apply the pipeline to both training and test data
        X_train, y_train = preprocessing_pipeline.fit_transform(train)
        X_test, y_test = preprocessing_pipeline.transform(test)

        return preprocessing_pipeline, X_train, y_train, X_test, y_test

    @staticmethod
    def __list_battery_files(root):
        """
        Lists all battery data files (.mat files) in the given directory.

        :param root: The root directory to search for data files.
        :type root: str
        :return: A Series containing the paths of the battery data files.
        :rtype: pd.Series
        """
        return pd.Series([
            f'{root}{os.sep}{filename}'
            for root, _, filenames in os.walk(root)
            for filename in filenames
            if re.match(r'^FastCharge_\d{6}_CH\d{1,2}_structure\.json$', filename)
        ])

    # Parses battery data into a DataFrame
    @staticmethod
    @lru_cache
    def __parse_battery_data(file):
        """
        Parses an individual battery data file into a structured DataFrame.

        :param file: The path of the battery data file to parse.
        :type file: str
        :return: A DataFrame containing parsed battery data.
        :rtype: pd.DataFrame
        """

        # Parses individual battery data file into a structured DataFrame
        filename, data = RulToyotaDatasetDataProcessor.__read_battery_data(file)

        battery_data = data['summary']
        battery_data_final = {
            'battery_filename': filename
        }
        for index in battery_data.index.values:
            battery_data_final[index] = battery_data_final.get(index, {})
            index_data = battery_data.loc[index]
            for cycle_index in battery_data['cycle_index']:
                battery_data_final[index][cycle_index] = index_data[cycle_index] \
                    if isinstance(index_data, list) \
                    else index_data

        battery_data_final = pd.DataFrame(battery_data_final)

        battery_data_final['timestamp'] = battery_data_final['date_time_iso'].apply(
            lambda x: datetime.fromisoformat(x).timestamp()
        )

        battery_data_final = battery_data_final.drop('cycle_index', axis=1) \
            .drop('date_time_iso', axis=1) \
            .dropna(axis=1, how='all')

        return battery_data_final

    @staticmethod
    def read_and_parse_multiple_files(files):
        """
        Reads and parses multiple battery data files, combining them into a single DataFrame.

        :param files: A list of file paths to read and parse.
        :type files: list
        :return: A DataFrame containing combined data from all files.
        :rtype: pd.DataFrame
        """
        battery_dfs = [RulToyotaDatasetDataProcessor.__parse_battery_data(file) for file in files]
        return pd.concat(battery_dfs).reset_index()

    @staticmethod
    def __read_battery_data(file):
        """
        Reads battery data from a .mat file and returns it as a dictionary.

        :param file: The path of the .mat file to read.
        :type file: str
        :return: A dictionary containing the battery data.
        :rtype: dict
        """
        with open(file, 'r') as f:
            battery_data = pd.DataFrame(json.loads(f.read()))
        return file, battery_data

    # Preprocesses the data before fitting the models
    @staticmethod
    def __preprocess_data_before_fitting(df):
        """
        Preprocesses the raw DataFrame before fitting models by generating statistical features.

        :param df: The raw DataFrame to preprocess.
        :type df: pd.DataFrame
        :return: Processed feature set X and target variable y.
        :rtype: tuple
        """
        df = df.dropna(axis=0)

        # Label encoder
        le = LabelEncoder()
        df['battery_index'] = le.fit_transform(df['battery_filename'])

        threshold = 0.88  # 1.1*80%

        if 'cycle_index' not in df.columns:
            df['cycle_index'] = df.groupby('battery_index').cumcount() + 1

        threshold_rolling_median = 0.5

        df['discharge_capacity'] = (pd.to_numeric(df['discharge_capacity'], errors='coerce').mask(
            df['discharge_capacity'].rolling(window=3, min_periods=1).median()
            - df['discharge_capacity'] > threshold_rolling_median,
            df['discharge_capacity'].rolling(window=3, min_periods=1).median()
        ))

        def rul_per_group(group):
            below_threshold_cycle = group[group['discharge_capacity'] < threshold]['cycle_index'].min()
            group['RUL'] = 0 \
                if pd.isna(below_threshold_cycle) \
                else np.maximum(below_threshold_cycle - group['cycle_index'], 0)

            return group

        df = df.groupby('battery_index').apply(rul_per_group)

        df = df.dropna()

        valid_indices = df.dropna(subset=['RUL'])['battery_index'].unique()
        df = df[df['battery_index'].isin(valid_indices)]

        df.reset_index(drop=True, inplace=True)

        df = df[df['cycle_index'] < 51]

        df = df.drop(columns=['discharge_capacity', 'battery_index', 'cycle_index'])

        def classify_rul(rul):
            if rul == 0:
                return 'expired'
            if rul <= 100:
                return 'short_lifespan'
            if rul <= 300:
                return 'medium_lifespan'
            if rul <= 500:
                return 'long_lifespan'
            return 'very_long_lifespan'

        df['RUL'] = df['RUL'].apply(classify_rul)

        X = df.drop(columns=['RUL'])
        y = df['RUL']

        return X, y


class LabelEncoderPipelineFriendly(BaseEstimator, TransformerMixin):
    def __init__(self, categories):
        self.categories = categories
        self.encoder = OneHotEncoder(categories=[categories], dtype=np.int64, sparse_output=False)

    def fit(self, tuple, y=None):
        X, y = tuple
        y_reshaped = np.array(y).reshape(-1, 1)
        self.encoder.fit(y_reshaped)
        return self

    def transform(self, tuple):
        X, y = tuple
        y_reshaped = y.values.reshape(-1, 1)
        new_y = pd.DataFrame(data=self.encoder.transform(y_reshaped), columns=self.categories[::-1])
        return X, new_y

    def fit_transform(self, tuple, y=None, **fit_params):
        return self.fit(tuple, None).transform(tuple)

    def inverse_transform(self, y):
        return pd.Series(self.encoder.inverse_transform(y).flatten())
