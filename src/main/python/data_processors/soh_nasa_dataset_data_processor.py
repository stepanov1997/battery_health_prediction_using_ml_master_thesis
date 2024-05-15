import os
from typing import Tuple

import pandas as pd
import scipy.io
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from functools import lru_cache
import numpy as np

from src.main.python.data_processors.data_processor import DataProcessor


class SohNasaDatasetDataProcessor(DataProcessor):
    """
    The NASA dataset implementation
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
            ('prefit_preprocessing', FunctionTransformer(func=self.__preprocess_data_before_fitting))
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
            if filename.startswith('B00') and filename.endswith('.mat')
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
        battery_df = pd.DataFrame(SohNasaDatasetDataProcessor.__read_battery_data(file))
        battery_df['battery_filename'] = file

        # Extracting nested data from the battery DataFrame
        first_level_data = battery_df['cycle'].apply(pd.Series)
        second_level_data = first_level_data['data'].apply(pd.Series)

        # Combining and cleaning data
        battery_df = battery_df.join(first_level_data) \
            .join(second_level_data)
        battery_df = battery_df[battery_df['Capacity'].notna()]
        return battery_df.drop(['cycle', 'data', 'type'], axis=1) \
            .dropna(axis=1, how='all') \
            .reset_index(drop=True)

    @staticmethod
    def read_and_parse_multiple_files(files):
        """
        Reads and parses multiple battery data files, combining them into a single DataFrame.

        :param files: A list of file paths to read and parse.
        :type files: list
        :return: A DataFrame containing combined data from all files.
        :rtype: pd.DataFrame
        """
        battery_dfs = [SohNasaDatasetDataProcessor.__parse_battery_data(file) for file in files]
        return pd.concat(battery_dfs, ignore_index=True)

    @staticmethod
    def __read_battery_data(file):
        """
        Reads battery data from a .mat file and returns it as a dictionary.

        :param file: The path of the .mat file to read.
        :type file: str
        :return: A dictionary containing the battery data.
        :rtype: dict
        """
        battery_name = os.path.splitext(os.path.basename(file))[0]
        battery_data = scipy.io.loadmat(file, simplify_cells=True)
        return battery_data[battery_name]

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
        for col in ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time']:
            df[col] = df[col].transform(np.array)

        df = df.drop('time', axis=1)

        def drop_empty_rows(df):
            for col in df.columns:
                df = df[df[col].apply(lambda x: not (isinstance(x, np.ndarray) and len(x) == 0))]
            return df

        # Izbacivanje redova koji sadrže prazne nizove
        df = drop_empty_rows(df)

        # Pronalaženje maksimalne dužine niza u kolonama koje sadrže nizove
        max_len = max(
            max(len(arr) if isinstance(arr, np.ndarray) else 1 for arr in df[col])
            for col in df.columns
        )

        # Funkcija za popunjavanje nizova do maksimalne dužine
        def interpolate_array(arr, max_len):
            if isinstance(arr, np.ndarray):
                x = np.linspace(0, 1, len(arr))
                f = interp1d(x, arr, kind='linear', fill_value="extrapolate")
                x_new = np.linspace(0, 1, max_len)
                return f(x_new)
            else:
                return np.array([arr] * max_len)

        y = pd.to_numeric(df['Capacity'] / 2, errors='coerce').apply(lambda health: 1 if health >= 1 else health)
        X = df.drop(['Capacity'], axis=1).drop(y[y.isna()].index).dropna()

        # Popunjavanje nizova u DataFrame-u
        for col in X.columns:
            X[col] = X[col].apply(lambda x: interpolate_array(x, max_len))

        # Kombinovanje kolona u trodimenzionalni tensor
        # Dimenzije će biti (num_samples, num_features, max_len)
        X = np.stack([np.stack(X[col].values) for col in X.columns], axis=1)

        # Preparing the target variable 'y' and feature set 'X'
        y = y.drop(y[y.isna()].index)

        return X, y