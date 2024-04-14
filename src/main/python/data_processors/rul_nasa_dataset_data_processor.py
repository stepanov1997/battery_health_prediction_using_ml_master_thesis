import os
from typing import Tuple

import pandas as pd
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from functools import lru_cache
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.main.python.data_processors.data_processor import DataProcessor


class RulNasaDatasetDataProcessor(DataProcessor):
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
            f'{root}\\{filename}'
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
        battery_df = pd.DataFrame(RulNasaDatasetDataProcessor.__read_battery_data(file))
        battery_df['battery_filename'] = file

        # Extracting nested data from the battery DataFrame
        first_level_data = battery_df['cycle'].apply(pd.Series)
        second_level_data = first_level_data['data'].apply(pd.Series)

        # Combining and cleaning data
        battery_df = battery_df.join(first_level_data) \
            .join(second_level_data)
        battery_df = battery_df[(battery_df['type'] != 'impedance')]
        return battery_df.drop(['cycle', 'data'], axis=1) \
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
        battery_dfs = [RulNasaDatasetDataProcessor.__parse_battery_data(file) for file in files]
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

        for column_name in ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load',
                            'Voltage_load', 'Current_charge', 'Voltage_charge']:
            RulNasaDatasetDataProcessor.__describe_nested_data(df, column_name)

        df['Time_max'] = df['Time'].apply(np.max)
        df = df.drop(['Time', 'time'], axis=1).round(5)

        # Label encoder
        le = LabelEncoder()
        df['battery_index'] = le.fit_transform(df['battery_filename'])

        threshold = 1.6  # 2*80%

        if 'cycle_index' not in df.columns:
            df['cycle_index'] = df.groupby('battery_index').cumcount() + 1

        discharge_cycles = df[df['type'] == 'discharge'].copy()

        threshold_rolling_median = 0.5

        discharge_cycles['Capacity'] = pd.to_numeric(discharge_cycles['Capacity'], errors='coerce')

        discharge_cycles['Capacity'] = discharge_cycles['Capacity'].mask(
            discharge_cycles['Capacity'].rolling(window=3, min_periods=1).median() - discharge_cycles[
                'Capacity'] > threshold_rolling_median,
            discharge_cycles['Capacity'].rolling(window=3, min_periods=1).median()
        )

        def rul_per_group(group):
            below_threshold_cycle = group[(group['type'] == 'discharge') & (group['Capacity'] < threshold)][
                'cycle_index'].min()
            group['RUL'] = np.nan \
                if pd.isna(below_threshold_cycle) \
                else np.maximum(below_threshold_cycle - group['cycle_index'], 0)
            return group

        df = df.groupby('battery_index').apply(rul_per_group)

        valid_indices = df.dropna(subset=['RUL'])['battery_index'].unique()
        df = df[df['battery_index'].isin(valid_indices)]

        df.reset_index(drop=True, inplace=True)

        df = df.groupby('battery_index').apply(lambda x: x.head(100))

        df.reset_index(drop=True, inplace=True)

        df = df.drop(columns=['Capacity'])

        X = df.drop(columns=['RUL'])
        y = df['RUL']

        return X, y

    @staticmethod
    def __describe_nested_data(df, column_name):
        """
        Helper method for preprocessing: computes statistical metrics for a given column in the DataFrame.

        :param df: The DataFrame to process.
        :type df: pd.DataFrame
        :param column_name: The name of the column to compute statistics for.
        :type column_name: str
        """

        df[f'{column_name}_max'] = df[column_name].apply(np.max)
        df[f'{column_name}_min'] = df[column_name].apply(np.min)
        df[f'{column_name}_avg'] = df[column_name].apply(np.average)
        df[f'{column_name}_std'] = df[column_name].apply(np.std)
        # Uncomment the following line if kurtosis is required
        # df[f'{column_name}_kurt'] = df[column_name].apply(kurtosis)
        df.drop([column_name], axis=1, inplace=True)
