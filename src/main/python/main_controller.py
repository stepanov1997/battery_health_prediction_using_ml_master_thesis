import os
from typing import Dict

import numpy as np
import pandas as pd

from data_processors.data_processor import DataProcessor
from model_trainer import ModelTrainer
from results_processor import ResultsProcessor
from serialization_util import SerializationUtil
from src.main.python.data_processors.rul_nasa_dataset_data_processor import RulNasaDatasetDataProcessor
from src.main.python.data_processors.rul_toyota_dataset_data_processor import RulToyotaDatasetDataProcessor
from src.main.python.data_processors.soh_nasa_dataset_data_processor import SohNasaDatasetDataProcessor
from src.main.python.data_processors.soh_nasa_randomized_data_processor import SohNasaRandomizedDataProcessor
from src.main.python.data_processors.soh_panasonic_dataset_data_processor import SohPanasonicDatasetDataProcessor
from src.main.python.data_processors.soh_toyota_dataset_data_processor import SohToyotaDatasetDataProcessor

NASA_DATASET_DIR = "NASA dataset"
PANASONIC_DATASET_DIR = "Panasonic 18650PF Data"
TOYOTA_DATASET_DIR = "Toyota"
NASA_RANDOMIZED_DATASET_DIR = "NASA randomized dataset"


def add_interpolated_rows(train_df, target_series, percentage=0.3):
    # Create copies of the dataframe and series to avoid modifying the originals
    new_train_df = train_df.copy()
    new_target_series = target_series.copy()

    # Ensure the 'battery_filename' column is treated as a string
    new_train_df['battery_filename'] = new_train_df['battery_filename'].astype(str)

    # Function to insert NaN rows for each group and interpolate values
    def insert_nans_and_interpolate(group, target_group):
        # Calculate the number of new rows to insert based on the percentage
        num_new_rows = int(len(group) * percentage)
        if num_new_rows == 0:
            return group, target_group

        # Generate random indices where new rows will be inserted
        random_indices = np.random.choice(len(group), size=num_new_rows, replace=False)
        random_indices.sort()

        # Create a new dataframe with NaN rows inserted
        expanded_indices = np.arange(len(group) + num_new_rows)
        new_group = pd.DataFrame(np.nan, index=expanded_indices, columns=group.columns)
        new_target_group = pd.Series(np.nan, index=expanded_indices)

        # Populate the new dataframe with the original data and NaN rows
        current_idx = 0
        for i in expanded_indices:
            if current_idx in random_indices:
                current_idx += 1
            if current_idx < len(group):
                new_group.iloc[i] = group.iloc[current_idx]
                new_target_group.iloc[i] = target_group.iloc[current_idx]
                current_idx += 1

        # Interpolate values for numeric columns, except 'battery_filename'
        numeric_columns = new_group.select_dtypes(include=[np.number]).columns
        new_group[numeric_columns] = new_group[numeric_columns].interpolate(method='linear')

        # Fill forward and backward for 'battery_filename' column
        new_group['battery_filename'] = new_group['battery_filename'].ffill().bfill()

        # Interpolate the target series
        new_target_group = new_target_group.interpolate(method='linear')

        return new_group, new_target_group

    # Group by 'battery_filename' and apply NaN insertion and interpolation
    grouped = new_train_df.groupby('battery_filename')
    interpolated_train_df_list = []
    interpolated_target_series_list = []

    for name, group in grouped:
        # Extract the target group corresponding to the current group
        target_group = new_target_series[group.index]
        # Insert NaNs and interpolate the current group and its target
        interpolated_group, interpolated_target_group = insert_nans_and_interpolate(group, target_group)
        interpolated_train_df_list.append(interpolated_group)
        interpolated_target_series_list.append(interpolated_target_group)

    # Concatenate all interpolated groups into a single dataframe and series
    new_train_df = pd.concat(interpolated_train_df_list).reset_index(drop=True)
    new_target_series = pd.concat(interpolated_target_series_list).reset_index(drop=True)

    return new_train_df, new_target_series


class MainController:
    """
    The MainController class serves as the central controller for the application, managing the end-to-end process
    from data preprocessing, through model training, to results processing and evaluation.

    It integrates several components including data processing, model training, and results handling,
    facilitating a streamlined workflow for machine learning model development and evaluation.

    Attributes: data_processor (DataProcessor): An instance of the DataProcessor class for handling data
    preprocessing. serialization_util (SerializationUtil): An instance of the SerializationUtil class for handling
    serialization tasks. model_trainer (ModelTrainer): An instance of the ModelTrainer class for managing the
    training of various machine learning models.
    """

    def __init__(self, data_directory, estimators_data_retriever):
        """
        Initializes the MainController class.

        :param data_directory: The directory containing the data for model training.
        :type data_directory: str
        :param estimators_data_retriever: Data retriever that containing various machine learning estimators and their
               configurations.
        :type estimators_data_retriever: function
        """

        self.data_directory = data_directory
        self.results_processor = ResultsProcessor()
        self.data_processors: Dict[str, DataProcessor] = {
            # NASA_DATASET_DIR: SohNasaDatasetDataProcessor(os.path.join(data_directory, NASA_DATASET_DIR)),
            # NASA_DATASET_DIR: RulNasaDatasetDataProcessor(os.path.join(data_directory, NASA_DATASET_DIR)),
            # PANASONIC_DATASET_DIR: SohPanasonicDatasetDataProcessor(os.path.join(data_directory, PANASONIC_DATASET_DIR)),
            TOYOTA_DATASET_DIR: SohToyotaDatasetDataProcessor(os.path.join(data_directory, TOYOTA_DATASET_DIR)),
            # TOYOTA_DATASET_DIR: RulToyotaDatasetDataProcessor(os.path.join(data_directory, TOYOTA_DATASET_DIR)),
            # NASA_RANDOMIZED_DATASET_DIR: SohNasaRandomizedDataProcessor(os.path.join(data_directory,
            #                                                                       NASA_RANDOMIZED_DATASET_DIR))

        }
        self.serialization_util = SerializationUtil()
        self.model_trainer = ModelTrainer(self.serialization_util, estimators_data_retriever)

    def run(self):
        """
        Executes the main process flow of the application. This includes setting up result folders,
        preprocessing data, training models, evaluating performance, and storing results.
        """

        for dataset_directory, data_processor in self.data_processors.items():
            print()
            print(f"###### {dataset_directory} ######")

            # Setting up a directory to store the results of the training process, including a timestamp
            results_directory, timestamp = self.results_processor.setup_result_folders(
                os.path.join(self.data_directory, dataset_directory)
            )

            # Preprocessing the data and splitting it into training and testing sets
            preprocessing_pipeline, X_train, y_train, X_test, y_test = data_processor.preprocess_data()

            X_train, y_train = add_interpolated_rows(X_train, y_train, 0.3)

            # Saving the preprocessing pipeline for future use, ensuring consistency in data processing
            self.serialization_util.save_preprocessor(results_directory, preprocessing_pipeline)

            # Main training process: training various models and finding the best performing model
            best_results = self.model_trainer.main_model_training_process(results_directory, X_train, y_train, X_test,
                                                                          y_test)
            (results, best_global_estimator_name, best_global_estimator, best_global_mse, best_global_r2) = best_results

            # Processing and storing the training results, including generating and saving performance charts
            self.results_processor.save_training_results(results_directory, results)
            self.results_processor.generate_and_save_performance_charts(results_directory, results)

            # Renaming the results folder to include the name and performance of the best model
            self.results_processor.rename_folder_to_contain_best_result(results_directory, best_global_estimator_name,
                                                                        best_global_mse)
