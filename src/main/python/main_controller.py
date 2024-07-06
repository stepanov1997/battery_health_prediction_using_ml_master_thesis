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

def add_gaussian_noise(train_df, target_df, sigma, exclude_columns, noise_percentage=0.3):
    # Calculate the number of rows to be noisy based on the noise percentage
    num_noisy_rows = int(noise_percentage * len(train_df))

    # Randomly select indices for the rows to be noisy
    noisy_indices = np.random.choice(train_df.index, size=num_noisy_rows, replace=False)

    # Create a copy of the original training data to avoid modifying it directly
    new_train_data = train_df.copy()

    # Generate Gaussian noise with mean 0 and standard deviation sigma for the selected rows and numerical columns
    noise = np.random.normal(0, sigma, size=(
        num_noisy_rows,
        train_df.drop(columns=exclude_columns).select_dtypes(include=[np.number]).shape[1]
    ))

    # Add the generated noise to the selected rows and numerical columns (excluding specified columns)
    new_train_data.loc[
        noisy_indices, train_df.select_dtypes(include=[np.number]).columns.difference(exclude_columns)] += noise

    # Extract the rows with added noise
    noisy_rows = new_train_data.loc[noisy_indices]

    # Concatenate the original training data with the noisy rows and reset the index
    augmented_train_data = pd.concat([train_df, noisy_rows]).sort_index(kind='merge').reset_index(drop=True)

    # Concatenate the original target data with the target values of the noisy rows and reset the index
    augmented_target_data = pd.concat([target_df, target_df.loc[noisy_indices]]).sort_index(kind='merge').reset_index(
        drop=True)

    # Return the augmented training and target data
    return augmented_train_data, augmented_target_data

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

            X_train, y_train = add_gaussian_noise(X_train, y_train, 0.01, ['index'], 0.3)

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
