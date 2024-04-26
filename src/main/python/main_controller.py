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
            # TOYOTA_DATASET_DIR: SohToyotaDatasetDataProcessor(os.path.join(data_directory, TOYOTA_DATASET_DIR)),
            TOYOTA_DATASET_DIR: RulToyotaDatasetDataProcessor(os.path.join(data_directory, TOYOTA_DATASET_DIR)),
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

            # Saving the preprocessing pipeline for future use, ensuring consistency in data processing
            self.serialization_util.save_preprocessor(results_directory, preprocessing_pipeline)

            # Main training process: training various models and finding the best performing model
            results = self.model_trainer.main_model_training_process(results_directory, X_train, y_train, X_test,
                                                                          y_test)

            results_df = pd.DataFrame(data=results).transpose()
            results_df = results_df.reset_index()
            mse, r2, regressor_name = None, None, None
            if 'mse' in results_df.columns:
                best_global_mse_row_id = results_df['mse'].astype(np.float64).idxmin()

                best_global_mse_row = results_df.iloc[best_global_mse_row_id]
                mse = best_global_mse_row['mse']
                regressor_name = best_global_mse_row['name']

            accuracy = None
            classificator_name = None
            if 'accuracy' in results_df.columns:
                best_global_accuracy_row_id = results_df['accuracy'].astype(np.float64).idxmax()
                best_global_accuracy_row = results_df.iloc[best_global_accuracy_row_id]
                accuracy = best_global_accuracy_row['accuracy']
                classificator_name = best_global_accuracy_row['name']


            best_result, name = (accuracy, classificator_name) \
                if accuracy \
                else (mse, regressor_name)

            # Processing and storing the training results, including generating and saving performance charts
            self.results_processor.save_training_results(results_directory, results)
            self.results_processor.generate_and_save_performance_charts(results_directory, results)

            # Renaming the results folder to include the name and performance of the best model
            self.results_processor.rename_folder_to_contain_best_result(results_directory, name, best_result)
