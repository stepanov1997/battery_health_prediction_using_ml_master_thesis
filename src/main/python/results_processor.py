import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


class ResultsProcessor:
    """
    The ResultsProcessor class is designed for handling the post-processing of machine learning model training
    results. It provides functionalities to save training results, generate and save performance charts, and manage
    the results' directory.

    This class plays a critical role in visualizing model performance metrics and organizing the output of the
    training process, making it easier to analyze and compare different models.

    Attributes:
        directory (str): The directory where the results and charts will be saved.
        results_df (DataFrame): A DataFrame representing the results of the model training process.

    Methods: save_training_results: Converts the training results into JSON format and saves them in the specified
    directory. generate_and_save_performance_charts: Generates and saves performance charts, including fit duration,
    prediction duration, MSE, and R2. rename_folder_to_contain_best_result: Renames the results directory to include
    the name and MSE of the best-performing model. setup_result_folders: Static method to set up folders for storing
    results, including a main folder and a timestamped subfolder.
    """

    def __init__(self):
        """
        Initializes the ResultsProcessor class.
        """

    def save_training_results(self, directory, results):
        """
        Converts the training results into JSON format and saves them in the specified directory.
        
        :param directory: The directory where the results and charts will be saved.
        :type directory: str
        :param results: The results of the model training process.
        :type results: dict
        """

        results = (pd.DataFrame.from_dict(results, orient='index')
                   .drop(['name'], axis=1))

        if 'mse' in results.columns:
            results = results.sort_values(by=['mse'])
        elif 'accuracy' in results.columns:
            results = results.sort_values(by=['accuracy'])

        results = results.to_json()
        result_json = json.loads(results)

        with open(os.path.join(directory, "results.json"), "w") as f:
            json.dump(result_json, f, indent=4)

    def generate_and_save_performance_charts(self, directory, results):
        """
        Generates and saves performance charts for the models. Charts include fit duration,
        prediction duration, mean squared error (MSE), and R-squared (R2) values.
        
        :param directory: The directory where the results and charts will be saved.
        :type directory: str
        :param results: The results of the model training process.
        :type results: dict
        """
        results_df = (pd.DataFrame.from_dict(results, orient='index')
                   .drop(['name'], axis=1))

        if 'mse' in results_df.columns:
            results_df = results_df.sort_values(by=['mse'])
        elif 'accuracy' in results_df.columns:
            results_df = results_df.sort_values(by=['accuracy'])

        df_fit_duration = pd.DataFrame(
            {'model': results_df.index, 'fit_duration': results_df['fit_duration']})
        df_prediction_duration = pd.DataFrame(
            {'model': results_df.index, 'prediction_duration': results_df['prediction_duration']})

        if 'mse' in results_df.columns:
            df_mse = pd.DataFrame({'model': results_df.index, 'mse': results_df['mse']})
            df_r2 = pd.DataFrame({'model': results_df.index, 'r2': results_df['r2']})
            df_mae = pd.DataFrame({'model': results_df.index, 'mae': results_df['mae']})
            df_mape = pd.DataFrame({'model': results_df.index, 'mape': results_df['mape']})
            df_msle = pd.DataFrame({'model': results_df.index, 'msle': results_df['msle']})

        elif 'accuracy' in results_df.columns:
            df_accuracy = pd.DataFrame({'model': results_df.index, 'accuracy': results_df['accuracy']})
            df_f1 = pd.DataFrame({'model': results_df.index, 'f1': results_df['f1']})
            df_precision = pd.DataFrame({'model': results_df.index, 'precision': results_df['precision']})
            df_recall = pd.DataFrame({'model': results_df.index, 'recall': results_df['recall']})
            df_roc_auc = pd.DataFrame({'model': results_df.index, 'roc_auc': results_df['roc_auc']})


        # Internal function to create and save charts for different metrics
        def plot_and_save_chart(df, y_label, file_name):
            plt.figure(figsize=(10, 6))
            plt.plot(df['model'], df.iloc[:, 1], marker='o')
            plt.xticks(rotation=45)
            plt.ylabel(y_label)
            plt.title(f'{y_label} by model')

            plt.savefig(os.path.join(directory, f"{file_name}.png"))
            plt.show()

        # Generating and saving charts for various performance metrics
        plot_and_save_chart(df_fit_duration, 'Fit Duration', 'fit_duration_chart')
        plot_and_save_chart(df_prediction_duration, 'Prediction Duration', 'prediction_duration_chart')
        if 'mse' in results_df.columns:
            plot_and_save_chart(df_mse, 'MSE', 'mse_chart')
            plot_and_save_chart(df_r2, 'R2', 'r2_chart')
            plot_and_save_chart(df_mae, 'R2', 'mae_chart')
            plot_and_save_chart(df_mape, 'R2', 'mape_chart')
            plot_and_save_chart(df_msle, 'R2', 'msle_chart')

        elif 'accuracy' in results_df.columns:
            plot_and_save_chart(df_accuracy, 'Accuracy', 'accuracy_chart')
            plot_and_save_chart(df_f1, 'F1', 'f1_chart')
            plot_and_save_chart(df_precision, 'PRECISION', 'precision_chart')
            plot_and_save_chart(df_recall, 'RECALL', 'recall_chart')
            plot_and_save_chart(df_roc_auc, 'ROC AUC', 'roc_auc_chart')


    def rename_folder_to_contain_best_result(self, directory, best_global_estimator_name, best_result):
        """
        Renames the results directory to include the name and MSE of the best-performing model.

        :param directory: The directory where the results will be saved.
        :type directory: str
        :param best_global_estimator_name: The name of the best-performing model.
        :type best_global_estimator_name: str
        :param best_global_mse: The MSE of the best-performing model.
        :type best_global_mse: float
        """
        os.rename(directory, f"{directory}-{best_global_estimator_name}-{best_result}")

    def setup_result_folders(self, directory):
        """
        Sets up folders for storing results. This includes creating a main folder for results
        and a timestamped subfolder for the current run.

        :param directory: The directory where the results will be saved.
        :type directory: str
        :return: The paths to the main result folder and the timestamped subfolder.
        :rtype: tuple
        """
        main_folder = os.path.join(directory, 'results')
        if not os.path.exists(main_folder):
            os.makedirs(main_folder)

        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        subfolder = os.path.join(main_folder, timestamp)
        os.makedirs(subfolder)

        return subfolder, timestamp
