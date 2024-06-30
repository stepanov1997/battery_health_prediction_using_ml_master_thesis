import time

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, precision_score, recall_score, \
    accuracy_score, roc_auc_score, multilabel_confusion_matrix, classification_report, mean_squared_log_error, \
    mean_absolute_error
import numpy as np


class ModelTrainer:
    """
    The ModelTrainer class is responsible for managing the training of machine learning models. It handles
    the training process using various estimators provided in the estimators_data, evaluates model performance,
    and serializes the results and trained models for future use.

    This class is designed to abstract the complexities of training and model selection, providing a streamlined
    interface for training multiple models and selecting the best performing one based on specified metrics.

    Attributes: serialization_util (SerializationUtil): An instance of the SerializationUtil class used for handling
    serialization tasks. estimators_data (list): A list containing data about various machine learning estimators and
    their grid search parameters.

    Methods: train_model_with_specified_estimator: Trains a model using a specified estimator and hyperparameter
    grid. compute_error_metrics_on_test_set: Computes error metrics for the model on the test dataset.
    main_model_training_process: Coordinates the process of training multiple models and selecting the best
    performing one.
    """

    def __init__(self, serialization_util, estimators_data_retriever):
        """
        Initializes the ModelTrainer class.

        :param serialization_util: Utility class for handling serialization tasks.
        :type serialization_util: SerializationUtil
        :param estimators_data_retriever: Data retriever containing various machine learning estimators and their configurations.
        :type estimators_data_retriever: function
        """
        self.serialization_util = serialization_util
        self.estimators_data_retriever = estimators_data_retriever

    def train_model_with_specified_estimator(self, X_train, y_train, estimator_tuple, grid_params,
                                             regression_or_classification):
        """
        Trains a model using the specified estimator and hyperparameter grid.

        :param X_train: The training feature data.
        :type X_train: DataFrame
        :param y_train: The training target data.
        :type y_train: Series or ndarray
        :param estimator_tuple: A tuple containing the estimator's name and the estimator object.
        :type estimator_tuple: tuple
        :param grid_params: Parameters for the grid search in hyperparameter tuning.
        :type grid_params: dict

        :return: A GridSearchCV object that has been fitted to the training data.
        :rtype: GridSearchCV
        """
        groups = X_train['battery_filename']

        # Creating a pipeline that includes preprocessing steps and the estimator
        pipeline = Pipeline([
            ('small_modifier',
             FunctionTransformer(func=lambda x: x.drop(['battery_filename'], axis=1).astype(np.float64))),
            ('scaler', StandardScaler()),
            estimator_tuple
        ])

        # Setting up a GroupKFold for cross-validation and initializing a GridSearchCV for hyperparameter tuning
        group_kfold = GroupKFold(n_splits=5)
        grid_search = GridSearchCV(pipeline, grid_params, verbose=10, cv=group_kfold,
                                   scoring='neg_mean_squared_error' if regression_or_classification == 'regression' else 'accuracy',
                                   n_jobs=-1)

        grid_search.fit(X_train, y_train, groups=groups)

        return grid_search

    def compute_error_metrics_on_test_set(self, grid_search, X_test, y_test, classification_or_regression):
        """
        Computes error metrics for the model on the test dataset.

        :param grid_search: A trained GridSearchCV object.
        :type grid_search: GridSearchCV
        :param X_test: Testing feature data.
        :type X_test: DataFrame
        :param y_test: Testing target data.
        :type y_test: Series or ndarray

        :return: Mean Squared Error (MSE) and R-squared (R2) score of the model on the test set.
        :rtype: tuple
        """
        y_pred = grid_search.predict(X_test)

        if classification_or_regression == 'classification':
            accuracy = None
            f1 = None
            precision = None
            recall = None
            roc_auc = None
            conf_matrix_dict = None
            try:
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro')
                precision = precision_score(y_test, y_pred, average='macro')
                recall = recall_score(y_test, y_pred, average='macro')
                roc_auc = roc_auc_score(y_test, y_pred, average='macro', multi_class='ovo')
                conf_matrix = multilabel_confusion_matrix(y_test, y_pred)
                conf_matrix_dict = {
                    elem: conf_matrix[index].tolist() for index, elem in
                    enumerate(['expired', 'short_lifespan', 'medium_lifespan', 'long_lifespan', 'very_long_lifespan'])
                }
            except ValueError as ex:
                print(str(ex))
            return accuracy, f1, precision, recall, roc_auc, conf_matrix_dict
        elif classification_or_regression == 'regression':
            mse = round(mean_squared_error(y_test, y_pred), 8)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mape = np.mean(np.abs(y_test - y_pred) / y_test) * 100
            msle = mean_squared_log_error(y_test, y_pred)
            return mse, r2, mae, mape, msle
        return None

    def main_model_training_process(self, train_folder, X_train, y_train, X_test, y_test):
        """
        The main method for training models, evaluating their performance, and determining the best model.

        :param train_folder: Path to the folder where training results and models will be saved.
        :type train_folder: str
        :param X_train: Training feature data.
        :type X_train: DataFrame
        :param y_train: Training target data.
        :type y_train: Series or ndarray
        :param X_test: Testing feature data.
        :type X_test: DataFrame
        :param y_test: Testing target data.
        :type y_test: Series or ndarray

        :return: A tuple containing the compiled results, the name of the best model, the best model itself,
                 the best MSE, and the best R2 score. :rtype: tuple
        """
        input_shape = X_train.shape[1] - 1
        estimators_data = self.estimators_data_retriever(input_shape)

        results = {}

        # Iterating through each estimator and its grid parameters to train and evaluate models
        estimators = [(estimator_dict['estimator'], estimator_dict['grid_param']) for estimator_dict in estimators_data]
        for estimator, grid_param in estimators:
            if isinstance(estimator[1], ClassifierMixin):
                self.process_classification_results(estimator, X_train, y_train, X_test, y_test, grid_param, results,
                                                    train_folder)
            elif isinstance(estimator[1], RegressorMixin):
                self.process_regression_results(estimator, X_train, y_train, X_test, y_test, grid_param, results,
                                                train_folder)

        return results

    def process_regression_results(self, estimator, X_train, y_train, X_test, y_test, grid_param, results,
                                   train_folder):
        fit_start_time = time.perf_counter()
        grid_search = self.train_model_with_specified_estimator(X_train, y_train, estimator, grid_param, 'regression')
        fit_duration = time.perf_counter() - fit_start_time

        prediction_start_time = time.perf_counter()
        mse, r2, mae, mape, msle = self.compute_error_metrics_on_test_set(grid_search, X_test, y_test,
                                                                          'regression')
        prediction_duration = time.perf_counter() - prediction_start_time

        best_local_estimator = grid_search.best_estimator_
        estimator_name = estimator[0]

        # Gathering statistics for each trained model
        statistics = {
            'name': estimator_name,
            'best_params': grid_search.best_params_,
            'fit_duration': fit_duration,
            'prediction_duration': prediction_duration,
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'msle': msle
        }

        # Saving the trained model and its statistics
        self.serialization_util.save_trained_estimator(train_folder, estimator_name, best_local_estimator,
                                                       statistics)

        results[estimator_name] = statistics

    def process_classification_results(self, estimator, X_train, y_train, X_test, y_test, grid_param, results,
                                       train_folder):
        fit_start_time = time.perf_counter()
        grid_search = self.train_model_with_specified_estimator(X_train, y_train, estimator, grid_param,
                                                                'classification')
        fit_duration = time.perf_counter() - fit_start_time

        prediction_start_time = time.perf_counter()
        accuracy, f1, precision, recall, roc_auc, conf_matrix = self.compute_error_metrics_on_test_set(grid_search,
                                                                                                       X_test, y_test,
                                                                                                       'classification')
        prediction_duration = time.perf_counter() - prediction_start_time

        best_local_estimator = grid_search.best_estimator_
        estimator_name = estimator[0]

        # Gathering statistics for each trained model
        statistics = {
            'name': estimator_name,
            'best_params': grid_search.best_params_,
            'fit_duration': fit_duration,
            'prediction_duration': prediction_duration,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'conf_matrix': conf_matrix
        }

        # Saving the trained model and its statistics
        self.serialization_util.save_trained_estimator(train_folder, estimator_name, best_local_estimator,
                                                       statistics)

        results[estimator_name] = statistics
