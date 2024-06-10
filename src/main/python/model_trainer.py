import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer


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

    def train_model_with_specified_estimator(self, X_train, y_train, estimator_tuple, grid_params):
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
        groups = X_train['battery_filename'] if isinstance(X_train, pd.DataFrame) else X_train[:, 0, 0, 0]

        def remove_filename(x):
            return x.drop(['battery_filename'], axis=1) \
                if isinstance(x, pd.DataFrame) \
                else np.delete(x, 0, axis=1)

        # Creating a pipeline that includes preprocessing steps and the estimator
        pipeline = Pipeline([
            ('small_modifier', FunctionTransformer(func=remove_filename)),
            ('scaler', NDScaler()),
            estimator_tuple
        ])

        # Setting up a GroupKFold for cross-validation and initializing a GridSearchCV for hyperparameter tuning
        group_kfold = GroupKFold(n_splits=5)
        grid_search = GridSearchCV(pipeline, grid_params, verbose=10, cv=group_kfold,
                                   scoring='neg_mean_squared_error',
                                   n_jobs=-1)
        grid_search.fit(X_train, y_train.astype(np.float64), groups=groups)

        return grid_search

    def compute_error_metrics_on_test_set(self, grid_search, X_test, y_test):
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

        mse = round(mean_squared_error(y_test, y_pred), 4)
        r2 = r2_score(y_test, y_pred)

        return mse, r2

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
        data_shape = X_train.shape
        input_shape = (data_shape[1] - 1, data_shape[2], data_shape[3])
        estimators_data = self.estimators_data_retriever(input_shape)

        best_global_estimator_name = None
        best_global_mse = 1.0
        best_global_r2 = 0.0
        best_global_estimator = None
        results = {}

        # Iterating through each estimator and its grid parameters to train and evaluate models
        estimators = [(estimator_dict['estimator'], estimator_dict['grid_param']) for estimator_dict in estimators_data]
        for estimator, grid_param in estimators:
            fit_start_time = time.perf_counter()
            grid_search = self.train_model_with_specified_estimator(X_train, y_train, estimator, grid_param)
            fit_duration = time.perf_counter() - fit_start_time

            prediction_start_time = time.perf_counter()
            best_local_mse, best_local_r2 = self.compute_error_metrics_on_test_set(grid_search, X_test, y_test)
            prediction_duration = time.perf_counter() - prediction_start_time

            best_local_estimator = grid_search.best_estimator_
            estimator_name = estimator[0]

            # Gathering statistics for each trained model
            statistics = {
                'name': estimator_name,
                'best_params': grid_search.best_params_,
                'fit_duration': fit_duration,
                'prediction_duration': prediction_duration,
                'mse': best_local_mse,
                'r2': best_local_r2,
            }

            # Saving the trained model and its statistics
            self.serialization_util.save_trained_estimator(train_folder, estimator_name, best_local_estimator,
                                                           statistics)

            results[estimator_name] = statistics
            # Updating the best global model if the current model performs better
            if best_local_mse < best_global_mse:
                best_global_mse = best_local_mse
                best_global_r2 = best_local_r2
                best_global_estimator_name = estimator_name
                best_global_estimator = best_local_estimator

        return results, best_global_estimator_name, best_global_estimator, best_global_mse, best_global_r2


class NDScaler(BaseEstimator, TransformerMixin):
    def __init__(self, **scaler_params):
        self.scaler_params = scaler_params
        if scaler_params:
            self.scaler = StandardScaler(**scaler_params)
        else:
            self.scaler = StandardScaler()

    def fit(self, X, y=None):
        # Reshape the ND input to 2D
        print(f"Original shape during fit: {X.shape}")
        X_reshaped = X.reshape(X.shape[0], -1)
        self.scaler.fit(X_reshaped, y)
        return self

    def transform(self, X):
        original_shape = X.shape
        # Reshape the ND input to 2D
        print(f"Original shape during transform: {original_shape}")
        X_reshaped = X.reshape(X.shape[0], -1)
        # Apply the scaler
        X_scaled = self.scaler.transform(X_reshaped)

        # Verify number of elements
        if X_scaled.size != np.prod(original_shape):
            raise ValueError(f"Cannot reshape array of size {X_scaled.size} into shape {original_shape}")

        # Reshape back to ND
        X_scaled_reshaped = X_scaled.reshape(original_shape)
        print(f"Shape after scaling and reshaping: {X_scaled_reshaped.shape}")
        return X_scaled_reshaped

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_params(self, deep=True):
        # Return parameters for this estimator
        return self.scaler.get_params(deep)

    def set_params(self, **params):
        # Extract scaler-specific parameters
        self.scaler.set_params(**params)
        return self


