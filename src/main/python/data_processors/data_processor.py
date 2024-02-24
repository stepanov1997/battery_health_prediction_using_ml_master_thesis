from abc import abstractmethod
from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline


class DataProcessor:
    """
    The DataProcessor class is responsible for handling the preprocessing of battery data.
    It includes methods to read, parse, and preprocess the data, preparing it for model training.
    """

    @abstractmethod
    def preprocess_data(self) -> Tuple[Pipeline, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Preprocesses the data by splitting it into training and testing sets and applying a preprocessing pipeline.

        :return: The preprocessing pipeline and the split training and testing data (X_train, y_train, X_test, y_test).
        :rtype: tuple
        """
        pass
