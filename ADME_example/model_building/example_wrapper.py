import os

import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from tempfile import NamedTemporaryFile
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


class ExampleWrapper:
    """
    A wrapper class for building and validating a feedforward neural network (FFNN) model
    for predicting a target property from molecular descriptors.

    :ivar y_col: The name of the target column in the dataset.
    :ivar ffnn: The feedforward neural network model.
    :ivar x_scaler: Scaler for the input features.
    :ivar y_scaler: Scaler for the target values.
    :ivar calc: Mordred descriptor calculator.
    :ivar desc_name: The name of the column containing the calculated descriptors.
    :ivar cache_file_name: Path to the cache file for storing calculated descriptors.
    """

    def __init__(self, y_col):
        """
        Initializes the ExampleWrapper with the target column name.

        :param y_col: The name of the target column in the dataset.
        """
        self.ffnn = None
        self.x_scaler = None
        self.y_scaler = None
        self.y_col = y_col
        self.calc = Calculator(descriptors, ignore_3D=True)
        self.desc_name = 'mordred_desc'
        self.cache_file_name = "/var/folders/mf/pm2p2q1s4gzbtyyz1cxhlpyc0000gn/T/tmp74z2covx"
        #self.cache_file_name = NamedTemporaryFile().name

    def fit(self, train):
        """
        Fits the FFNN model to the training data.

        :param train: The training dataset containing molecular descriptors and target values.
        """
        # scale the input features and target values
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        X = self.x_scaler.fit_transform(np.stack(train[self.desc_name].values))
        y = self.y_scaler.fit_transform(train[self.y_col].values.reshape(-1, 1)).ravel()
        # train the FFNN model
        self.ffnn.fit(np.stack(X), y)

    def predict(self, test):
        """
        Predicts the target values for the test data using the trained FFNN model.

        :param test: The test dataset containing molecular descriptors.
        :return: The predicted target values.
        """
        # scale the input features and predict the target values
        X = self.x_scaler.fit_transform(np.stack(test[self.desc_name].values))
        pred = self.ffnn.predict(np.stack(X))
        return self.y_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()

    def validate(self, train, test):
        """
        Validates the FFNN model by training on the training data and predicting on the test data.

        :param train: The training dataset.
        :param test: The test dataset.
        :return: The predicted target values for the test dataset.
        """
        self.ffnn = MLPRegressor(hidden_layer_sizes=(300, 300), max_iter=1000, activation='relu')
        # Combine train and test datasets to calculate descriptors
        combo_df = pd.concat([train, test]).copy()
        desc_df = self.calc_descriptors(combo_df)
        # add the descriptors to the train and test datasets
        train = train.merge(desc_df[["SMILES", self.desc_name]], on='SMILES', how='left')
        test = test.merge(desc_df[["SMILES", self.desc_name]], on='SMILES', how='left')
        # train the model and predict on the test set
        self.fit(train)
        pred = self.predict(test)
        return pred

    def calc_descriptors(self, df):
        """
        Calculates molecular descriptors for the given dataset.

        :param df: The dataset containing SMILES strings.
        :return: The dataset with calculated molecular descriptors.
        """
        tqdm.pandas()
        if os.path.exists(self.cache_file_name):
            df = pd.read_parquet(self.cache_file_name)
        else:
            calc = Calculator(descriptors, ignore_3D=True)
            df['mol'] = df.SMILES.progress_apply(Chem.MolFromSmiles)
            desc_df = calc.pandas(df['mol'], nproc=10).values
            # mordred fails to calculate some descriptors for some molecules
            # this code removes columns with failed descriptors
            desc_array = np.array(desc_df, dtype=float)
            nan_cols = np.any(np.isnan(desc_array), axis=0)
            arr_clean = desc_array[:, ~nan_cols]
            # add the descriptors to the dataframe
            df[self.desc_name] = arr_clean.tolist()
            # remove the mol column
            df.drop(columns=['mol'], inplace=True)
            # write the descriptors to a cache file
            df.to_parquet(self.cache_file_name)
            print(f"Descriptors cached to {self.cache_file_name}")
        return df


def main():
    df = pd.read_csv("sol_processed.csv")
    df = df.dropna(subset=["Sol"])
    example_wrapper = ExampleWrapper("Sol")
    for i in range(10):
        train, test = train_test_split(df)
        pred = example_wrapper.validate(train, test)
        print(r2_score(test["Sol"], pred), root_mean_squared_error(test["Sol"], pred))


if __name__ == "__main__":
    main()
