#!/usr/bin/env python

import os
import tempfile
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
import chemprop
import shutil

class ChemPropWrapper:
    def __init__(self, y_name):
        self.y_name = y_name

    def validate(self, train, test):
        with tempfile.TemporaryDirectory() as temp_dirname:
        
            st_cols = ["SMILES", "Sol"]

            train['Sol_bins'] = pd.cut(train['Sol'], bins=10, labels=False)
            train, val = train_test_split(train, test_size=0.1, stratify=train['Sol_bins'], random_state=42)

            # Write the input files
            train[st_cols].to_csv(f"{temp_dirname}/train.csv", index=False)
            val[st_cols].to_csv(f"{temp_dirname}/val.csv", index=False)
            test[st_cols].to_csv(f"{temp_dirname}/test.csv", index=False)

            # Set up ChemProp arguments for training
            data_path = f"{temp_dirname}/train.csv"
            separate_val_path = f"{temp_dirname}/val.csv"
            separate_test_path = f"{temp_dirname}/test.csv"
            save_dir = f"{temp_dirname}/result"

            if os.path.exists(save_dir):
                shutil.rmtree(save_dir, ignore_errors=True)

            arguments = [
                '--data_path', data_path,
                '--separate_test_path', separate_test_path,
                '--separate_val_path', separate_val_path,
                '--num_folds', '1',
                '--epochs', '30',
                '--ensemble_size', '10',
                '--smiles_columns', 'SMILES',
                '--quiet',
                '--dataset_type', 'regression',
                '--save_dir', save_dir,
                '--save_preds',
                '--gpu', '0'
            ]
            args = chemprop.args.TrainArgs().parse_args(arguments)

            # Train the model
            mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

            # Make model predictions
            predict_args = [
                '--test_path', separate_test_path,
                '--preds_path', f"{temp_dirname}/pred.csv",
                '--checkpoint_dir', save_dir,
                '--smiles_columns', 'SMILES',
                '--gpu', '0'
            ]
            predict_args = chemprop.args.PredictArgs().parse_args(predict_args)
            chemprop.train.make_predictions(args=predict_args)

            # Read the results
            result_df = pd.read_csv(f"{temp_dirname}/pred.csv")
            return result_df['Sol'].values

def main():
    df = pd.read_csv("sol_processed.csv")
    train, test = train_test_split(df)
    chemprop_wrapper = ChemPropWrapper("Sol")
    pred = chemprop_wrapper.validate(train, test)
    print(pred)

if __name__ == "__main__":
    main()
