#!/usr/bin/env python

import itertools
import logging
import warnings

import numpy as np
import useful_rdkit_utils as uru
from chemprop import data, featurizers, nn, models
from lightning import pytorch as pl
from sklearn.model_selection import train_test_split
import pandas as pd

import torch




class ChemPropMultitaskWrapper:
    def __init__(self, y_name, y_list):
        self.y_name = y_name
        self.y_list = y_list

    def validate(self, train, test):
        pred = run_chemprop_multitask(train, test, self.y_list, num_epochs=50, accelerator="gpu")
        y_index = self.y_list.index(self.y_name)
        return pred[:, y_index]


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


def run_chemprop_multitask(train, test, y_cols, num_epochs=30, accelerator="gpu"):
    # Generate the validation set
    train, val = train_test_split(train, test_size=len(test))
    # Convert data to MoleculeDatapoints
    cols = ["SMILES"] + y_cols
    train_pt = [data.MoleculeDatapoint.from_smi(smi, ys) for smi, *ys in train[cols].values]
    val_pt = [data.MoleculeDatapoint.from_smi(smi, ys) for smi, *ys in val[cols].values]
    test_pt = [data.MoleculeDatapoint.from_smi(smi, ys) for smi, *ys in test[cols].values]
    # Instantiate the featurizer
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    # Create datasets and scalers
    train_dset = data.MoleculeDataset(train_pt, featurizer)
    scaler = train_dset.normalize_targets()

    val_dset = data.MoleculeDataset(val_pt, featurizer)
    val_dset.normalize_targets(scaler)

    test_dset = data.MoleculeDataset(test_pt, featurizer)
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    # Generate data loaders
    num_workers = 0
    train_loader = data.build_dataloader(train_dset, num_workers=num_workers)
    val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
    test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)
    # Create the FFNN
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        ffn_input_dim = mp.output_dim
        ffn = nn.RegressionFFN(input_dim=ffn_input_dim, output_transform=output_transform, n_tasks=len(y_cols))
    # Create the MPNN
    batch_norm = True
    metric_list = [nn.metrics.RMSEMetric() for _ in y_cols]
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)
    ensemble = []
    n_models = 10
    for _ in range(n_models):
        ensemble.append(mpnn)
    # mpnn = models.MPNN.load_from_checkpoint("converted_model_mt.pt")
    # mpnn.message_passing.W_i = torch.nn.Linear(in_features=ffn_input_dim, out_features=300, bias=False)
    # Train the model
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        with DisableLogger():
            for model in ensemble:
                trainer = pl.Trainer(
                    logger=False,
                    enable_checkpointing=False,
                    enable_progress_bar=False,
                    accelerator=accelerator,
                    devices=[2],
                    max_epochs=num_epochs,
                )
                trainer.fit(model, train_loader, val_loader)
            predictions = []
            for model in ensemble:
                predictions.append(torch.concat(trainer.predict(model, test_loader)))

    predictions_tensor = torch.stack(predictions, dim=0)  # Shape: (n_ensemble, n_samples, n_tasks)
    pred_tensor = predictions_tensor.mean(dim=0)  # Shape: (n_samples, n_tasks)
    pred = pred_tensor.numpy()
    return pred

def calc_descriptors(smi_list):
    generator = uru.RDKitDescriptors(hide_progress=True)
    return generator.pandas_smiles(smi_list).values.tolist()


# if __name__ == "__main__":
#     df = pd.read_csv("/home/ubuntu/software/benchmark/data/biogen_logS.csv")
#     y_col = "logS"
#     train, test = train_test_split(df)
#     model = ChemPropRDKitWrapper(y_col)
#     pred = model.validate(train, test)
#     print(pred)
