#!/usr/bin/env python

from typing import List, Callable, Tuple

import pandas as pd
import useful_rdkit_utils as uru
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm.auto import tqdm

from lgbm_wrapper import LGBMPropWrapper, LGBMMorganCountWrapper
from chemprop_wrapper import ChemPropWrapper, ChemPropRDKitWrapper
from chemprop_wrapper_multitask import ChemPropMultitaskWrapper
# import polaris as po


def cross_validate(df: pd.DataFrame,
                   model_list,
                   y_col: str,
                   y_col_multi: List[str],
                   group_list: List[Tuple[str, Callable]],
                   metric_list: List[Tuple[str, Callable]],
                   n_outer: int = 5,
                   n_inner: int = 5,
                   random_state: int = 42) -> List[dict]:
    metric_vals = []
    fold_df_list = []
    input_cols = df.columns
    for i in tqdm(range(0, n_outer), leave=False):
        kf = uru.GroupKFoldShuffle(n_splits=n_inner, shuffle=True, random_state=random_state)
        for group_name, group_func in group_list:
            # assign groups based on cluster, scaffold, etc
            current_group = group_func(df.SMILES)
            for j, [train_idx, test_idx] in enumerate(
                    tqdm(kf.split(df, groups=current_group), total=n_inner, leave=False)):
                fold = i * n_outer + j
                train = df.iloc[train_idx].copy()
                test = df.iloc[test_idx].copy()

                train['dset'] = 'train'
                test['dset'] = 'test'
                train['group'] = group_name
                test['group'] = group_name
                train['fold'] = fold
                test['fold'] = fold

                for model_name, model_class in model_list:
                    if model_name == 'chemprop_mt':
                        model = model_class(y_col, y_col_multi)
                    else:
                        model = model_class(y_col)
                    pred = model.validate(train, test)
                    test[model_name] = pred
                    metric_dict = {'group': group_name, 'model': model_name, 'fold': fold}
                    for metric_name, metric_func in metric_list:
                        metric_dict[metric_name] = metric_func(test[y_col], pred)
                    metric_vals.append(metric_dict)
                fold_df_list.append(pd.concat([train, test]))
    output_cols = list(input_cols) + ['dset','group','fold'] + [x[0] for x in model_list]
    pd.concat(fold_df_list)[output_cols].to_csv(f"{y_col}_folds_multitask_run3.csv", index=False)
    return metric_vals


def cross_validate_polaris(dataset_name, y, y_multi):
    ds = pd.read_csv(dataset_name)
    df = ds.dropna(subset=y).copy()
    df.rename(columns={"smiles" : "SMILES"},inplace=True)
    print(f"Processing {y} with {len(df)} records")
    model_list = [("lgbm_morgan", LGBMMorganCountWrapper), ("chemprop_st",ChemPropWrapper),
                   ("chemprop_mt",ChemPropMultitaskWrapper)]
    group_list = [("butina", uru.get_butina_clusters), ("random", uru.get_random_split),
                ("scaffold", uru.get_bemis_murcko_clusters)]
    metric_list = [("R2", r2_score), ("MAE", mean_absolute_error)]
    result_list = cross_validate(df, model_list, y, y_multi, group_list, metric_list, 5, 5, 42)
    result_df = pd.DataFrame(result_list)
    print(result_df.head())


        
        
if __name__ == "__main__":
    dataset_name = "sol_processed.csv"
    y = "Sol"
    y_multi =  ['HLM_CLint', 'ER', 'HPPB', 'RPPB', 'RLM_CLint', 'Sol']
    cross_validate_polaris(dataset_name, y, y_multi)
