#!/usr/bin/env python

from chemprop_wrapper import ChemPropWrapper
from chemprop_wrapper_multitask import ChemPropMultitaskWrapper
from lgbm_wrapper import LGBMMorganCountWrapper
import useful_rdkit_utils as uru
import pandas as pd

def main():
    dataset_name = "sol_processed.csv"
    df = pd.read_csv(dataset_name)
    y = "Sol" 
    df.dropna(subset=y, inplace=True)
    print(f"Processing {y} with {len(df)} records")
    model_list = [("chemprop_st",ChemPropWrapper),
                  ("chemprop_mt",ChemPropMultitaskWrapper),
                  ("lgbm_morgan", LGBMMorganCountWrapper)]
    group_list = [("random", uru.get_random_clusters),
                  ("butina", uru.get_butina_clusters),
                  ("scaffold", uru.get_bemis_murcko_clusters)]
    result_df = uru.cross_validate(df,model_list,y,group_list)
    result_df.to_csv(f"{y}_results.csv",index=False)


if __name__ == "__main__":
    main()
