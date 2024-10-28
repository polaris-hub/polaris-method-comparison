# Model building

1. `01_process_data.ipynb` pulls data from Polaris and processes the data for model building. See the [quick start guide](https://polaris-hub.github.io/polaris/stable/quickstart.html) to Polaris.
2. `02_run_benchmark.py` runs 5x5 repeated cross validation and saves results for downstream analysis.

**run_benchmark.py** was adapted from [ml_benchmark](https://github.com/PatWalters/ml_benchmark).  All that is necessary for the script is a wrapper class that wraps an ML model and supports a **validate** method. The wrapper class is instantiated with the name of column to be predicted.  The validate method takes dataframes containing training and test sets as input and returns a list of predicted values for the test set. See **wrapper_class.md** for a tutorial on creating a wrapper class. If the ML method requires a validation set, this can be created inside the wrapper by further splitting the training set. 

```python
df = pd.read_csv("myfile.csv")
train, test = train_test_split(df)
chemprop_wrapper = ChemPropWrapper("logS")
pred = chemprop_wrapper.validate(train, test)
```

The **cross_validate** method in **run_benchmark.py** comes from the [useful_rdkit_utils](https://github.com/PatWalters/useful_rdkit_utils).  The **cross_validate** function has four required arguments.

- **df** - a dataframe with a SMILES column  
- **model_list** - a list of tuples containing the model name and the wrapper class described above  
- **y_col** - the name of the column with the y value to be predicted  
- **group_list** - a list of group_names and group memberships (e.g. cluster ids), these can be calculated using the functions get_random_clusters, get_scaffold_clusters, get_butina_clusters, and get_umap_clusters in useful_rkdkit_utils.  

```python

y = "logS"
model_list = [("chemprop",ChemPropWrapper),("lgbm_morgan", LGBMMorganCountWrapper),("lgbm_prop",LGBMPropWrapper)]
group_list = [("random", uru.get_random_clusters),("butina", uru.get_butina_clusters)]
result_df = uru.cross_validate(df,model_list,y,group_list)
```



