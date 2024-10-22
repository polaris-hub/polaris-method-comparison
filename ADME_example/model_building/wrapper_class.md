### Writing a wrapper class for a model

In this example, we will write a wrapper class for a model that
uses [mordred](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0258-y) descriptors to generate
predictions using
a [feed-forward neural network](https://scikit-learn.org/dev/modules/generated/sklearn.neural_network.MLPRegressor.html).
The code for this class is the file **example_wrapper.py**. This class contains the
model and the methods to train and evaluate the model. There are only two requirements for a wrapper class:

1. The class should have an `__init__` method that initializes the wrapper and specifies the y column to be predicted.
2. The class should have a `validate` method that takes a training set and a test set and returns a list of predicted
   values.

We begin with the `__init__` method. This method initializes the wrapper with the target column name. The '__init__'
method has the following parameters:

- self.ffnn: A placeholder for the feed-forward neural network model.
- self.x_scaler: A placeholder for the x scaler.
- self.y_scaler: A placeholder for the y scaler.
- self.y_col: The name of the target column in the dataset.
- self.calc: A calculator object that calculates the mordred descriptors.
- self.desc_name: The name of the column that contains the mordred descriptors.
- self.cache_file_name: The name of the cache file that stores the mordred descriptors.

```python
 def __init__(self, y_col):
    self.ffnn = None
    self.x_scaler = None
    self.y_scaler = None
    self.y_col = y_col
    self.calc = Calculator(descriptors, ignore_3D=True)
    self.desc_name = 'mordred_desc'
    self.cache_file_name = NamedTemporaryFile().name
```

Next, we define the `validate` method. This method takes a training set and a test set as input and returns a list of
predicted values. We first initialize the feed-forward neural network model with the specified hyperparameters. We then
combine the training and test datasets to calculate the mordred descriptors. We add the descriptors to the training and test
datasets. We then train the model on the training set and predict the target column on the test set. Finally, we return
the predicted values.

```python
 def validate(self, train, test):
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
```

The two methods above are the only requirements for the wrapper class. The methods below provide the functionality for
the validate method. The `calc_descriptors` method calculates the mordred descriptors for the dataset. Calculating
these descriptors can be computationally expensive, so we cache the results in a file. This way the descriptors only
need
to be calculated once.

We begin by using `tqdm.pandas()` to show a progress bar while calculating the descriptors. We then check if the cache
file exists. If the cache file exists, we read the descriptors from the file. If the cache file does not exist, we use
the mordred calculator to calculate the descriptors. We then remove columns with failed descriptors and add the
descriptor values to the dataframe. We then remove the **mol** column and write the descriptors to a cache file. Finally, we
return the dataframe with the descriptors.

```python
 def calc_descriptors(self, df):
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
```

The `fit` method trains the feed-forward neural network model on the training set. We first scale the x and y values. We
then train the model on the scaled x and y values.

```python
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
```

The `predict` method predicts the target column on the test set. We first scale the x values. We then predict the
target values using the trained model and return the predicted values.  Note that the predicted values are scaled, so 
we need to inverse transform them before returning them.

```python
 def predict(self, test):
    # scale the input features and predict the target values
    X = self.x_scaler.fit_transform(np.stack(test[self.desc_name].values))
    pred = self.ffnn.predict(np.stack(X))
    return self.y_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
```

There are also several example wrappers in this directory. 
