## ADME modeling case study

This directory contains the case study shown in Section 3.3.1. 

The primary statistical testing workflow we are recommending is here: `ML_Regression_Comparison.ipynb`.

The `model_building` directory contains:

1. `process_data.ipynb` pulls data from Polaris and processes the data for model building. See the [quick start guide](https://polaris-hub.github.io/polaris/stable/quickstart.html) to Polaris.
2. `run_benchmark.py` runs 5x5 repeated cross validation and saves results for downstream analysis.

The `supplementary_workflows` directory contains:

1. Statistical testing for classification models (`ML_Classification_Comparison.ipynb`)
2. Statistical testing for exceptional cases (`ML_Regression_Comparison_nonparametric.ipynb`)
3. Additional visualization ideas (`alternate_visualizations.ipynb`) 
