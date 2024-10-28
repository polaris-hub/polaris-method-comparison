## ADME modeling case study

This directory contains the case study shown in Section 3.3.1. 

`ML_Regression_Comparison.ipynb` demonstrates the recommended statistical testing workflow.

`model_building` contains the following:

1. `process_data.ipynb` pulls data from Polaris and processes the data for model building. See [here](https://polaris-hub.github.io/polaris/stable/quickstart.html) for a quick start guide to Polaris.
2. `run_benchmark.py` runs 5x5 repeated cross validation and saves results for downstream analysis

`supplementary_workflows` contains the following:

1. Statistical testing for classification models (`ML_Classification_Comparison.ipynb`)
2. Statistical testing for exceptional cases (`ML_Regression_Comparison_nonparametric.ipynb`)
3. Additional visualization ideas (`alternate_visualizations.ipynb`) 
