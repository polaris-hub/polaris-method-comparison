import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.libqsturng import psturng, qsturng


def calc_regression_metrics(df, cycle_col, val_col, pred_col, thresh):
    """
    Calculate regression metrics (MAE, MSE, R2, prec, recall) for each method and split
x
    :param df: input dataframe must contain columns [method, split] as well the columns specified in the arguments
    :param cycle_col: column indicating the cross-validation fold
    :param val_col: column with the ground truth value
    :param pred_col: column with predictions
    :param thresh: threshold for binary classification
    :return: a dataframe with [cv_cycle, method, split, mae, mse, r2, prec, recall]
    """
    df_in = df.copy()
    metric_ls = ["mae", "mse", "r2", "rho", "prec", "recall"]
    metric_list = []
    df_in['true_class'] = df_in[val_col] > thresh
    # Make sure the thresh variable creates 2 classes
    assert len(df_in.true_class.unique()) == 2, "Binary classification requires two classes"
    df_in['pred_class'] = df_in[pred_col] > thresh

    for k, v in df_in.groupby([cycle_col, "method", "split"]):
        cycle, method, split = k
        mae = mean_absolute_error(v[val_col], v[pred_col])
        mse = mean_squared_error(v[val_col], v[pred_col])
        r2 = r2_score(v[val_col], v[pred_col])
        recall = recall_score(v.true_class, v.pred_class)
        prec = precision_score(v.true_class, v.pred_class)
        rho, _ = spearmanr(v[val_col], v[pred_col])
        metric_list.append([cycle, method, split, mae, mse, r2, rho, prec, recall])
    metric_df = pd.DataFrame(metric_list, columns=["cv_cycle", "method", "split"] + metric_ls)
    return metric_df


def rm_tukey_hsd(df, metric, group_col, alpha=0.05):
    """
    Perform repeated measures Tukey HSD test on the given dataframe.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    metric (str): The metric column name to perform the test on.
    group_col (str): The column name indicating the groups.
    alpha (float): Significance level for the test. Default is 0.05.

    Returns:
    tuple: A tuple containing:
        - result_tab (pd.DataFrame): DataFrame with pairwise comparisons and adjusted p-values.
        - df_means (pd.DataFrame): DataFrame with mean values for each group.
        - df_means_diff (pd.DataFrame): DataFrame with mean differences between groups.
        - pc (pd.DataFrame): DataFrame with adjusted p-values for pairwise comparisons.
    """
    df_means = df.groupby(group_col).mean(numeric_only=True)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                message='divide by zero encountered in scalar divide')
        aov = pg.rm_anova(dv=metric, within=group_col, subject='cv_cycle', data=df, detailed=True)
    mse = aov.loc[1, 'MS']
    df_resid = aov.loc[1, 'DF']

    methods = df[group_col].unique()
    n_groups = len(methods)
    n_per_group = df[group_col].value_counts().mean()

    tukey_se = np.sqrt(2 * mse / (n_per_group))
    q = qsturng(1 - alpha, n_groups, df_resid)

    num_comparisons = len(methods) * (len(methods) - 1) // 2
    result_tab = pd.DataFrame(index=range(num_comparisons),
                              columns=["group1", "group2", "meandiff", "lower", "upper", "p-adj"])

    df_means_diff = pd.DataFrame(index=methods, columns=methods, data=0.0)
    pc = pd.DataFrame(index=methods, columns=methods, data=1.0)

    # Calculate pairwise mean differences and adjusted p-values
    row_idx = 0
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:
                group1 = df[df[group_col] == method1][metric]
                group2 = df[df[group_col] == method2][metric]
                mean_diff = group1.mean() - group2.mean()
                studentized_range = np.abs(mean_diff) / tukey_se
                adjusted_p = psturng(studentized_range * np.sqrt(2), n_groups, df_resid)
                if isinstance(adjusted_p, np.ndarray):
                    adjusted_p = adjusted_p[0]
                lower = mean_diff - (q / np.sqrt(2) * tukey_se)
                upper = mean_diff + (q / np.sqrt(2) * tukey_se)
                result_tab.loc[row_idx] = [method1, method2, mean_diff, lower, upper, adjusted_p]
                pc.loc[method1, method2] = adjusted_p
                pc.loc[method2, method1] = adjusted_p
                df_means_diff.loc[method1, method2] = mean_diff
                df_means_diff.loc[method2, method1] = -mean_diff
                row_idx += 1

    df_means_diff = df_means_diff.astype(float)

    result_tab["group1_mean"] = result_tab["group1"].map(df_means[metric])
    result_tab["group2_mean"] = result_tab["group2"].map(df_means[metric])

    result_tab.index = result_tab['group1'] + ' - ' + result_tab['group2']

    return result_tab, df_means, df_means_diff, pc


# -------------- Plotting routines -------------------#


def make_boxplots_parametric(df, metric_ls):
    """
    Create boxplots for each metric using repeated measures ANOVA.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    metric_ls (list of str): List of metric column names to create boxplots for.

    Returns:
    None
    """
    sns.set_context('notebook')
    sns.set(rc={'figure.figsize': (4, 3)}, font_scale=1.5)
    sns.set_style('whitegrid')
    figure, axes = plt.subplots(1, len(metric_ls), sharex=False, sharey=False, figsize=(28, 8))
    # figure, axes = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(16, 8))

    for i, stat in enumerate(metric_ls):
        model = AnovaRM(data=df, depvar=stat, subject='cv_cycle', within=['method']).fit()
        p_value = model.anova_table['Pr > F'].iloc[0]
        ax = sns.boxplot(y=stat, x="method", hue="method", ax=axes[i], data=df, palette="Set2", legend=False)
        title = stat.upper()
        ax.set_title(f"p={p_value:.1e}")
        ax.set_xlabel("")
        ax.set_ylabel(title)
        x_tick_labels = ax.get_xticklabels()
        label_text_list = [x.get_text() for x in x_tick_labels]
        new_xtick_labels = ["\n".join(x.split("_")) for x in label_text_list]
        ax.set_xticks(list(range(0, len(x_tick_labels))))
        ax.set_xticklabels(new_xtick_labels)
    plt.tight_layout()


def make_normality_diagnostic(df, metric_ls, val_col, pred_col, classification_threshold, group_col="method",
                              cycle_col="cv_cycle"):
    """
    Create a grid of histograms with KDE plots for each metric and method.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    metric_ls (list of str): List of metric column names to create plots for.

    Returns:
    None
    """
    df_metrics = calc_regression_metrics(df, cycle_col=cycle_col, val_col=val_col, pred_col=pred_col,
                                         thresh=classification_threshold)
    df_scaffold_split = df_metrics.query("split == 'scaffold'").copy()

    df_scaffold_split_stacked = df_scaffold_split.melt(id_vars=[cycle_col, group_col, "split"],
                                                       value_vars=metric_ls,
                                                       var_name="metric",
                                                       value_name="value")
    sns.set_context('notebook', font_scale=1)  # Increase font scale
    sns.set_style('whitegrid')
    figure = sns.FacetGrid(df_scaffold_split_stacked, col="method", row="metric", sharex=False, sharey=False,
                           height=1.5, aspect=2.5)
    figure.map_dataframe(sns.histplot, x="value", kde=True)
    plt.tight_layout()


def mcs_plot(pc, effect_size, means, labels=True, cmap=None, cbar_ax_bbox=None,
             ax=None, show_diff=True, cell_text_size=16, axis_text_size=12,
             show_cbar=True, reverse_cmap=False, vlim=None, **kwargs):
    """
    Create a multiple comparison of means plot using a heatmap.

    Parameters:
    pc (pd.DataFrame): DataFrame containing p-values for pairwise comparisons.
    effect_size (pd.DataFrame): DataFrame containing effect sizes for pairwise comparisons.
    means (pd.Series): Series containing mean values for each group.
    labels (bool): Whether to show labels on the axes. Default is True.
    cmap (str): Colormap to use for the heatmap. Default is None.
    cbar_ax_bbox (tuple): Bounding box for the colorbar axis. Default is None.
    ax (matplotlib.axes.Axes): The axes on which to plot the heatmap. Default is None.
    show_diff (bool): Whether to show the mean differences in the plot. Default is True.
    cell_text_size (int): Font size for the cell text. Default is 16.
    axis_text_size (int): Font size for the axis text. Default is 12.
    show_cbar (bool): Whether to show the colorbar. Default is True.
    reverse_cmap (bool): Whether to reverse the colormap. Default is False.
    vlim (float): Limit for the colormap. Default is None.
    **kwargs: Additional keyword arguments for the heatmap.

    Returns:
    matplotlib.axes.Axes: The axes with the heatmap.
    """
    for key in ['cbar', 'vmin', 'vmax', 'center']:
        if key in kwargs:
            del kwargs[key]

    if not cmap:
        cmap = "coolwarm"
    if reverse_cmap:
        cmap = cmap + "_r"

    significance = pc.copy().astype(object)
    significance[(pc < 0.001) & (pc >= 0)] = '***'
    significance[(pc < 0.01) & (pc >= 0.001)] = '**'
    significance[(pc < 0.05) & (pc >= 0.01)] = '*'
    significance[(pc >= 0.05)] = ''

    np.fill_diagonal(significance.values, '')

    # Create a DataFrame for the annotations
    if show_diff:
        annotations = effect_size.round(3).astype(str) + significance
    else:
        annotations = significance

    hax = sns.heatmap(effect_size, cmap=cmap, annot=annotations, fmt='', cbar=show_cbar, ax=ax,
                      annot_kws={"size": cell_text_size},
                      vmin=-vlim if vlim else None, vmax=vlim if vlim else None, **kwargs)

    if labels:
        y_label_list = list(means.index)
        x_label_list = [x + f'\n{means.loc[x].round(2)}' for x in y_label_list]
        hax.set_xticklabels(x_label_list, size=axis_text_size, ha='center', va='top', rotation=0,
                            rotation_mode='anchor')
        hax.set_yticklabels(y_label_list, size=axis_text_size, ha='center', va='center', rotation=90,
                            rotation_mode='anchor')

    hax.set_xlabel('')
    hax.set_ylabel('')

    return hax


def make_mcs_plot_grid(df, stats, group_col, alpha=.05,
                       figsize=(20, 10), direction_dict=None, effect_dict=None, show_diff=True,
                       cell_text_size=16, axis_text_size=12, title_text_size=16):
    """
    Create a grid of multiple comparison of means plots using Tukey HSD test results.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    stats (list of str): List of statistical metrics to create plots for.
    group_col (str): The column name indicating the groups.
    alpha (float): Significance level for the Tukey HSD test. Default is 0.05.
    figsize (tuple): Size of the figure. Default is (20, 10).
    direction_dict (dict): Dictionary indicating whether to minimize or maximize each metric.
    effect_dict (dict): Dictionary with effect size limits for each metric.
    show_diff (bool): Whether to show the mean differences in the plot. Default is True.
    cell_text_size (int): Font size for the cell text. Default is 16.
    axis_text_size (int): Font size for the axis text. Default is 12.
    title_text_size (int): Font size for the title text. Default is 16.

    Returns:
    None
    """
    fig, ax = plt.subplots(2, 3, figsize=figsize)

    for i, stat in enumerate(stats):
        row = i // 3
        col = i % 3

        reverse_cmap = False
        if direction_dict[stat] == 'minimize':
            reverse_cmap = True

        _, df_means, df_means_diff, pc = rm_tukey_hsd(df, stat, group_col, alpha=alpha)

        hax = mcs_plot(pc, effect_size=df_means_diff, means=df_means[stat],
                       show_diff=show_diff, ax=ax[row, col], cbar=True,
                       cell_text_size=cell_text_size, axis_text_size=axis_text_size,
                       reverse_cmap=reverse_cmap, vlim=effect_dict[stat])
        hax.set_title(stat.upper(), fontsize=title_text_size)
        # hax.legend(loc='upper right')

    # If there are less plots than cells in the grid, hide the remaining cells
    if len(stats) < 6:
        for i in range(len(stats), 6):
            row = i // 3
            col = i % 3
            ax[row, col].set_visible(False)

    plt.tight_layout()


def make_scatterplot(df, val_col, pred_col, thresh, cycle_col="cv_cycle", group_col="method"):
    """
    Create scatter plots for each method showing the relationship between predicted and measured values.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    val_col (str): The column name for the ground truth values.
    pred_col (str): The column name for the predicted values.
    thresh (float): Threshold for binary classification.
    cycle_col (str): The column name indicating the cross-validation fold. Default is "cv_cycle".
    group_col (str): The column name indicating the groups/methods. Default is "method".

    Returns:
    None
    """
    df_split_metrics = calc_regression_metrics(df, cycle_col=cycle_col, val_col=val_col, pred_col=pred_col,
                                               thresh=thresh)
    methods = df[group_col].unique()

    fig, axs = plt.subplots(nrows=1, ncols=len(methods), figsize=(25, 10))

    for ax, method in zip(axs, methods):
        df_method = df.query(f"{group_col} == @method")
        df_metrics = df_split_metrics.query(f"{group_col} == @method")
        ax.scatter(df_method[pred_col], df_method[val_col], alpha=0.3)
        ax.plot([df_method[val_col].min(), df_method[val_col].max()],
                [df_method[val_col].min(), df_method[val_col].max()], 'k--', lw=1)

        ax.axhline(y=thresh, color='r', linestyle='--')
        ax.axvline(x=thresh, color='r', linestyle='--')
        ax.set_title(method)

        y_true = df_method[val_col] > thresh
        y_pred = df_method[pred_col] > thresh
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        metrics_text = f"MAE: {df_metrics['mae'].mean():.2f}\nMSE: {df_metrics['mse'].mean():.2f}\nR2: {df_metrics['r2'].mean():.2f}\nrho: {df_metrics['rho'].mean():.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}"
        ax.text(0.05, .5, metrics_text, transform=ax.transAxes, verticalalignment='top')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Measured')

    plt.tight_layout()
    plt.show()


def ci_plot(result_tab, ax_in, name):
    """
    Create a confidence interval plot for the given result table.

    Parameters:
    result_tab (pd.DataFrame): DataFrame containing the results with columns 'meandiff', 'lower', and 'upper'.
    ax_in (matplotlib.axes.Axes): The axes on which to plot the confidence intervals.
    name (str): The title of the plot.

    Returns:
    None
    """
    result_err = np.array([result_tab['meandiff'] - result_tab['lower'],
                           result_tab['upper'] - result_tab['meandiff']])
    sns.set(rc={'figure.figsize': (6, 2)})
    sns.set_context('notebook')
    sns.set_style('whitegrid')
    ax = sns.pointplot(x=result_tab.meandiff, y=result_tab.index, marker='o', linestyle='', ax=ax_in)
    ax.errorbar(y=result_tab.index, x=result_tab['meandiff'], xerr=result_err, fmt='o', capsize=5)
    ax.axvline(0, ls="--", lw=3)
    ax.set_xlabel("Mean Difference")
    ax.set_ylabel("")
    ax.set_title(name)


def make_ci_plot_grid(df_in, metric_list, group_col="method"):
    """
     Create a grid of confidence interval plots for multiple metrics using Tukey HSD test results.

     Parameters:
     df_in (pd.DataFrame): Input dataframe containing the data.
     metric_list (list of str): List of metric column names to create confidence interval plots for.
     group_col (str): The column name indicating the groups. Default is "method".

     Returns:
     None
     """
    figure, axes = plt.subplots(len(metric_list), 1, figsize=(8, 2 * len(metric_list)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for i, metric in enumerate(metric_list):
        df_tukey, _, _, _ = rm_tukey_hsd(df_in, metric, group_col=group_col)
        ci_plot(df_tukey, ax_in=axes[i], name=metric)
    figure.suptitle("Multiple Comparison of Means\nTukey HSD, FWER=0.05")
    plt.tight_layout()
