#!/usr/bin/env python

"""
Draw pretty plots.
"""

import os

from IPython import embed
import pandas as pd
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns

def plot_by_hour():
    """Plot likelihood of delays vs. scheduled departure time."""

    df = pd.read_csv('2006.csv')
    df = df.dropna(subset=['DepDelay'])
    df['IsDelayed'] = df.DepDelay > 15
    df.CRSDepTime = df.CRSDepTime // 100  

    sns.set()
    f, ax = plt.subplots(figsize=(5, 3.75))
    data = df.groupby('CRSDepTime').IsDelayed.mean()
    data.plot(ax=ax, kind='bar', color=sns.color_palette()[0])
    ax.set_xlabel('Departure hour')
    ax.set_ylabel('')
    yticks = ax.get_yticks()
    ax.set_yticklabels(['{:.0f}%'.format(x * 100) for x in yticks])
    ax.set_title('Delayed flights by departure hour')
    ax.set_xlim((-0.5, 24))

    plt.tight_layout()
    f.savefig('plots/by_hour.png', bbox_inches='tight')


def plot_timings():
    df = pd.read_csv('plots/exc_times.csv', index_col='Model')

    sns.set()
    f, ax = plt.subplots(figsize=(5, 3.5))
    df.plot(ax=ax, kind='bar', rot=0)
    ax.set_title('Training time')
    ax.set_xlabel('')
    ax.set_ylabel('Time [s]')
    plt.tight_layout()
    f.savefig('plots/execution_time.png', bbox_inches='tight')


if __name__ == '__main__':
    data = np.load('airlines_data.npz')

    experiments = {
        'pred_xgb_t050_d06.npy': {'label': 'XGBoost (50 trees)'},

        'tf_t050_d06_ex01000/pred_tf.npy': {'label': 'TensorFlow (1k ex/layer)'},
        'tf_t050_d06_ex05000/pred_tf.npy': {'label': 'TensorFlow (5k ex/layer)'},
    }

    plot_curves = []
    exp_metrics = []

    for pred_path, exp in experiments.items():
        y_prob = np.load(os.path.join('outputs', pred_path))
        y_pred = y_prob > 0.5
        false_pos_rate, true_pos_rate, _ = metrics.roc_curve(data['y_test'], y_prob)
        plot_curves.append(
            [false_pos_rate, true_pos_rate, exp['label']])
        exp_metrics.append({
            'Model': exp['label'],
            'AUC score': 100 * metrics.roc_auc_score(data['y_test'], y_prob),
        })

    # Do the actual plotting
    sns.set()
    f, ax = plt.subplots(figsize=(5, 3.5))

    for fpr, tpr, label in plot_curves:
        ax.plot(fpr, tpr, label=label)
    
    ax.legend()
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC')
    plt.tight_layout()
    f.savefig('plots/roc.png', bbox_inches='tight')

    # Score table
    metrics_df = pd.DataFrame.from_dict(exp_metrics)
    with open('plots/results_table.txt', 'w') as f:
        metrics_df.to_string(
            f, index=False,
            columns=['Model', 'AUC score'],
            float_format=lambda x: '{:0.1f}'.format(x))

    # Other plots
    plot_timings()
    plot_by_hour()
