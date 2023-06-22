import sys
import argparse
import pandas as pd
import numpy as np
from numpy import inf
import matplotlib as mp
import matplotlib.pyplot as plt
import plotly.express as px # to plot the time series plot

import sklearn as sk
from sklearn import metrics # for the evaluation
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split # for dataset splitting
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from skimage.segmentation import mark_boundaries
from sklearn.impute import SimpleImputer 
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import LeakyReLU, PReLU, Dense
from tensorflow.keras.constraints import max_norm, unit_norm, non_neg, min_max_norm
from scipy.cluster.hierarchy import dendrogram, linkage
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

import lime as lm
import matplotlib as mpl
import decimal
import lime
import lime.lime_tabular
import random 
import shap
from shap.plots._beeswarm import summary_legacy #fix for beeswarm plot with Kernel Explainer https://github.com/slundberg/shap/issues/1460
import databricks as dbs
import seaborn as sns
import sklearn as sk
from sklearn import metrics # for the evaluation
from sklearn.metrics import confusion_matrix, auc, roc_curve, ConfusionMatrixDisplay


class AiVisualizer(object):
    """description of class"""

    def __init__(self):
        print("AiVisualizer loaded.")

    def plotAuc(self, modelPrediction, y_data, showPlot):
         # Area Under Curve
        y_pred_keras = modelPrediction.ravel()
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_data, y_pred_keras)
        auc_keras = auc(fpr_keras, tpr_keras)
        print("AUC: ", auc_keras)

        if showPlot:
            plt.clf()
            plt.figure(1)
            plt.plot([0, 1], 'k--')
            plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
            plt.show()
        return

    # Plot Distribution          
    def plotDistributions(self, xTicks, title, df, x):
        f = sns.countplot(x=x, data=df)
        f.set_title(title)
        f.set_xticklabels(xTicks)
        plt.clf()
        plt.xlabel("")
        plt.show() 
        plt.clf()
        return

     # Plot Mean Distribution   
    def plotMeanDistribution(self, df, x, y, xLabel, yLabel, title):
        plt.clf()
        f = sns.barplot(data=df, x=x, y=y, ci='sd')
        f.set_title(title)
        f.set_xlabel(xLabel)
        f.set_ylabel(yLabel)
        sns.despine()
        plt.show()

    # Correlation matrix and heatmap
    def plotCorrHeatmap(self, dfData):   
        plt.clf()
        corrMatrix = dfData.corr()
        sns.heatmap(corrMatrix, annot=True)
        plt.show()
        return

    # Plot confusion matrix
    def plotConfusionMatrix(self, y_data, y_dataPred, cm_labels):
        plt.clf()
        cm = confusion_matrix(y_data, y_dataPred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
        return

    # Plot features
    def plotFeature(self, dfData, featureName, xName, outcomeName):
        plt.clf()
        df = dfData[[outcomeName, xName, featureName]]
        df.apply(pd.to_numeric)

        df = df.groupby([outcomeName, xName])

        fg = (
            df.mean()
            .reset_index()
            .pipe((sns.catplot, 'data'), x=xName, y=featureName, hue=outcomeName)
        )

        plt.show()

    # Box plot of features
    def boxPlotFeature(self, dfData, featureName, xName, outcomeName):
        plt.clf()
        df = dfData[[outcomeName, xName, featureName]]
        df.apply(pd.to_numeric)
        sns.boxplot(x=xName, y=featureName, hue=outcomeName, data=df, showmeans=True, showfliers = False)
        sns.pointplot(data=df, x=xName, y=featureName, hue=outcomeName, ci=None)
        plt.show()
        return
