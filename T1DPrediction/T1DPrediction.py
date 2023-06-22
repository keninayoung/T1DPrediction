#======================================================================================================
# Program: T1DPrediction
# Date: June 22, 2023
# Developers: Kenneth Young, Ph.D.
# Data Sets: Passed in as argument. Temporal data and static data.
# Description: Neural Network aiming to predict IA+/- from TEDDY data
# Notes: The drop column names must be changed to conform to your dataset variable names, also, update
#        the ids used to merge data. The datasets (temporal and static) used for testing had a primary
#        key (mask_id) that was used for the merging.  
#======================================================================================================

import gc
import sys
import argparse
import pandas as pd
import numpy as np
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
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek


import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import LeakyReLU, PReLU, Dense
from tensorflow.keras.constraints import max_norm, unit_norm, non_neg, min_max_norm
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

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


# Import local packages
import imputation
import visualizer
import aiexplainer
import dataprep as dp
import deepLearning as dl
import IAPrediction as iaPred


#------------------------------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------------------------------
def main(argv):
    gc.collect(generation=2)

    # Test GPU availability and instantiate memory growth limitation if True:
    physical_devices = tf.config.list_physical_devices('GPU')

    if (len(physical_devices) > 0):
        print('GPU Available\n')
        print("Num GPUs:", len(physical_devices))
        print("GPUs are available...")
    else:
        print('Running on CPU')

    parser = argparse.ArgumentParser(description='Neural Network aiming to predict T1D.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('--dataStatic', default="data/ncc_static.txt")
    parser.add_argument('--dataTemporal', default="data/ncc_temporal.txt")
    args = parser.parse_args(argv)

    # IA Prediction AI
    runIAPredAI(args.dataTemporal, args.dataStatic)


    return

# Function runs the IA Prediction AI
def runIAPredAI(temporalData, staticData):
    # Set datasets on object
    objIAPred = iaPred.IAPredictionNN(temporalData, staticData)
    # Run AI on time series (temporal) and static data
    objIAPred.runAItemporalAndStatic()

    return

   
if __name__ == '__main__':
    main(sys.argv[1:])