
import sys
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import argparse
import pandas as pd
import numpy as np
from numpy import array
from numpy import inf
from pandas.plotting import andrews_curves

import matplotlib as mp
import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")

import plotly.express as px # to plot the time series plot
import sklearn as sk
from sklearn import metrics # for the evaluation
from sklearn.metrics import confusion_matrix, auc, roc_curve, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split # for dataset splitting
from sklearn.model_selection import TimeSeriesSplit # splitting time series data
from sklearn.model_selection import GroupShuffleSplit # splitting grouped time series data
from sklearn.model_selection import GroupKFold 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
from skimage.segmentation import mark_boundaries
from sklearn.impute import SimpleImputer 
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

import tensorflow as tf
tf.compat.v1.disable_v2_behavior() # SHAP does not support tensorflow 2.x
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, Dense, Dropout
from tensorflow.keras.constraints import max_norm, unit_norm, non_neg, min_max_norm
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from scipy.cluster.hierarchy import dendrogram, linkage
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Bidirectional, TimeDistributed, Flatten

import lime as lm
import matplotlib as mpl
import decimal
import lime.lime_tabular
import random 
import shap
from shap.plots._beeswarm import summary_legacy #fix for beeswarm plot with Kernel Explainer https://github.com/slundberg/shap/issues/1460
import databricks as dbs
import seaborn as sns
from enum import Enum

#from raiwidgets import ExplanationDashboard
#from interpret.ext.blackbox import TabularExplainer
#from interpret.ext.greybox import DeepExplainer

import deepLearning as dl
import dataprep as dp

class ClassificationMethods(Enum):
    SVM = 1 
    RANDOMFOREST = 2

class AiExplain(object):
    """description of class"""
    X_SCALAR = RobustScaler()
    objDeepLearning = dl.DeepLearning()
    

    def __init__(self):
        print("AiExplain loaded.")

    def f_wrapper(self, X):
        return estimator.predict(X).flatten()
    
    def ExplainTemporal(self, aiModel, dfData, x_dataTrain, x_dataTest, y_dataTest, featureNames, n_topFeatures=20, plotValidationFeatures=False):
        # Need line of code to run SHAP. 
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough

        # init the JS visualization code
        shap.initjs()
        # Create Shap Explainer
        explainer = shap.DeepExplainer(aiModel, x_dataTrain)
        
        #x_test_sample = x_dataTest[:50]
        x_test_sample = x_dataTest
        # Compute Shap values
        shap_values  = explainer.shap_values(x_test_sample, check_additivity=False)

        x_test = np.array(x_test_sample)
        x_test_flatten = x_test.reshape(x_test.shape[0]*x_test.shape[1], x_test.shape[2])
        
        shapValues = np.array(shap_values)
        shap_values_flatten = shapValues.reshape(shapValues.shape[1]*shapValues.shape[2], shapValues.shape[3])
        shap.summary_plot(shap_values_flatten, x_test_flatten, feature_names=featureNames)
        plt.show()

        #importantFeatures = self.get_shap_importance(shap_values_flatten)

        vals= np.abs(shap_values_flatten).mean(0)
        features = pd.DataFrame(featureNames)
        feature_importance = pd.DataFrame(list(zip(featureNames, vals)), columns=['col_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
        feature_importance.head()

        # Get top n feature names
        topFeatureNames = feature_importance['col_name'].values[0:int(n_topFeatures-1)]
        # Filter data by top n features
        dfDataTopF = dfData.filter(np.array(topFeatureNames))
        # Scale data
        dfDataTopFs = self.X_SCALAR.fit_transform(dfDataTopF)
        # Cluster Data
        #objDataPrep.clusterData(dfDataTopFs, 5)

        # Plot the features
        shap.force_plot(explainer.expected_value, shap_values_flatten, x_test_flatten, feature_names = featureNames)

        #y_dataTest_flatten = y_dataTest.repeat(n_timesteps)
        # Plot the Observational Causal Inference
        #clust = shap.utils.hclust(x_test_flatten, y_dataTest_flatten, linkage="single")
        #shap.plots.bar(shap_values_flatten, clustering=clust, clustering_cutoff=1)
     
        # Plot dependecies plots for top 5 features
        iFeature = 0
        
        while(iFeature < 5):
            shap.dependence_plot("rank(" + str(iFeature) + ")", shap_values_flatten, x_test_flatten, feature_names=featureNames)
            #Increment counter
            iFeature += 1


        ## Plot shap metrics on validation data
        #shap_values2  = explainer.shap_values(x_dataValidation, check_additivity=False)

        #x_val = np.array(x_dataValidation)
        #x_val_flatten = x_val.reshape(x_val.shape[0]*x_val.shape[1], x_val.shape[2])
        
        #shapValues2 = np.array(shap_values2)
        #shap_values_flatten2 = shapValues2.reshape(shapValues2.shape[1]*shapValues2.shape[2], shapValues2.shape[3])
        #shap.summary_plot(shap_values_flatten2, x_val_flatten, feature_names=featureNames)

        #plt.show()

        if plotValidationFeatures:
             # Plot shap metrics on training data
            shap_values3  = explainer.shap_values(x_dataTrain, check_additivity=False)

            x_val = np.array(x_dataTrain)
            x_val_flatten = x_val.reshape(x_val.shape[0]*x_val.shape[1], x_val.shape[2])
        
            shapValues3 = np.array(shap_values3)
            shap_values_flatten3 = shapValues3.reshape(shapValues3.shape[1]*shapValues3.shape[2], shapValues3.shape[3])
            shap.summary_plot(shap_values_flatten3, x_val_flatten, feature_names=featureNames)

            plt.show()

        return topFeatureNames

    def ExplainTemporalConcat(self, aiModel, dfData, x_dataTrain, x_dataTest, x_dataStaticTrain, x_dataStaticTest, y_dataTest, temporalFeatureNames, staticFeatureNames, featureNames, n_topFeatures=20):
        
            
        #explainer = DeepExplainer(aiModel, [x_dataTrain, x_dataStaticTrain])
        #global_explanation = explainer.explain_global([x_dataTrain, x_dataStaticTrain])
        #local_explanation = explainer.explain_local([x_dataTest, x_dataStaticTest])


        #print('Ranked global importance values: {}'.format(global_explanation.get_ranked_global_values()))

        #ExplanationDashboard(local_explanation, aiModel, [x_dataTrain, x_dataStaticTrain], 
        plt.clf()
        # Need line of code to run SHAP. 
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough

        # init the JS visualization code
        shap.initjs()
        # Create Shap Explainer
        explainer = shap.DeepExplainer(aiModel, [x_dataTrain, x_dataStaticTrain])
        
        
        # Compute Shap values 
        shap_values  = explainer.shap_values([x_dataTest, x_dataStaticTest], check_additivity=False)

        # Temporal Layers Shap Values
        temporalShapVals = shap_values[0][0]
        staticShapVals = shap_values[0][1]

        # Get temporal shap values and plot
        x_testTemporal = np.array(x_dataTest)
        x_testTemporal_flatten = x_testTemporal.reshape(x_testTemporal.shape[0]*x_testTemporal.shape[1], x_testTemporal.shape[2])
        
        
        temporalShapValues = np.array(temporalShapVals)
        temporalShapValuesFlatten = temporalShapValues.reshape(temporalShapValues.shape[0]*temporalShapValues.shape[1], temporalShapValues.shape[2])
        shap.summary_plot(temporalShapValuesFlatten, x_testTemporal_flatten, feature_names=temporalFeatureNames)
        plt.show()

        # Get static shap values and plot
        plt.clf()
        x_testStatic = np.array(x_dataStaticTest)
        staticShapValues = np.array(staticShapVals)
        shap.summary_plot(staticShapValues, x_testStatic, feature_names=staticFeatureNames)
        plt.show()

        # Mean value of temporal shap values
        valsTemporal= np.abs(temporalShapValuesFlatten).mean(0)
        # Mean value of static shap values
        valsStatic = np.abs(staticShapValues).mean(0)

        featuresTemporal = pd.DataFrame(temporalFeatureNames)
        featuresStatic = pd.DataFrame(staticFeatureNames)
        
        # Get list of top temporal features names
        featuresImportanceTemporal = pd.DataFrame(list(zip(temporalFeatureNames, valsTemporal)), columns=['col_name','feature_importance_vals'])
        featuresImportanceTemporal.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
        featuresImportanceTemporal.head()
        # Get top n feature names
        topFeatureNamesTemporal = featuresImportanceTemporal['col_name'].values[0:int(n_topFeatures-1)]

         # Get list of top static features names
        featuresImportanceStatic = pd.DataFrame(list(zip(staticFeatureNames, valsStatic)), columns=['col_name','feature_importance_vals'])
        featuresImportanceStatic.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
        featuresImportanceStatic.head()
        # Get top n feature names
        topFeatureNamesStatic = featuresImportanceStatic['col_name'].values[0:int(n_topFeatures-1)]
        # Combine temporal and static feature names
        topFeatureNames = topFeatureNamesTemporal.tolist() + topFeatureNamesStatic.tolist()
        # Return all feature names
        return topFeatureNamesTemporal, topFeatureNamesStatic, topFeatureNames


    def ExplainStatic(self, aiModel, dfData, x_dataTrain, x_dataTest, y_dataTest, featureNames, n_topFeatures=20, plotValidationFeatures=False):
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
        # Use the training examples as our background dataset to integrate over
        #explainer = shap.KernelExplainer(predModel, x_dataTrain[:200])
        if(len(x_dataTrain) >= 500):
            explainer = shap.DeepExplainer(aiModel, x_dataTrain[:500]) 
        else:
            explainer = shap.DeepExplainer(aiModel, x_dataTrain) 
        
        # explain the first X predictions
        shap_values = explainer.shap_values(x_dataTest, check_additivity=False)    
        shap.initjs() 
        shap.summary_plot(shap_values, x_dataTest, feature_names=featureNames)
        plt.show()
        plt.clf()

        # Beeswarm plot
        #from shap.plots._beeswarm import summary_legacy
        #summary_legacy(shap_values, featureNames)
        #plt.show()
        #plt.clf()


        #estimator = KerasClassifier(compModel, verbose=0)
        #history = estimator.fit(x_dataTrain, y_dataTrain, validation_data=(x_dataTest, y_dataTest), batch_size=batchSize, epochs=epochsNum)
        #y_predSkl = estimator.predict(x_dataTest)

        #def f_wrapper(X):
        #    return estimator.predict(X).flatten()

        ## Compute Shap values
        #if(len(x_dataTrain) >= 100):
        #    explainer = shap.KernelExplainer(f_wrapper, x_dataTrain[:100])
        #else:
        #    explainer = shap.KernelExplainer(f_wrapper, x_dataTrain)   
        ##Make plot with combined shap values
        #X_test_sample = x_dataTest
        #shap_values  = explainer.shap_values(X_test_sample)
        #shap.summary_plot(shap_values, X_test_sample, feature_names=featureNames)
        #plt.show()

        vals= np.abs(shap_values).mean(0)
        features = pd.DataFrame(featureNames)
        feature_importance = pd.DataFrame(list(zip(featureNames, sum(vals))), columns=['col_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
        feature_importance.head()

        # Get top n features
        topFeatures = feature_importance['col_name'].values[0:int(n_topFeatures-1)]

        return topFeatures

    # Get shap feature importance
    def get_shap_importance(self, shap_values):
        cohorts = {"": shap_values}
        cohort_labels = list(cohorts.keys())
        cohort_exps = list(cohorts.values())
        for i in range(len(cohort_exps)):
            if len(cohort_exps[i].shape) == 2:
                cohort_exps[i] = cohort_exps[i].abs.mean(0)
        features = cohort_exps[0].data
        feature_names = cohort_exps[0].feature_names
        values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])
        feature_importance = pd.DataFrame(
            list(zip(feature_names, sum(values))), columns=['features', 'importance'])
        feature_importance.sort_values(
            by=['importance'], ascending=False, inplace=True)
        return feature_importance

    # Get important features from time series data
    def getImportantFeaturesTemporal(self, dfData, x_dataTrain, y_dataTrain, x_dataTest, y_dataTest, featureNames, classification=ClassificationMethods.RANDOMFOREST, topN=20):
        objDataPrep = dp.DataPrep(dfData)
        # Reshape time series 3d array into 2d array for top feature classification
        x_dt = np.array(x_dataTrain)
        n = x_dt.shape[0]
        t = x_dt.shape[1]

        x_dt_flatten = x_dt.reshape(x_dt.shape[0]*x_dt.shape[1], x_dt.shape[2])

        y_dt = np.array(y_dataTrain)
        yList = []

        for i in range(0, n):
            for x in range(0, t):
                yList.append(y_dt[i])

        y_dt_flatten = np.array(yList)


        # Reshape time series 3d array into 2d array for top feature classification
        x_dtest = np.array(x_dataTest)
        n2 = x_dtest.shape[0]
        t2 = x_dtest.shape[1]

        x_dtest_flatten = x_dtest.reshape(x_dtest.shape[0]*x_dtest.shape[1], x_dtest.shape[2])

        y_dtest = np.array(y_dataTest)
        yList2 = []

        for i in range(0, n2):
            for x in range(0, t2):
                yList2.append(y_dt[i])

        y_dtest_flatten = np.array(yList2)

        #self.objDeepLearning.xgboostTimeSeriesClassification(x_dt_flatten, y_dt_flatten, x_dtest_flatten, y_dtest_flatten, featureNames, topN)

        if(classification == ClassificationMethods.RANDOMFOREST):
            # Run random forest classification and plot features. Get top n feature names
            topFeatureNames = self.objDeepLearning.randForestClassificationFeatures(100, x_dt_flatten, y_dt_flatten, featureNames, topN)
        else:
            # Run svm classification and plot features. Get top n feature names
            topFeatureNames = self.objDeepLearning.svmClassificationFeatures(x_dt_flatten, y_dt_flatten, featureNames, topN)

        # Filter data by top n features
        dfDataTopF = dfData.filter(np.array(topFeatureNames))
        # Scale data
        dfDataTopFs = self.X_SCALAR.fit_transform(dfDataTopF)
        # Cluster Data
        objDataPrep.clusterData(dfDataTopFs, 5)
     
        return topFeatureNames



