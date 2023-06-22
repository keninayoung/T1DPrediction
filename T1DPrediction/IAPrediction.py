#======================================================================================================
# Program: IAPrediction
# Date: June 22, 2023 
# Developers: Kenneth Young, Ph.D.
# Data Sets: Passed in as argument. Temporal data and static data.
# Description: Neural Network aiming to predict IA+/- from TEDDY data
# Notes: The drop column names must be changed to conform to your dataset variable names, also, update
#        the ids used to merge data. The datasets (temporal and static) used for testing had a primary
#        key (mask_id) that was used for the merging.  
#======================================================================================================

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
print(mp.matplotlib_fname())


import os


import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")
print(mp.get_backend())


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
#tf.compat.v1.disable_v2_behavior() # SHAP does not support tensorflow 2.x
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, optimizers
from tensorflow.keras.initializers import LecunNormal
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adamax
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, Dense, Dropout, Concatenate, RepeatVector, AlphaDropout, GaussianDropout
from tensorflow.keras.constraints import max_norm, unit_norm, non_neg, min_max_norm
from tensorflow.keras.metrics import PrecisionAtRecall, RecallAtPrecision, Recall
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from scipy.cluster.hierarchy import dendrogram, linkage
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Bidirectional, TimeDistributed, Flatten, BatchNormalization

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
import warnings


# Import local packages
import imputation
import visualizer
import aiexplainer
import dataprep as dp
import deepLearning as dl


class IAPredictionNN(object):
    # Global Variables 
    warnings.filterwarnings("ignore")
    className = "IAPredictionNN"
    olinkFileName = ""
    staticFileName = ""
    outcomeName = "IA_case_outcome"
    TEST_SIZE = 0.10
    VALIDATION_SIZE = 0.23
    SEEDS = [88, 44, 22, 11, 1]
    SEED = 88
    RND_STATE = np.random.RandomState(SEED)
    #X_SCALAR = StandardScaler()
    X_SCALAR = RobustScaler()
    time_index = 'due_num'
    IMPUTATION_METHOD = imputation.ImputationMethods.INTERPOLATE
    CLASSIFICATION_METHOD = aiexplainer.ClassificationMethods
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
    SAVE_MODEL = True

    # Columns to drop prior to running AI model
    DROP_COLS = ['mask_id',
                 'IA_case_ind',
                 'ia_case_endptmonth',
                 'visit_bf_sero',
                 'months_bf_sero',
                 'match_pair_olink',
                 'vial_barcode_num',
                 'olink_draw_dt',
                 'olink_agedys',
                 'olink_agemos',
                 'month_olink_sample',
                 'year_olink_sample',
                 'olink_run',
                 'olink_SETID',
                 'olink_SPORDER',
                 'GDNF','IL20RA','IL2RB','IL1_alpha','IL2','TSLP','FGF5','IL22RA1']
     
   # GDNF,IL-20RA,IL-2RB,IL-1-alpha,IL-2,TSLP,FGF-5,IL-22RA1  (Olink Below LOD)

    STATIC_DROP_COLS = ['ia_subj',
                        'multi_IA_subj',
                        'subject_id',
                        'year_birth',
                        'probiotics_days_dur',
                        'probiotics_days_stop', 
                        'probiotics_days_start',
                        #'probiotics_duration',
                        #'probiotics_wk_stop', 
                        #'probiotics_wk_start',
                        'formula_any_mos']
                      
                  
    AI_Visualizer = visualizer.AiVisualizer()
    AI_Explainer = aiexplainer.AiExplain()
   
   
    
    def printClassName(self):
        print("Class Name: ", self.className)
       
    def __init__(self, olinkData, staticData):
        self.olinkFileName = olinkData
        self.staticFileName = staticData
        

    #------------------------------------------------------------------------------------------------------
    # Function to perform the primary AI (Neural Network) logic on time series data
    #------------------------------------------------------------------------------------------------------
    def runAItemporalAndStatic(self):
        n_steps = 3
        n_months_start = 3
        n_months = 24   
        T1D_atrisk_age_min = 0
        T1D_atrisk_age_max = 36
        n_timesteps = int((n_months - n_months_start) / n_steps) + 1
        col_timeIndex = self.time_index
        uid = "subjuid"
        joinId = "mask_id"

        # Flag to use LSTM or Transformer
        useLSTM = True

        objDeepLearning = dl.DeepLearning()
       
        dfData = pd.read_csv(self.olinkFileName, sep='\t', lineterminator='\r', na_values=['(NA)', '?']).fillna(0)

        # Drop last row 
        # by selecting all rows except last row
        dfData = dfData.iloc[:-1 , :]
       
        dfData["IA_case_ind"] = dfData["IA_case_ind"].astype(int)
        
        #create a unique id by IA case ind and IA case outcome
        dfData[uid] = dfData["IA_case_ind"].astype(int).astype(str) + "." + dfData["IA_case_outcome"].astype(int).astype(str)
        # Sort data by IA case ind in ascending order
        dfData.sort_values(by=['IA_case_ind', self.time_index], ascending=True, inplace=True, kind="mergesort")
        # Remove NA joinid values 
        dfData = dfData[dfData[joinId].notna()]
        dfData.mask_id = dfData.mask_id.astype(int)

        includeLODasBin = False

        if(includeLODasBin == True):
            # Rename binary fields that are below LOD (Limit of detection). We will analyze these biomarkers as binary
            # 'GDNF','IL20RA','IL2RB','IL1_alpha','IL2','TSLP','FGF5','IL22RA1'
            dfData.rename(columns={'GDNF_yes': 'GDNF_bin', 'IL20RA_yes': 'IL20RA_bin', 'IL2RB_yes': 'IL2RB_bin', 'IL1_alpha_yes': 'IL1_alpha_bin'}, inplace=True)
            dfData.rename(columns={'IL2_yes': 'IL2_bin', 'TSLP_yes': 'TSLP_bin', 'FGF5_yes': 'FGF5_bin', 'IL22RA1_yes': 'IL22RA1_bin'}, inplace=True)

        # Create a copy of the temporal data
        dfDataTemporal = dfData.copy(deep=True) #dfData.drop(joinId, axis=1)
        objTemporalPrep = dp.DataPrep(dfDataTemporal)
        dfDataTemporal = objTemporalPrep.qc_data(dfDataTemporal, self.DROP_COLS)
        
        
        # Remove binary _yes columns from Temporal Data
        dfDataTemporal = dfDataTemporal.loc[:,~dfDataTemporal.columns.str.contains('_yes', case=False)] 
        # Replace NaN values
        dfDataTemporal = objTemporalPrep.prepDataForAI(dfDataTemporal)

        idfTemporal, x_dataTemporal, y_dataTemporal, temporalFeatureNames = objTemporalPrep.getXYFeaturesData(dfDataTemporal, self.outcomeName)
        # Top Temporal Features
        #topTemporalFeatureNames = temporalFeatureNames

        #04/12/2023 Static Features based on best AI model features
        topTemporalFeatureNames = objDeepLearning.kBestFeatures(x_dataTemporal, y_dataTemporal, 'all')

        #only keep important feature columns of temporal data
        dfData = dfData[topTemporalFeatureNames + [str(self.time_index )] + self.DROP_COLS + [str(uid)] + [str(self.outcomeName)]]

        # Remove any duplicated column names
        dfData = dfData.loc[:,~dfData.columns.duplicated()].copy()

        # Remove binary _yes columns
        dfData = dfData.loc[:,~dfData.columns.str.contains('_yes', case=False)] 

        # Load and Prepare Static Data
        dfStaticData = pd.read_csv(self.staticFileName, sep='\t', lineterminator='\r', na_values=['(NA)', '?']).fillna(0)       
        # Drop last row; Todo: need to figure out why last row is empty
        dfStaticData = dfStaticData.iloc[:-1 , :]
        #dfStaticData.rename(columns = {'REGISTRATION_ID': joinId}, inplace = True)
        # Remove duplicate records with same registration id
        dfStaticData.drop_duplicates(joinId, keep='first', inplace=True)
        objStaticDataPrep = dp.DataPrep(dfStaticData)
        objStaticDataPrep.DROP_COLS = self.STATIC_DROP_COLS
        dfStaticData = objStaticDataPrep.prepDataForAI(dfStaticData, 'babybirthtype2')
        dfStaticData = dfStaticData[dfStaticData[joinId].notna()]
        dfStaticData.mask_id = dfStaticData.mask_id.astype(int)
        dfStaticData = objStaticDataPrep.qc_data(dfStaticData, self.STATIC_DROP_COLS)


        # Drop joinid prior to selecting important features
        dfStaticDataF = dfStaticData.drop(joinId, axis=1)
        idfStatic, x_dataStatic, y_dataStatic, staticFeatureNames = objStaticDataPrep.getXYFeaturesData(dfStaticDataF, self.outcomeName)
    
        #04/12/2023 Static Features based on best AI model features
        topStaticFeatureNames = staticFeatureNames
      
        topStaticFeatureNames = topStaticFeatureNames + ['T1D_atrisk_age']

        #only keep important feature columns of static data
        dfStaticData = dfStaticData[topStaticFeatureNames + [str(joinId)]] 


        # Get number of unique registration ids
        dfUniqueRegIds = dfData[~dfData[joinId].isin(dfStaticData[joinId])]

        dfDataMerged = pd.merge(dfStaticData, dfData, on=[joinId], how="inner")


        ####################################################
        #  merge temporal and static data
        ####################################################
        dfData = dfDataMerged

        # Create a data prep class object
        # Used to prep data for AI/Neural Network
        objDataPrep = dp.DataPrep(dfData)
        # Set columns to drop on data prep object
        objDataPrep.DROP_COLS = self.DROP_COLS

        # Convert dt column to datetime or else it will be treated as 
        # an object. Then convert to int. 
        # QC on data
        dfData = objDataPrep.qc_data(dfData, self.DROP_COLS)

        # Filter data with query method, limit to due_num <= n months
        #dfData.query(str(self.time_index) + ' <= ' + str(n_months), inplace = True)
        dfData.query('T1D_atrisk_age >= ' + str(T1D_atrisk_age_min) + ' & ' + 'T1D_atrisk_age < ' + str(T1D_atrisk_age_max) + ' & ' + str(self.time_index) + ' >= ' + str(n_months_start) + ' & ' + str(self.time_index) + ' <= ' + str(n_months), inplace = True)

        # Sort data so time series in correct order
        dfData.sort_values(by=[uid, col_timeIndex], ascending=True, inplace=True, kind="mergesort")     

         # Impute data (LOCF, Remove, Interpolate)        
        if self.IMPUTATION_METHOD == imputation.ImputationMethods.LOCFIMPUTE:
            # Filter data so only study participant data are within
            # the specified timepoints
            #dfNN = self.imputeMissingTimepoints(dfData, n_timesteps, uid)
            impute = imputation.Imputate()
            dfNN = impute.imputeMissingTimepoints(dfData, n_timesteps, n_months_start, n_steps, uid)            
            # Sort data by subjuid and due num in ascending order
            dfNN.sort_values(by=[uid, 'due_num'], ascending=True, inplace=True, kind="mergesort")
            dfData = dfNN
            # resetting the DataFrame index
            dfData = dfData.reset_index(drop=True)
        elif self.IMPUTATION_METHOD == imputation.ImputationMethods.REMOVEIMPUTE:
            # Filter data so only study participant data are within
            # the specified timepoints
            impute = imputation.Imputate()
            dfData = impute.filterTimepoints(dfData, n_timesteps, uid)
            # resetting the DataFrame index
            dfData = dfData.reset_index(drop=True)
        elif self.IMPUTATION_METHOD == imputation.ImputationMethods.INTERPOLATE:
            interop = imputation.Interpolate()
            dfData = interop.interpolateMissingTimepoints(dfData, n_months_start, n_timesteps, n_steps, '1D', self.time_index, uid)
             # resetting the DataFrame index
            dfData = dfData.reset_index(drop=True)
            # Reset due_nums
            dn = n_months_start      
            dnTimeSteps = n_timesteps
            dnCount = 0
            # iterate through each row and select 
            for i in range(len(dfData)) :
                # Update time_index (typically due_num)
                dfData.loc[i, self.time_index] = float(dn)
                dn += n_steps
                # Is time_index > max months? Reset back to n_steps
                if(dn > n_months):
                    dn = n_months_start

        # Get unique ids
        u_SubjId = uid
        lstUids = dfData[uid].unique()
        dfUids = pd.DataFrame(lstUids)

        dataSeq = list()
        dataY = list()
        ctr = 0

        for index, row in dfUids.iterrows():
            dfSeq = dfData.query(uid + ' == ' + str(int(row[0])), inplace=False)
            dataSeq.append(dfSeq)
            ctr = ctr + 1
     
        # Replace NaN values
        dfData = objDataPrep.replaceNaN(dfData)

        # Deep copy the dfData data frame
        dfDataPreserve = dfData.copy(deep=True)

        # Drop col time index (due_num)
        dfData.drop(col_timeIndex, axis=1, inplace=True)

        # Split temporal and static data int x, y, and feature names  
        idf, x_data, x_dataStatic, y_data, featureNames, featureNamesTemporal, featureNamesStatic = objDataPrep.splitDataTemporalAndStatic(dfData, uid, topStaticFeatureNames, self.outcomeName)

        plotDists = False

        if plotDists:
            # Plot IA Distribution
            self.AI_Visualizer.plotDistributions(['IA-', 'IA+'], 'IA distribution', dfData, self.outcomeName)

        # fit data
        #x_data, y_data, featureNames = objDataPrep.fitData(idf, x_data, y_data, self.outcomeName)

        # Remove uid from featureNames
        featureNames.remove(uid)

        # Drop unique id column from data sets, not needed for AI 
        x_data.drop(uid, axis=1, inplace=True)
        dfData.drop(uid, axis=1, inplace=True)

        # Correlation matrix and heatmap
        #self.AI_Visualizer.plotCorrHeatmap(dfData)

        x_data = x_data.to_numpy()
        x_dataStatic = x_dataStatic.to_numpy()

        print('x_data shape: ' + str(x_data.shape))
        print('x_dataStatic shape: ' + str(x_dataStatic.shape))

        #NOTE: Fixed bug temporalizing data 
        # Transform temporal data from 2d to 3d
        x_data, y_data = objDataPrep.split_sequences(x_data, y_data, n_timesteps)

        print('x_data 3D shape: ' + str(x_data.shape))
        print(type(x_data))

        print('y_data shape: ' + str(y_data.shape))
        print(type(y_data))


        # Number of features
        n_features = x_data[0].shape[1]
        #n_features = x_data.shape[2]

        # split data into a training set and test dataset
        x_dataTrain, x_dataTest, x_dataStaticTrain, x_dataStaticTest, y_dataTrain, y_dataTest = train_test_split(x_data, x_dataStatic, y_data, test_size = self.TEST_SIZE, random_state=self.RND_STATE, stratify=y_data, shuffle=True)
               
        # split training set into training and validation dataset
        x_dataTrain, x_dataValidation, x_dataStaticTrain, x_dataStaticValidation, y_dataTrain, y_dataValidation = train_test_split(x_dataTrain, x_dataStaticTrain, y_dataTrain, test_size = self.VALIDATION_SIZE, random_state=self.RND_STATE, stratify=y_dataTrain, shuffle=True)
        
        print(x_dataTrain.shape, x_dataValidation.shape, x_dataTest.shape)
        print(y_dataTrain.shape, y_dataValidation.shape, y_dataTest.shape)      

        # Scale and normalize temporal data
        x_dataTrain = objDataPrep.scaleTemporalData(x_dataTrain, True)
        x_dataValidation = objDataPrep.scaleTemporalData(x_dataValidation, False)
        x_dataTest = objDataPrep.scaleTemporalData(x_dataTest, False)

        # Scale and normalize static data
        x_dataStaticTrain = self.X_SCALAR.fit_transform(x_dataStaticTrain)
        x_dataStaticValidation = self.X_SCALAR.transform(x_dataStaticValidation)
        x_dataStaticTest = self.X_SCALAR.transform(x_dataStaticTest)

        rowCountTest = x_dataTest.shape[0]
        rowCountTrain = x_dataTrain.shape[0]
        rowCountValidation = x_dataValidation.shape[0]

    
        if(rowCountTest < 64):
            objDeepLearning.batchSize = rowCountTest*2
        else:
            objDeepLearning.batchSize = 64
        
        # time span less than 24 months? Tune hyperparameters
        if n_months < 28:
            objDeepLearning.n_epochs = 1500
            objDeepLearning.learning_rate = 0.00009
            objDeepLearning.beta_1=0.8
            objDeepLearning.beta_2=0.9
        else:
            objDeepLearning.n_epochs = 2000
            objDeepLearning.learning_rate = 0.00005
            objDeepLearning.beta_1=0.9
            objDeepLearning.beta_2=0.999

        objDeepLearning.dropoutRate = 0.50
        objDeepLearning.alphaDropoutRate = 0.40
        objDeepLearning.gaussianDropoutRate = 0.50
        objDeepLearning.decayRate = objDeepLearning.learning_rate / objDeepLearning.n_epochs
        objDeepLearning.momentum = 0.9
        objDeepLearning.amsgrad=False
        

        if(useLSTM == True):
            # Get LSTM model
            nnModel = objDeepLearning.biLstmModel(n_timesteps, n_features, x_dataStaticTrain.shape[1], autoEncoder=True)
        else:
            # Get Transformer model
            nnModel = objDeepLearning.transformer_model(
                n_timesteps, 
                n_features,
                x_dataTrain.shape[1:], 
                x_dataStaticTrain.shape[1], 
                head_size=32, 
                num_heads=1, 
                ff_dim=1,
                num_transformer_blocks=1,
                mlp_units=[32],
                mlp_dropout=objDeepLearning.alphaDropoutRate,
                dropout=objDeepLearning.dropoutRate)


        #print model summary

        nnModel.summary()

        dateTimeObj = datetime.datetime.now()
        timestampStr = dateTimeObj.strftime("%Y%b%d-%H%M%S")
        print("Time Model Fit Start: ", timestampStr) # Time AI Model Fit Started

        # NN model without transformer
        model_history = nnModel.fit([x_dataTrain, x_dataStaticTrain], y_dataTrain, epochs=objDeepLearning.n_epochs, batch_size=objDeepLearning.batchSize, validation_data=([x_dataValidation, x_dataStaticValidation], y_dataValidation), validation_split=0.0, shuffle=False, verbose=0)
        
        # Save and Plot model?
        if(self.SAVE_MODEL == True):
            ###############################################################################################################################
            # Plot Model
            # NOTE:if receiving error, you must install pydot and graphviz; 32-bit version of graphviz is needed currently)
            ###############################################################################################################################
            tf.keras.utils.plot_model(nnModel, to_file='models/IA_temporal_pred_model_' + timestampStr + '.png', show_shapes=True)
     
            # Save Model
            nnModel.save('models/IA_tp_pred_model' + timestampStr)
        
        # Test Model
        #evalscore = nnModel.evaluate([x_dataTest, x_dataStaticTest], y_dataTest, batch_size=batchSize, verbose=0)

        evalscore = nnModel.evaluate([x_dataTest, x_dataStaticTest], y_dataTest, verbose=0)

        dateTimeObj = datetime.datetime.now()
        timestampStr = dateTimeObj.strftime("%Y%b%d-%H%M%S")
        print("Time Model Eval End: ", timestampStr) # Time AI Model Eval Ended

        print("Model Test Loss: ", evalscore[0]) # How far the predicted values deviate from the actual values in the training data
        print("Model Test Binary Accuracy: ", evalscore[1]) # Number of IA +/- accurately predicted
        print("Model Test Accuracy: ", evalscore[2]) # Number of IA +/- accurately predicted
        print("Model Test Precision: ", evalscore[3]) # When model predicts IA+, it is correct X% of time.
        print("Model Test Recall: ", evalscore[4]) # Model correctly identifies X% of all IA+

        # Plot accuracy
        plt.clf()
        plt.plot(model_history.history['binary_accuracy'])
        plt.plot(model_history.history['val_binary_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='upper left')        
        plt.show()
        
        # Plot loss
        plt.clf()
        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        # Plot precision
        plt.clf()
        plt.plot(model_history.history['precision'])
        plt.plot(model_history.history['val_precision'])
        plt.title('Model Precision')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        # Plot recall
        plt.clf()
        plt.plot(model_history.history['recall'])
        plt.plot(model_history.history['val_recall'])
        plt.title('Model Recall')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        
        modelPrediction = nnModel.predict([x_dataTest, x_dataStaticTest], batch_size=objDeepLearning.batchSize)
        
        # Print the shape
        print("Prediction Shape:", modelPrediction.shape)

        # Get optimal number of epoch
        predModelEpochOpt = objDeepLearning.optimalEpoch(model_history)

        y_modelPrediction = (modelPrediction > 0.5).astype("int32")
        y_modelPredFlat = y_modelPrediction.flatten()

        m = tf.keras.metrics.binary_accuracy(y_dataTest, y_modelPredFlat, threshold=0.5)
       
        predOutputMatches = y_modelPredFlat == y_dataTest

        print("Y values: ", *y_dataTest, sep = ", ")
    
        # Convert predicted output matches from "True, False" to "1, 0" Binary (Int32)
        predOutputMatchesBin = predOutputMatches.astype("int32")
        print("Which predictions match with binary labels:", predOutputMatches)

        # Get the sum of accurately predicted records, count the true values (1)
        accCount = sum(predOutputMatches)

        # Correct number of predications
        correct_predictions = sum(predOutputMatchesBin)
        # Incorrect number of predictions
        incorrect_predictions =  (rowCountTest - correct_predictions)

        print("Correct predictions: ", correct_predictions)
        print("Incorrect predictions: ", incorrect_predictions)

        # Accuracy calculated
        accuracyCalculated = (correct_predictions / rowCountTest)
        # Calculate F Test score. Range from 1 (highest accuracy) to 0 (lowest accuracy)
        f1Score = metrics.f1_score(y_dataTest, y_modelPredFlat, pos_label=0, average="weighted")
        
        print("f-score: ", f1Score)

        # Plot AUC
        self.AI_Visualizer.plotAuc(modelPrediction, y_dataTest, True)

        # Generate a confusion matrix
        self.AI_Visualizer.plotConfusionMatrix(y_dataTest, y_modelPredFlat, ["IA-", "IA+"])

        # Explain blackbox AI model with SHAP and get important features from AI model
        topFeaturesTemporal, topFeaturesStatic, topFeaturesCombined = self.AI_Explainer.ExplainTemporalConcat(nnModel, dfData, x_dataTrain, x_dataTest, x_dataStaticTrain, x_dataStaticTest, y_dataTest, featureNamesTemporal, featureNamesStatic, featureNames, 25)

        print("Top AI Model Features: ")
        print(topFeaturesCombined)

         # Get important features for each test participant (top 5)

        topFeatureCount = 1
        icount = 0
        for i in range(len(x_dataTest)):
            if(icount < topFeatureCount): 
                # Add a dimension to the arrays
                ix_dataTest = np.expand_dims(x_dataTest[i], axis=0)
                ix_dataStaticTest = np.expand_dims(x_dataStaticTest[i], axis=0)
                iy_dataTest = np.expand_dims(y_dataTest[i], axis=0)
                itopFeaturesTemporal, itopFeaturesStatic, itopFeaturesCombined = self.AI_Explainer.ExplainTemporalConcat(nnModel, dfData, x_dataTrain, ix_dataTest, x_dataStaticTrain, ix_dataStaticTest, iy_dataTest, featureNamesTemporal, featureNamesStatic, featureNames, 25)
            else:
                break 
               
            icount += 1

        
        plotFeatures = True
       
        if plotFeatures:
            count = 0
            # Plot top AI temporal features
            for fnt in topFeaturesTemporal:
                # plot each feature
                if(count < topFeatureCount):
                    self.AI_Visualizer.boxPlotFeature(dfDataPreserve, fnt, col_timeIndex, self.outcomeName)
                    self.AI_Visualizer.plotFeature(dfDataPreserve, fnt, col_timeIndex, self.outcomeName)
                    count += 1
                else:
                    break
            count = 0
            # Plot top AI static features
            for fns in topFeaturesStatic:
                 # plot each feature
                if(count < topFeatureCount):
                    title = 'Mean ' + str(fns) + ' for ' + str(self.outcomeName)
                    self.AI_Visualizer.plotMeanDistribution(dfDataPreserve, self.outcomeName, fns, str(self.outcomeName), str(fns), title)
                    count += 1
                else:
                    break

        # Explain blackbox AI model with SHAP and get important features from AI model
        topFeaturesTemporal, topFeaturesStatic, topFeaturesCombined = self.AI_Explainer.ExplainTemporalConcat(nnModel, dfData, x_dataTrain, x_dataValidation, x_dataStaticTrain, x_dataStaticValidation, y_dataValidation, featureNamesTemporal, featureNamesStatic, featureNames, 25)

        print("Top AI Model Validation Features: ")
        print(topFeaturesCombined)

       # End AI timeseries function
        return

    
    # define model
    def compModel(n_timesteps, n_features, dropoutRate, learning_rate, beta_1, beta_2):
        model = Sequential()
        model.add(InputLayer(input_shape=(n_timesteps, n_features)))
        model.add(TimeDistributed(Dense(250, activation='selu')))
        model.add(layers.Dropout(dropoutRate))
        model.add(Bidirectional(LSTM(350, activation='selu', recurrent_dropout=dropoutRate, return_sequences=True)))
        model.add(layers.Dropout(dropoutRate))
        model.add(Flatten())
        model.add(Dense(15, activation='selu'))
        model.add(layers.Dropout(dropoutRate))
        model.add(Dense(5, activation='selu'))
        model.add(layers.Dropout(dropoutRate))
        model.add(Dense(1, activation='sigmoid'))
       
        #opt = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
        #opt = SGD(learning_rate=0.0008, momentum=momentum, nesterov=True) # achieving 91-92% accuracy, but needs ~600 epochs
        #opt = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        opt = Adam(learning_rate=learning_rate, beta_1 = beta_1, beta_2 = beta_2, amsgrad=False)
            
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy', 'accuracy'])

        return model


    def lstmModel(n_timesteps, n_features, dropoutRate, learning_rate, beta_1, beta_2):
        rnn_input = tf.keras.Input(shape=(n_timesteps, n_features), name='rnn_input')
        rnn_input_x = TimeDistributed(Dense(250, activation='selu'))(rnn_input)
        rnn_input_x = Dropout(dropoutRate)(rnn_input_x)
        rnn_input_x = Bidirectional(LSTM(350, activation='selu', recurrent_dropout=dropoutRate, return_sequences=True))(rnn_input_x)
        rnn_input_x = Dropout(dropoutRate)(rnn_input_x)
        rnn_input_x = Flatten()(rnn_input_x)
        rnn_input_x = Dense(15, activation='selu')(rnn_input_x)
        rnn_input_x = Dropout(dropoutRate)(rnn_input_x)
        rnn_input_x = Dense(5, activation='selu')(rnn_input_x)
        rnn_input_x = Dropout(dropoutRate)(rnn_input_x)

        output = Dense(1, activation='sigmoid', name='output')(rnn_input_x)
        opt = Adam(learning_rate=learning_rate, beta_1 = beta_1, beta_2 = beta_2, amsgrad=False)
            
        model = Model(inputs=[rnn_input], outputs=[output])
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy', 'accuracy'])

        return model

        