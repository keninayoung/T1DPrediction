import sys
import argparse
import pandas as pd
import numpy as np
from numpy import inf, array
import matplotlib as mp
import matplotlib.pyplot as plt
import plotly.express as px # to plot the time series plot

import sklearn as sk
from sklearn import metrics # for the evaluation
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split # for dataset splitting
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
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

import imputation

class DataPrep(object):
   
    dataFileName = ""
    DROP_COLS = []
    SEED = 88
    RND_STATE = np.random.RandomState(SEED)
    X_SCALAR = RobustScaler()
     
    def __init__(self, df):
        self.dataFileName = df


    #------------------------------------------------------------------------------------------------------
    # Function to perform the primary AI (Neural Network) logic
    # Returns: DataFrame
    #------------------------------------------------------------------------------------------------------
    def prepDataForAI(self, dfData, colsToCat=None): 
        # Replace new lines
        dfData = dfData.replace(r'\n','', regex=True) 
        dfData.replace('?',np.NaN,inplace=True)

        # columns to convert to Categorical
        if(colsToCat != None):
            #dfData[colsToCat] = pd.Categorical(dfData[colsToCat])
            dfData[colsToCat] = dfData[colsToCat].astype('category')
        # Get all categorical columns
        cat_columns = dfData.select_dtypes(['category']).columns
        # convert categorical columns to numeric
        if(cat_columns.size > 0):
            dfData[cat_columns] = dfData[cat_columns].apply(lambda x: x.cat.codes)

        # Get all object columns
        object_columns = dfData.select_dtypes(['object']).columns
        # convert object columns to numeric
        if(object_columns.size > 0):
            dfData[object_columns] = dfData[object_columns].apply(lambda x: x.astype(str).astype(float))

        impute = imputation.Imputate()
        # Impute missing values
        dfData = impute.simpleImputation(dfData)
        return dfData

    #------------------------------------------------------------------------------------------------------
    # Function to perform the primary AI (Neural Network) logic
    # Returns:  idf, x_data, y_data
    #------------------------------------------------------------------------------------------------------
    def imputeData(self, dfData, outcome): 
        impute = imputation.Imputate()
        return impute.simpleImputation(dfData, outcome)


    #------------------------------------------------------------------------------------------------------
    # Function to perform the primary AI (Neural Network) logic
    # Returns: DataFrame
    #------------------------------------------------------------------------------------------------------
    def imputeData(self, dfData): 
        impute = imputation.Imputate()
        return impute.simpleImputation(dfData)

    #------------------------------------------------------------------------------------------------------
    # Function to replace ? with NaN
    # Returns: DataFrame
    #------------------------------------------------------------------------------------------------------
    def replaceNaN(self, dfData):
        dfData.replace('?',np.NaN,inplace=True)
        return dfData

    #------------------------------------------------------------------------------------------------------
    # Function to extract x and y data from dataset
    #------------------------------------------------------------------------------------------------------
    def getXYData(self, dfData, outcome): 
        idf = pd.DataFrame(dfData)
        idf.columns = dfData.columns

        print(idf.isnull().sum())

        x_data = idf.loc[:, idf.columns != outcome]
        y = idf[outcome]
        y_data = y.values
        y_data = y_data.astype('int32')

        return idf, x_data, y_data

    #------------------------------------------------------------------------------------------------------
    # Function to extract x and y data from dataset
    #------------------------------------------------------------------------------------------------------
    def getXYFeaturesData(self, dfData, outcome): 
        idf = pd.DataFrame(dfData)
        idf.columns = dfData.columns

        print(idf.isnull().sum())

        x_data = idf.loc[:, idf.columns != outcome]
        y = idf[outcome]
        y_data = y.values
        y_data = y_data.astype('int32')

        featureNames = (idf.loc[:, idf.columns != outcome]).columns.tolist()

        return idf, x_data, y_data, featureNames

    #------------------------------------------------------------------------------------------------------
    # Function to return only outcome column value = 1
    #------------------------------------------------------------------------------------------------------
    def getOutcomeOnly(self, df, outcome):
        return df.loc[df[outcome] == 1]

    #------------------------------------------------------------------------------------------------------
    # Function to return only outcome column value = 1
    #------------------------------------------------------------------------------------------------------
    def getOutcomeExcluded(self, df, outcome):
        return df.loc[df[outcome] == 0]

    #------------------------------------------------------------------------------------------------------
    # Function to perform the quality control on the data
    #------------------------------------------------------------------------------------------------------
    def qc_data(self, dfData, colsToDrop): 

        strCols = dfData.columns[dfData.dtypes.eq('object')]
        # Convert all string values to numeric
        dfData[strCols] = dfData[strCols].apply(pd.to_numeric, errors='coerce')

        # Get column names
        data_top = dfData.head() 
        
        # Remove non-predicting columns by name
        dfData.drop(colsToDrop, inplace=True, axis = 1)
            
        rowCount = dfData.shape[0]
        colCount = dfData.shape[1]

        dfData = dfData.replace(r'\\n',  ' ', regex=True)
        # summarize first 5 rows
        print(dfData.head(25))

        dfData.replace('?', np.nan, inplace= True)

        print(dfData.isnull().sum())

        # summarize the number of rows with missing values for each column
        for i in range(dfData.shape[1]):
            # count number of rows with missing values
            n_miss = dfData.iloc[i, :].isnull().sum()
            perc = n_miss / dfData.shape[0] * 100
            print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))

        return dfData

    #------------------------------------------------------------------------------------------------------
    # Function to oversample data using Synthetic Minority Oversampling Technique (SMOTE) data augumentation
    #------------------------------------------------------------------------------------------------------
    def overUndersampleData(self, idf, x_data, y_data, outcome): 
        # Use Synthetic Minority Oversampling Technique (SMOTE) data augumentation. 
        # Involves developing predictive models on classification datasets that have a severe class imbalance.
        # Transform the dataset with SMOTE. SMOTE is an oversampling technique and is very effective in handling 
        # class imbalance. SMOTEEN combines undersampling technqies (ENN or Tomek) with oversampling of SMOTE, to 
        # increase the effectiveness of handling imbalanced classes.

        oversample = SMOTETomek(random_state=self.RND_STATE)
        x_dataS, y_dataS = oversample.fit_resample(x_data, y_data)

        featureNames = (idf.loc[:, idf.columns != outcome]).columns.tolist()

        return x_dataS, y_dataS, featureNames


    #------------------------------------------------------------------------------------------------------
    # Function to oversample data using Synthetic Minority Oversampling Technique (SMOTE) data augumentation
    #------------------------------------------------------------------------------------------------------
    def overUndersampleDataENN(self, idf, x_data, y_data, outcome): 
        # Use Synthetic Minority Oversampling Technique (SMOTE) data augumentation. 
        # Involves developing predictive models on classification datasets that have a severe class imbalance.
        # Transform the dataset with SMOTE. SMOTE is an oversampling technique and is very effective in handling 
        # class imbalance. SMOTEEN combines undersampling technqies (ENN or Tomek) with oversampling of SMOTE, to 
        # increase the effectiveness of handling imbalanced classes.

        oversample = SMOTEENN(random_state=self.RND_STATE)
        x_dataS, y_dataS = oversample.fit_resample(x_data, y_data)

        featureNames = (idf.loc[:, idf.columns != outcome]).columns.tolist()

        return x_dataS, y_dataS, featureNames
    
    
    #------------------------------------------------------------------------------------------------------
    # Function to oversample data using Synthetic Minority Oversampling Technique (SMOTE) data augumentation
    #------------------------------------------------------------------------------------------------------
    def oversampleData(self, idf, x_data, y_data, outcome): 
        # Use Synthetic Minority Oversampling Technique (SMOTE) data augumentation. 
        # Involves developing predictive models on classification datasets that have a severe class imbalance.
        # Transform the dataset with SMOTE. SMOTE is an oversampling technique and is very effective in handling 
        # class imbalance. SMOTEEN combines undersampling technqies (ENN or Tomek) with oversampling of SMOTE, to 
        # increase the effectiveness of handling imbalanced classes.

        oversample = SMOTE(random_state=self.RND_STATE)
        x_dataS, y_dataS = oversample.fit_resample(x_data, y_data)

        featureNames = (idf.loc[:, idf.columns != outcome]).columns.tolist()

        return x_dataS, y_dataS, featureNames

    #------------------------------------------------------------------------------------------------------
    # Function to fit data with feature names
    #------------------------------------------------------------------------------------------------------
    def fitData(self, idf, x_data, y_data, outcome): 
        x_dataS = x_data
        y_dataS =  y_data
        featureNames = (idf.loc[:, idf.columns != outcome]).columns.tolist()

        return x_dataS, y_dataS, featureNames


    #------------------------------------------------------------------------------------------------------
    # Function to split combined temporal and static data with feature names
    #------------------------------------------------------------------------------------------------------
    def splitDataTemporalAndStatic(self, dfData, uidName, staticFeatureNames, outcome): 
        idf = pd.DataFrame(dfData)
        idf.columns = idf.columns

        print(idf.isnull().sum())

        x_data = idf.loc[:, idf.columns != outcome]
        # Drop static columns from temporal data set
        x_dataTemporal = x_data.loc[:, ~x_data.columns.isin(staticFeatureNames)]
       
        x_dataStatic = idf.loc[:, idf.columns != outcome]
        # Remove duplicate records with same uid
        x_dataStatic.drop_duplicates(subset=[uidName], keep='first', inplace=True)
        # Drop temporal columns from static data set
        x_dataStatic = x_dataStatic.loc[:, ~x_dataStatic.columns.isin(x_dataTemporal.columns)]

        y = idf[outcome]
        y_data = y.values
        y_data = y_data.astype('int32')

        featureNames = (idf.loc[:, idf.columns != outcome]).columns.tolist()
        featureNamesTemporal = (x_dataTemporal.loc[:, x_dataTemporal.columns != uidName]).columns.tolist()
        featureNamesStatic = x_dataStatic.columns.tolist()

        return idf, x_dataTemporal, x_dataStatic, y_data, featureNames, featureNamesTemporal, featureNamesStatic

    #------------------------------------------------------------------------------------------------------
    # Function to perform clustering on the data
    #------------------------------------------------------------------------------------------------------
    def clusterData(self, dfData, kNum):
        try:
            kmeans_kwargs = {
            "init": "random",
            "n_init": 10,
            "max_iter": 1000,
            "random_state": 44,
            }

            # Num of clusters to use for Kmeans
            k = kNum
    
            # Kmeans model
            kmodel = KMeans(n_clusters=k, **kmeans_kwargs)
            # k is range of number of clusters.

            # https://www.scikit-yb.org/en/latest/api/cluster/elbow.html
            #visualizer = KElbowVisualizer(kmodel, k=(2,20), metric='calinski_harabasz', timings= True)
            #visualizer = KElbowVisualizer(kmodel, k=(2,20), metric='silhouette', timings= True)
            #visualizer = KElbowVisualizer(kmodel, k=(2,20), timings= True)
            visualizer = SilhouetteVisualizer(kmodel, colors='yellowbrick')
          
            #dfData.replace({-np.inf: -1_000_000, np.inf: 1_000_000}, inplace=True)
        
            # Repace -inf and inf (infinite vals) if they exist
            dfData[dfData == -inf] = -1_000_000
            dfData[dfData == inf] = 1_000_000

            # Fit the data to the visualizer
            visualizer.fit(dfData)      

            # Finalize and render the figure
            visualizer.show()     

            optimalK = 0
            score = 0
    
            if(type(visualizer) is SilhouetteVisualizer):
                optimalK = visualizer.n_clusters_
                score = visualizer.silhouette_score_
            else:
                optimalK = visualizer.elbow_value_
                score = visualizer.elbow_score_

            # Cluster data
            nClusters = optimalK
            clusters = AgglomerativeClustering(n_clusters=nClusters, affinity='euclidean', linkage='ward', compute_distances=True)
            clusters.fit(dfData)

            kargs = {'p':int(optimalK), 'truncate_mode':'level', 'max_d':int(90)}

            # Plot dendogram of clusters
            self.plot_dendrogram(clusters, **kargs)

            #return clusters
        except:
            print("An error occured during clustering.")

    #------------------------------------------------------------------------------------------------------
    # Function to plot a dendrogram
    #------------------------------------------------------------------------------------------------------
    def plot_dendrogram(self, model, **kwargs):
    
        max_d = kwargs.pop('max_d', None)
    
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)

        # Plot the corresponding dendrogram
        dg = dendrogram(linkage_matrix, **kwargs)
        #dg = dendrogram(linkage_matrix)
        if max_d:
            plt.axhline(y=max_d, c='k')
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('Sample index or (cluster size)')
        plt.ylabel('Distance')
        plt.show()
        plt.clf()


    # split a multivariate sequence into samples
    def split_sequences(self, sequences, n_steps):
        X= list()
        appendCount = 0
        maxLen = (len(sequences)/n_steps)
        beginSeq = 0

        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            #if end_ix > len(sequences):
            if appendCount >= maxLen:
                break
            # gather input and output parts of the pattern
            seq_x = sequences[beginSeq:end_ix, :]
            X.append(seq_x)
            appendCount += 1
            beginSeq = end_ix
        return array(X)

    #split a multivariate sequence into samples
    def split_sequences(self, sequences, y_vals, n_steps):
        X= list()
        Y= list()

        origSequences = sequences.copy()
        appendCount = 0
        maxLen = (len(sequences)/n_steps)
        beginSeq = 0
        for i in range(len(sequences)):
            vi = i
            # find the end of this pattern
            end_ix = beginSeq + n_steps
            # check if we are beyond the dataset
            #if end_ix > len(sequences):
            if appendCount >= maxLen:
                break
            # gather input and output parts of the pattern
            seq_x = sequences[beginSeq:end_ix, :]
            seq_y = y_vals[beginSeq]
            X.append(seq_x)
            Y.append(seq_y)
            appendCount += 1
            beginSeq = end_ix

        return array(X), array(Y)

    # Scale and normalize x data
    def scaleTemporalData(self, x_data, fitAndTransform=False):
        scalers = {}
        # Normalize and transform  data
        for i in range(x_data.shape[1]):
            scalers[i] = self.X_SCALAR
            if fitAndTransform:
                x_data[:, i, :] = scalers[i].fit_transform(x_data[:, i, :]) 
            else:
                x_data[:, i, :] = scalers[i].transform(x_data[:, i, :]) 
        return x_data


    # Add months to specified date
    def getDateFromMonthNumber(self, givenDate, addMonths):
        dateFormat = '%d/%m/%Y'
        dtObj = datetime.datetime.strptime(givenDate, dateFormat)
        futureDate = dtObj + relativedelta(months=addMonths)
        return futureDate


    def reshapeDataLSTM(self, df, col_uid, col_targetY, n_timesteps):
        dfFeatures = df.drop(col_targetY, axis=1)
        dfTarget = df[[col_targetY]]

        # Get reshaped values for 3D Tensor
        #n_samples = len(dfFeatures.index.get_level_values(col_uid).unique().tolist())
        n_samples = len(dfFeatures[col_uid].value_counts().tolist())
        n_features = dfFeatures.shape[1]

        # Reshape input array to 3D
        x = dfFeatures.to_numpy().reshape(n_samples, n_timesteps, n_features)
        return x