#======================================================================================================
# Program: IA Prediction Environmental Approach
# Date: Nov 29, 2021
# Developer: Kenneth Young
# Data Set: Kristian Provided Cytokine data
# Description:Neural Network aiming to predict IA from TEDDY data.
#======================================================================================================

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
from sklearn.impute import SimpleImputer, KNNImputer
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


from scipy.cluster.hierarchy import dendrogram, linkage
# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

import lime as lm
import matplotlib as mpl
import decimal
import lime
import lime.lime_tabular
import random 
import shap
from shap.plots._beeswarm import summary_legacy #fix for beeswarm plot with Kernel Explainer https://github.com/slundberg/shap/issues/1460
from shap.plots._waterfall import waterfall_legacy

import databricks as dbs
import seaborn as sns

# Test GPU availability and instantiate memory growth limitation if True:
if tf.test.is_gpu_available():
    print('GPU Available\n')
    limit_gpu()
else:
    print('Running on CPU')


def getT1DOnly(df):
    t1dOnlyDf = df.loc[df['T1D'] == 1]
    return t1dOnlyDf

def f(**kwargs):
    for k, v in kwargs.items():
        print(k, v)

def plot_dendrogram(model, **kwargs):
    
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
    plt.title('T1D Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Distance')
    plt.show()
    plt.clf()


# Define data split sizes
# Training / validation / test 
# Validation is split from traning after training and test data are split from main data
testSize = 0.10
validationSize = 0.15
x_scalar = StandardScaler()

filename = r'C:\Users\youngkg\Documents\fac-youngkg-teddy-MP229\ken_local\input\teddy_ia_pos_key_vars.txt'
#filename = r'C:\Users\youngkg\Documents\fac-youngkg-teddy-MP229\ken_local\input\teddy_ia_pos_key_vars_red.txt'

dfData = pd.read_csv(filename, sep='\t', lineterminator='\r', na_values=['(NA)', '?']).fillna(0)

strCols = dfData.columns[dfData.dtypes.eq('object')]
# Convert all string values to numeric
dfData[strCols] = dfData[strCols].apply(pd.to_numeric, errors='coerce')

# Get column names
data_top = dfData.head() 
    
# Remove non-predicting columns by name
dfData.drop(['SUBJECT_ID', 
            'REGISTRATION_ID',
            'cc', # removed from data on 10/04/2021
            'country', # removed from data on 10/04/2021
            'persist_gad_only', # removed from data on 10/04/2021
            'persist_miaa_only', # removed from data on 10/04/2021
            #'hla_5grps', # removed from data on 10/04/2021
            'dataset_date', 
            'BabyBirthType', 
            'family',
            'year_birth', 
            'month_birth', 
            'day_birth',
            'quarter_birth',
            'fdr_proband',
            'fdr_proband_screen',
            'Europe',
            'fdr_types',
            'fdr_screen_types',
            'persist_atrisk_dt',
            'persist_atrisk_age',
            'persist_atrisk_days_age',
            'persist_atrisk_visit',
            'multi_persist_atrisk_dt',
            'multi_persist_atrisk_age',
            'multi_persist_atrisk_days_age',
            'multi_persist_atrisk_visit',
            'persist_conf_first_combo',
            'mult_persist_conf_ab',
            #'T1D',
            'T1D_atrisk_visit',
            'T1D_atrisk_visit',
            'T1D_atrisk_dt',
            'T1D_atrisk_agemos',
            'T1D_atrisk_agedays',
            'time_mos_IA_to_T1D',
            'time_mos_IA_to_2IA',
            'time_mos_2IA_to_T1D',
            'hla_DR4',
            'hla_DR3',
            'hla_DR34',
            'married',
            'single_parent_household',
            'married_code',
            'parent_abode',
            'brst_fed_6mo_stop',
            'probiotics_by_28days',
            'solid_food_day',
            'solid_food_mos',
            'veg_solid_food_day',
            'veg_solid_food_mos',
            'animal_solid_food_day',
            'animal_solid_food_mos',
            'babysweightdg',
            'height_9mo',
            'weight_9mo',
            'diabetes',
            'momheight',
            'momweight_before',
            'momweight_end',
            'multiple_1yr',
            'filter_$'], inplace=True, axis = 1)
      

rowCount = dfData.shape[0]
colCount = dfData.shape[1]

dfData = dfData.replace(r'\\n',  ' ', regex=True)
# summarize first 5 rows
print(dfData.head(25))

dfData.replace('?', np.nan, inplace= True)

print(dfData.isnull().sum())

# summarize the number of rows with missing values for each column
for i in range(dfData.shape[0]):
	# count number of rows with missing values
	n_miss = dfData.iloc[i, :].isnull().sum()
	perc = n_miss / dfData.shape[0] * 100
	print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))

rc = rowCount

# Define imputer. Imputation transformer for completing missing values.
# If “most_frequent”, then replace missing using the most frequent value along each column. 
# Can be used with strings or numeric data. 
# If there is more than one such value, only the smallest is returned.
imputer = SimpleImputer(strategy='most_frequent')
idf = pd.DataFrame(imputer.fit_transform(dfData))
idf.columns = dfData.columns
idf.index = dfData.index

print(idf.isnull().sum())

x_data = idf.loc[:, idf.columns != 'T1D']
y = idf['T1D']
y_data = y.values
y_data = y_data.astype('int32')

# Get a dataset that only contains T1D individuals
dfT1DOnly = getT1DOnly(idf)
y_dfT1DOnly = dfT1DOnly['T1D'] 
# Remove T1D from dataset
dfT1DOnly = dfT1DOnly.loc[:, dfT1DOnly.columns != 'T1D']

def clusterT1DParticipants(t1dData):
    kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 1000,
    "random_state": 44,
    }
    # Num of clusters to use for Kmeans
    k = 10
    
    # Kmeans model
    kmodel = KMeans(n_clusters=k, **kmeans_kwargs)
    # k is range of number of clusters.

    # Clustering visualizers: Used to help select the optimal number of clusters
    # by fitting the model with a range of values for K.

    #visualizer = KElbowVisualizer(kmodel, k=(2,50), metric='calinski_harabasz', timings= True)
    visualizer = KElbowVisualizer(kmodel, k=(2,50), metric='silhouette', timings= True)
    #visualizer = KElbowVisualizer(kmodel, k=(2,50), timings= True)
    #visualizer = SilhouetteVisualizer(kmodel, colors='yellowbrick')
           
    # Fit the data to the visualizer
    visualizer.fit(t1dData)      
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

    # Cluster T1D data
    nClusters = optimalK
    clusterT1D = AgglomerativeClustering(n_clusters=nClusters, affinity='euclidean', linkage='ward', compute_distances=True)
    #clusterT1D.fit_predict(dfT1DOnly)
    clusterT1D.fit(t1dData)

    kargs = {'p':int(optimalK), 'truncate_mode':'level', 'max_d':int(40)}
    # Plot dendogram of clusters
    plot_dendrogram(clusterT1D, **kargs)

dfT1DOnly = x_scalar.fit_transform(dfT1DOnly)
clusterT1DParticipants(dfT1DOnly)

# Plot the clusters
#plt.figure(figsize=(10, 7))
#plt.scatter(dfT1DOnly, clusterT1D.labels_, cmap='rainbow')
#plt.show()
#plt.clf()


rSeed = 77
rndState = np.random.RandomState(rSeed)

# Use Synthetic Minority Oversampling Technique (SMOTE) data augumentation. 
# Involves developing predictive models on classification datasets that have a severe class imbalance.
# Transform the dataset with SMOTE. SMOTE is an oversampling technique and is very effective in handling 
# class imbalance. SMOTEEN combines undersampling technqies (ENN or Tomek) with oversampling of SMOTE, to 
# increase the effectiveness of handling imbalanced classes.
oversample = SMOTETomek(random_state=rndState)
x_dataS, y_dataS = oversample.fit_resample(x_data, y_data)

featureNames = (idf.loc[:, idf.columns != 'T1D']).columns.tolist()

# SPLIT DATASETS:
# Create the training, validation, and test data sets. This uses the scikit learn train_test_split function.
# This is a critical step Splitting your dataset is essential for an unbiased evaluation of prediction performance. 
# TRAINING SET: The training set is applied to train, or fit, the model.
# VALIDATION SET: The validation set is used for unbiased model evaluation during hyperparameter tuning. 
#   For example, when we want to find the optimal number of neurons in a neural network or the best kernel for a support vector machine, 
#   we experiment with different values. For each considered setting of hyperparameters, we fit the model with the training set and assess 
#   its performance with the validation set.
# TEST SET: The test set is needed for an unbiased evaluation of the final model. It is NOT used for fitting or validation.



# Some of the variables are categorical. So we have to transform those to numbers 
# and use MinMaxScaler (Normalize) or StandardScaler (Standardize) to scale down the values. 
# The neural network converges sooner when it exposes the same scaled features and gives better accuracy.

#x_scalar = MinMaxScaler(feature_range=(-1,1))
#x_scalar = StandardScaler()

# split data into a training set and test dataset
x_dataTrain, x_dataTest, y_dataTrain, y_dataTest = train_test_split(x_dataS, y_dataS, test_size = testSize, random_state=rndState)

# Normalize and transform training data
#x_dataTrain = x_scalar.fit_transform(x_dataTrain)

# split training set into training and validation dataset
x_dataTrain, x_dataValidation, y_dataTrain, y_dataValidation = train_test_split(x_dataTrain, y_dataTrain, test_size = validationSize, random_state=rndState)

# Normalize and transform training data
x_dataTrain = x_scalar.fit_transform(x_dataTrain)
# Normalize (standardize) validation data
x_dataValidation = x_scalar.transform(x_dataValidation)
# Normalize (standardize) test data
x_dataTest = x_scalar.transform(x_dataTest)

rowCountTrain = x_dataTrain.shape[0]
colCountTrain = x_dataTrain.shape[1]
rowCountValid = x_dataValidation.shape[0]
colCountValid = x_dataValidation.shape[1]
rowCountTest = x_dataTest.shape[0]
colCountTest = x_dataTest.shape[1]

# NOTE: Set plotDists to False if you do not want all of the plots and heatmap to show
# Running the plots will take time, so you can bypass to save time and get to the 
# neural network model compiliation faster.
plotDists = True

if plotDists:
    # Plot T1D Distribution
    f = sns.countplot(x='T1D', data=idf)
    f.set_title("T1D distribution")
    f.set_xticklabels(['No T1D', 'T1D'])
    plt.xlabel("");
    plt.show() 
    plt.clf()

    # Plot T1D Distribution by Gender
    f = sns.countplot(x='T1D', data=idf, hue='female')
    plt.legend(['Male', 'Female'])
    f.set_title("T1D disease by gender")
    f.set_xticklabels(['No T1D', 'T1D'])
    plt.xlabel("");
    plt.show() 
    plt.clf()

    #plt.close("all")
    #plt.figure(figsize=(16, 6))
    ## Store heatmap object in a variable to easily access it when you want to include more features (such as title).
    ## Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
    #heatmap = sns.heatmap(idf.corr(), vmin=-1, vmax=1, annot=True, fmt='.2f', annot_kws={'size': 3}, cmap='BrBG')
    ## Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
    #heatmap.set_title('T1D Predication Correlation Heatmap', fontdict={'fontsize':8}, pad=12);
    #plt.show()
    #plt.clf()

    # Plot T1D Distribution by HLA_5grps
    #f = sns.countplot(x='T1D', data=idf, hue='hla_5grps')
    #plt.legend(['','1','2','3','4','5'])
    #f.set_title("T1D disease by HLA 5 Groups")
    #f.set_xticklabels(['No T1D', 'T1D'])
    #plt.xlabel("");
    #plt.show() 
    #plt.clf()


    groupByHlaGrps = idf.groupby(['T1D', 'CTSH_yes_rs3825932_T', 'hla_5grps'])
    countHlaGrpByT1dRs3763305 = groupByHlaGrps.size().unstack()
    countHlaGrpByT1dRs3763305.plot(kind='bar', stacked=True, figsize=[16,6], colormap='winter')
    #plt.show()
    #plt.clf()

    df2 = idf.groupby(['T1D', 'CTSH_yes_rs3825932_T', 'hla_5grps'])['T1D'].count().unstack().fillna(0).plot(kind='bar', stacked=True)
    r = [0,1,2,3,4]
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel('Total')
    plt.title('Grouped by HLA groups, rs3763305_A, and T1D')
    plt.show() 
    plt.clf()

    
    # Plot CTSH_yes_rs3825932_T Distribution by HLA_5grps
    f = sns.countplot(x='CTSH_yes_rs3825932_T', data=idf, hue='hla_5grps')
    plt.legend(['','1','2','3','4','5'])
    f.set_title("CTSH_yes_rs3825932_T by HLA 5 Groups")
    f.set_xticklabels(['No CTSH_yes_rs3825932_T', 'CTSH_yes_rs3825932_T'])
    plt.xlabel("");
    plt.show() 
    plt.clf()

     # Plot T1D Distribution by CTSH_yes_rs3825932_T
    f = sns.countplot(x='T1D', data=idf, hue='CTSH_yes_rs3825932_T')
    plt.legend(['0','1'])
    f.set_title("T1D disease by CTSH_yes_rs3825932_T")
    f.set_xticklabels(['No T1D', 'T1D'])
    plt.xlabel("");
    plt.show() 
    plt.clf()

     
    # Plot INS_yes_rs1004446_A Distribution by HLA_5grps
    f = sns.countplot(x='INS_yes_rs1004446_A', data=idf, hue='hla_5grps')
    plt.legend(['','1','2','3','4','5'])
    f.set_title("INS_yes_rs1004446_A by HLA 5 Groups")
    f.set_xticklabels(['No INS_yes_rs1004446_A', 'INS_yes_rs1004446_A'])
    plt.xlabel("");
    plt.show() 
    plt.clf()

     # Plot T1D Distribution by INS_yes_rs1004446_A
    f = sns.countplot(x='T1D', data=idf, hue='INS_yes_rs1004446_A')
    plt.legend(['0','1'])
    f.set_title("T1D disease by INS_yes_rs1004446_A")
    f.set_xticklabels(['No T1D', 'T1D'])
    plt.xlabel("");
    plt.show() 
    plt.clf()


    # Plot BTNL2_yes_rs3763305_A Distribution by HLA_5grps
    f = sns.countplot(x='BTNL2_yes_rs3763305_A', data=idf, hue='hla_5grps')
    plt.legend(['','1','2','3','4','5'])
    f.set_title("BTNL2_yes_rs3763305_A by HLA 5 Groups")
    f.set_xticklabels(['No BTNL2_yes_rs3763305_A', 'BTNL2_yes_rs3763305_A'])
    plt.xlabel("");
    plt.show() 
    plt.clf()

    # Plot T1D Distribution by BTNL2_yes_rs3763305_A
    f = sns.countplot(x='T1D', data=idf, hue='BTNL2_yes_rs3763305_A')
    plt.legend(['0','1'])
    f.set_title("T1D disease by BTNL2_yes_rs3763305_A")
    f.set_xticklabels(['No T1D', 'T1D'])
    plt.xlabel("");
    plt.show() 
    plt.clf()

    #heat_map = sns.heatmap(idf.corr(method='pearson'), annot=True, fmt='.2f', linewidths=2)
    #heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=45);
    #plt.rcParams["figure.figsize"] = (colCountTrain,colCountTrain)
    #plt.show()
    #plt.clf()

#######################################################################################################################################
# Batching approaches:
# Batch Gradient Descent: Batch size is set to the total number of examples in the training dataset.
# Stochastic Gradient Descent: Batch size is set to one.
# Minibatch Gradient Descent: Batch size is set to more than one and less than the total number of examples in the training dataset.
#######################################################################################################################################
# Input settings for neural network
#######################################################################################################################################
modelUnits = 16
dropoutRate = 0.50
regWeight = 0.00001
batchSize = int(rowCountTrain*validationSize)
bufferSize = 16
epochsNum = 225
dataSize = rowCountTrain
stepsPerEpoch = int(dataSize / (batchSize * 1.5))

# Shape of Training data
inputs = tf.keras.Input(shape=(colCountTrain,))
inputs.shape
inputs.dtype

# Activation parameters
para_relu = PReLU()
leaky_relu = LeakyReLU(alpha=0.01)

##############################################################################################################################
# 
# Deep Learning Functions
#
##############################################################################################################################

# Function to train a deep learning model.
def deepLearningModel(model, x_train, y_train, x_valid, y_valid, num_epochs, batch_size):
     
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

    history = model.fit(x_train
                       , y_train
                       , epochs=num_epochs
                       , batch_size=batch_size
                       , validation_data=(x_valid, y_valid)
                       , verbose=0)
    return history

# Function to evaluate a trained model on a selecte metric
def evaluationMetric(model, history, metric_name, num_epochs):
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]
    e = range(1, num_epochs + 1)
    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.xlabel('Epoch number')
    plt.ylabel(metric_name)
    plt.title('Comparing training and validation ' + metric_name + ' for ' + model.name)
    plt.legend()
    plt.show()

# Function to test the model on new data after training the model 
# on the full training data with the optimal number of epochs
def testModel(model, x_train, y_train, x_test, y_test, num_epochs, batch_size):
    model.fit(x_train
              , y_train
              , epochs=num_epochs
              , batch_size=batch_size
              , verbose=0)
    results = model.evaluate(x_test, y_test)
    print()
    print('Test accuracy: {0:.2f}%'.format(results[1]*100))
    return results


def predictModel(model, x_test, batch_size):
    pred = model.predict(x_test, batch_size=batch_size, verbose=0, use_multiprocessing=False)
    return pred

# Function to compare a metric between two models 
def compareModelsByMetric(model_1, model_2, model_hist_1, model_hist_2, metric, num_epochs):
    metric_model_1 = model_hist_1.history[metric]
    metric_model_2 = model_hist_2.history[metric]
    e = range(1, num_epochs + 1)
    
    metrics_dict = {
        'acc' : 'Training Accuracy',
        'loss' : 'Training Loss',
        'val_acc' : 'Validation accuracy',
        'val_loss' : 'Validation loss'
    }
    
    metric_label = metrics_dict[metric]
    plt.plot(e, metric_model_1, 'bo', label=model_1.name)
    plt.plot(e, metric_model_2, 'b', label=model_2.name)
    plt.xlabel('Epoch number')
    plt.ylabel(metric_label)
    plt.title('Comparing ' + metric_label + ' between models')
    plt.legend()
    plt.show()

# Function to return the epoch number where the validation loss is at its minimum
def optimalEpoch(model_hist):
    min_epoch = np.argmin(model_hist.history['val_loss']) + 1
    print("Minimum validation loss reached in epoch {}".format(min_epoch))
    return min_epoch

##############################################################################################################################
# Deep Learning Functions
#
# Adjust model paramenters to avoid underfitting (high bias), overfitting (high variance). 
# Aiming for a good fit (low bias, low variance).
##############################################################################################################################

tf.keras.backend.clear_session()

t1dPredModel = tf.keras.models.Sequential()
t1dPredModel.add(tf.keras.layers.InputLayer(input_shape=[colCountTrain]))
t1dPredModel.add(layers.Dense(24, activation=leaky_relu, input_shape=[colCountTrain]))
t1dPredModel.add(layers.Dropout(dropoutRate))
t1dPredModel.add(layers.BatchNormalization())
t1dPredModel.add(layers.Dense(1, activation=leaky_relu))
t1dPredModel.add(layers.BatchNormalization())
t1dPredModel.add(layers.Dropout(dropoutRate))
t1dPredModel.add(layers.Dense(1, activation="sigmoid"))
#t1dPredModel.add(layers.Dense(1, kernel_initializer="glorot_uniform", activation="sigmoid"))

t1dPredModelHistory = deepLearningModel(t1dPredModel, x_dataTrain, y_dataTrain, x_dataValidation, y_dataValidation, epochsNum, batchSize)
# Get optimal number of epoch
t1dPredModelEpochOpt = optimalEpoch(t1dPredModelHistory)

# Get loss of model
evaluationMetric(t1dPredModel, t1dPredModelHistory, 'loss', epochsNum)
# Get accuracy of model
evaluationMetric(t1dPredModel, t1dPredModelHistory, 'binary_accuracy', epochsNum)

# Reduce batch size of test data if test data rows are smaller than current batch
if(batchSize > x_dataTest.shape[0]):
    batchSizeTest = x_dataTest.shape[0]
else:
    batchSizeTest = batchSize

# Evaluate the Neural Network Model with the test data
evalscore = testModel(t1dPredModel, x_dataTrain, y_dataTrain, x_dataTest, y_dataTest, epochsNum, batchSizeTest)

print("Test Loss", evalscore[0])
print("Test Accuracy", evalscore[1])

# Generate a predication of the test data
modelPrediction = predictModel(t1dPredModel, x_dataTest, batchSizeTest)
# Print the shape
print("Prediction Shape:", modelPrediction.shape)

y_modelPrediction = (modelPrediction > 0.5).astype("int32")
m = tf.keras.metrics.binary_accuracy(y_dataTest, y_modelPrediction, threshold=0.5)
predOutput = (m.numpy() > 0.5).astype("int32")

predOutputMatches = predOutput == y_dataTest
# Convert predicted output matches from "True, False" to "1, 0" Binary (Int32)
predOutputMatchesBin = predOutputMatches.astype("int32")
print("Which predictions match with binary labels:", predOutputMatches)

m = tf.keras.metrics.BinaryAccuracy()
m.update_state(y_dataTest, modelPrediction)
print("Binary Accuracy: ", m.result().numpy())

# Get the sum of accurately predicted records, count the true values (1)
accCount = sum(predOutput == y_dataTest)

# Correct number of predications
correct_predictions = sum(predOutputMatchesBin)
# Incorrect number of predictions
incorrect_predictions =  (rowCountTest - correct_predictions)
# Accuracy calculated
accuracyCalculated = (correct_predictions / rowCountTest)

f1Score = metrics.f1_score(y_dataTest, predOutput, average='binary')

print(correct_predictions," classified correctly")
print(incorrect_predictions," classified incorrectly")

# Generate a confusion matrix
cm = confusion_matrix(y_dataTest, predOutput)
print(cm)

def kernelExplainer(modelPred, testData):
    return shap.KernelExplainer(modelPred, testData)

def tabExplainer(trainingData):
  exp = lime.lime_tabular.LimeTabularExplainer(np.asarray(trainingData), feature_names=list(trainingData), class_names=['T1D'], verbose=True, mode='classification')
  return exp

def plotExplanationFeatures(explainer, modelPred, start, stop):
    expVals = explainer

    # TODO: Look into explainer... need to find out if this is a from the shap library.
    exp = explainer.explain_row()
    
    exp = exp.explain_instance(np.asarray(x_dataTest[start]), modelPred, num_features=len(featureNames))
    exPlot = exp.as_pyplot_figure()
    exPlot.plot()
    exPlot.show()
    
#explainer = lime.lime_tabular.LimeTabularExplainer(x_dataTrain, feature_names=list(x_dataTrain), class_names=[0, 1], mode='classification')
#exp = explainer.explain_instance(x_dataTest[2], modelKeras.predict, num_features=15, top_labels=1)
#exPlot = exp.as_pyplot_figure(label=1)
#exPlot.plot()
#exPlot.show()

# Per GitHub, needed to add this line in order to by pass a runtime error in SHAP.
# https://github.com/slundberg/shap/issues/1110
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough

# Use the first 100 training examples as our background dataset to integrate over
explainer = shap.KernelExplainer(t1dPredModel, x_dataTrain[:100])
#explainer = shap.KernelExplainer(t1dPredModel, x_dataTrain)
# explain the first n predictions
# explaining each prediction requires 2 * background dataset size runs
#shapValues = explainer.shap_values(x_dataTest[:80])
# all test batch rows
nSamples = batchSizeTest
#nSamples = 25
shapValues = explainer.shap_values(x_dataTest, nsamples=nSamples)

# summarize the effects of all the features
shap.plots._beeswarm.summary_legacy(shapValues, featureNames, show=False)
#shap.plots.beeswarm(explainer)
plt.show()
plt.clf()

# Plots an explantion of a single prediction as a waterfall plot.
# The SHAP value of a feature represents the impact of the evidence provided by that feature on the model’s output
shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shapValues[0])
plt.show()
plt.clf()


# plot the explanation of the first prediction
# Note the model is "multi-output" because it is rank-2 but only has one column
#shap.force_plot(explainer.expected_value[0], shapValues[0][0], x_dataTest[0])
#plt.show()
#plt.clf()

shap.summary_plot(shapValues[0], x_dataTest[0])
plt.show()
plt.clf()

shap.summary_plot(shapValues, x_dataTest, feature_names = featureNames)
plt.show()
plt.clf()




