import sys
import argparse
from typing_extensions import Self
import pandas as pd
import numpy as np
from numpy import loadtxt

import matplotlib as mp
import matplotlib.pyplot as plt
import plotly.express as px # to plot the time series plot


import scipy
import sklearn as sk
from sklearn import metrics # for the evaluation
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split # for dataset splitting
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from skimage.segmentation import mark_boundaries
from sklearn.impute import SimpleImputer 
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

import tslearn
from tslearn.svm import TimeSeriesSVC

import xgboost
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.initializers import *
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import LeakyReLU, PReLU, Dense
from tensorflow.keras.constraints import max_norm, unit_norm, non_neg, min_max_norm
from scipy.cluster.hierarchy import dendrogram, linkage
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from yellowbrick.model_selection import FeatureImportances

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

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


class DeepLearning(object):
   
    className = ""
    batchSize = 32
    n_epochs = 500
    learning_rate = 0.001
    beta_1=0.8
    beta_2=0.9
    dropoutRate = 0.50
    alphaDropoutRate = 0.20
    gaussianDropoutRate = 0.50
    decayRate = learning_rate / n_epochs
    momentum = 0.9
    amsgrad=False

    def __init__(self):
        self.className = "DeepLearning"

    ##############################################################################################################################
    # 
    # Deep Learning Functions
    #
    ##############################################################################################################################

    # Function to train a deep learning model.
    def deepLearningModel(self, model, x_train, y_train, x_valid, y_valid, num_epochs, batch_size):
        #model.compile(optimizer='adam'
        #              , loss='binary_crossentropy'
        #              , metrics=['accuracy'])
    
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Accuracy()])

        history = model.fit(x_train
                           , y_train
                           , epochs=num_epochs
                           , batch_size=batch_size
                           , validation_data=(x_valid, y_valid)
                           , verbose=0)
        return history
    
    # Function to evaluate a trained model on a selecte metric
    def evaluationMetric(self, model, history, metric_name, num_epochs):
        metric = history.history[metric_name]
        val_metric = history.history['val_' + metric_name]
        e = range(1, num_epochs + 1)
        plt.clf()
        plt.plot(e, metric, 'bo', label='Train ' + metric_name)
        plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
        plt.xlabel('Epoch number')
        plt.ylabel(metric_name)
        plt.title('Comparing training and validation ' + metric_name + ' for ' + model.name)
        plt.legend()
        plt.show()

    # Function to test the model on new data after training the model 
    # on the full training data with the optimal number of epochs
    def testModel(self, model, x_train, y_train, x_test, y_test, num_epochs, batch_size):
        model.fit(x_train
                  , y_train
                  , epochs=num_epochs
                  , batch_size=batch_size
                  , verbose=0)
        results = model.evaluate(x_test, y_test)
        print()
        print('Test accuracy: {0:.2f}%'.format(results[1]*100))
        return results


    def predictModel(self, model, x_test, batch_size):
        pred = model.predict(x_test, batch_size=batch_size, verbose=0, use_multiprocessing=False)
        return pred

    def predictProbaModel(self, model, x_test, batch_size):
        pred = model.predict_proba(x_test, batch_size=batch_size, verbose=0, use_multiprocessing=False)
        return pred

    def evaluateModel(self, model, x_test, y_test, batch_size):
        eval = model.evaluate(x_test, y_test, batch_size=batch_size)
        return eval

    # Function to compare a metric between two models 
    def compareModelsByMetric(self, model_1, model_2, model_hist_1, model_hist_2, metric, num_epochs):
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
        plt.clf()
        plt.plot(e, metric_model_1, 'bo', label=model_1.name)
        plt.plot(e, metric_model_2, 'b', label=model_2.name)
        plt.xlabel('Epoch number')
        plt.ylabel(metric_label)
        plt.title('Comparing ' + metric_label + ' between models')
        plt.legend()
        plt.show()

    # Function to return the epoch number where the validation loss is at its minimum
    def optimalEpoch(self, model_hist):
        min_epoch = np.argmin(model_hist.history['val_loss']) + 1
        print("Minimum validation loss reached in epoch {}".format(min_epoch))
        return min_epoch

    # This model solves a regression model where the loss function is the 
    # linear least squares function and regularization is given by the l2-norm. 
    # Also known as Ridge Regression or Tikhonov regularization. 
    def ridgeRegression(self, x_train, x_val, y_train, y_val):
        model = Ridge(alpha=1e-2).fit(x_train, y_train)
        scr = model.score(x_train, y_train)
        print("Ridge Regression Score: " + str(scr))
        return model

    def xgboostTimeSeriesClassification(self, x_data, y_data, x_test, y_test, featureNames, topN):
        # fit model no training data
        model = XGBClassifier(learning_rate =0.5,
                 n_estimators=207,
                 max_depth=2,
                 min_child_weight=1,
                 gamma=0.8,
                 subsample=0.8,
                 colsample_bytree=0.8,
                 reg_alpha=0.001,
                 objective= 'binary:logistic',
                 nthread=4,
                 scale_pos_weight=1,
                 seed=27)
       
        model.fit(x_data, y_data)
         # set feature names
        model.get_booster().feature_names = featureNames

        # make predictions for test data
        y_pred = model.predict(x_test)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        # get importance
        importance = model.feature_importances_
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        plt.clf()
        plt.bar([x for x in range(len(importance))], importance)
        plt.show()

    def svcTimeSeriesClassification(self, x_data, y_data, x_test, y_test, featureNames, topN):
        model = TimeSeriesSVC(kernel="gak", gamma="auto", probability=True)
        #model = TimeSeriesSVC(kernel="linear", gamma="auto", probability=True)
        m = model.fit(x_data, y_data)
        m2 = m.predict(x_data)
        totalCorrect = sum(m2 == y_data)
        testScore = m.score(x_test, y_test)
        print("SVC time series correct classification rate on test data:", testScore)

        coefs = model.svm_estimator_.dual_coef_.flatten()
        # Zip coefficients and names together and make a DataFrame
        zipped = zip(featureNames, coefs)
        df = pd.DataFrame(zipped, columns=["feature", "value"])
        # Sort the features by the absolute value of their coefficient
        df["abs_value"] = df["value"].apply(lambda x: abs(x))
        df["colors"] = df["value"].apply(lambda x: "green" if x > 0 else "red")
        df = df.sort_values("abs_value", ascending=False)

        import seaborn as sns
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        sns.barplot(x="feature",
                    y="value",
                    data=df.head(topN),
                   palette=df.head(topN)["colors"])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=20)
        ax.set_title("Top Features", fontsize=25)
        ax.set_ylabel("Coef", fontsize=22)
        ax.set_xlabel("Feature Name", fontsize=22)
        plt.show()

        top_pos = np.argsort(model.svm_estimator_.dual_coef_.ravel())[:topN]
        top_neg = np.argsort(model.svm_estimator_.dual_coef_.ravel())[-topN:]
        
        top_coef = np.hstack([top_pos, top_neg])

        imp = top_coef
        names = featureNames
        imp,names = zip(*sorted(zip(imp,names)))
        plt.barh(range(len(names)), imp, align='center')
        plt.yticks(range(len(names)), names)
        plt.show()

        return names
        
    def svmClassificationFeatures(self, x_data, y_data, featureNames, topN):
        model = svm.LinearSVC()
        #model = svm.NuSVC(gamma="auto")
        return self.vizFeatureImportance(model, x_data, y_data, featureNames, topN)

    def randForestClassificationFeatures(self, nEstimators, x_data, y_data, featureNames, topN):
        model = RandomForestClassifier(n_estimators=nEstimators)
        return self.vizFeatureImportance(model, x_data, y_data, featureNames, topN)

    def vizFeatureImportance(self, model, x_data, y_data, featureNames, topN):
        viz = FeatureImportances(model, labels=featureNames, topn=topN, relative=False)
        viz.fit(x_data, y_data)
        viz.show()
        return list(viz.features_)


    def kBestFeatures(self, x_data, y_data, topN):
        selector = SelectKBest(f_classif, k=topN)
        selected_features = selector.fit_transform(x_data, y_data)
        column_names = x_data.columns[selector.get_support()]
        column_names = column_names.values.tolist()
        return column_names

    # If model score is significantly larger than the chance level, it is possible to use the permutation_importance 
    # function to probe which features are most predictive
    def getPermutationImportantFeatures(self, model, x_val, y_val, nRepeats, randomState, featureNames):
        pi = permutation_importance(model, x_val, y_val, n_repeats=nRepeats, random_state=randomState)
        #dfFeatures = pd.DataFrame(data=None, columns=['feature_name', 'importances_mean', 'importances_std'])
        rowIdx = 0
        for metric in pi:
            r = pi[metric]    
            for i in r.importances_mean.argsort()[::-1]:
           
                if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                    new_val = [(featureNames[metric])]
                    print(f"    {diabetes.feature_names[i]:<8}"
                    f"{r.importances_mean[i]:.3f}"
                    f" +/- {r.importances_std[i]:.3f}")
                    rowIdx += 1
        return pi

    # Transformer encoder
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-8)(inputs)
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-8)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    def transformer_model(self, n_timesteps, n_features, input_shape, static_input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0,autoEncoder=True):
        # Weights
        reg = tf.keras.regularizers.l1_l2(l1=0.00008, l2=0.00008)
        actReg = tf.keras.regularizers.l2(l2=0.000001)

        temporal_input = tf.keras.Input(shape=input_shape)
        temporal_input_x = temporal_input
        for _ in range(num_transformer_blocks):
            temporal_input_x = self.transformer_encoder(temporal_input_x, head_size, num_heads, ff_dim, dropout)

        temporal_input_x = layers.GlobalAveragePooling1D(data_format="channels_first")(temporal_input_x)

        for dim in mlp_units:
            temporal_input_x = layers.Dense(dim, activation="relu")(temporal_input_x)
            temporal_input_x = layers.Dropout(mlp_dropout)(temporal_input_x)



        # LSTM (Bidirectional LSTM Layers)
        lstm_input = tf.keras.Input(shape=(n_timesteps, n_features), name='lstm_input')

        if(autoEncoder == True):
            # Encoder
            lstm_encoder = Bidirectional(LSTM(16, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, kernel_initializer='glorot_uniform', dropout=self.dropoutRate, activity_regularizer=actReg, return_sequences=True))(lstm_input)
            lstm_encoder = Bidirectional(LSTM(6, activation='tanh',  recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, kernel_initializer='glorot_uniform', dropout=self.dropoutRate*1.00, activity_regularizer=actReg, return_sequences=False))(lstm_encoder)
                       
            #Decoder
            lstm_decoder = RepeatVector(n_timesteps)(lstm_encoder)
            lstm_decoder = Bidirectional(LSTM(6, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, dropout=self.dropoutRate*1.00, activity_regularizer=actReg, return_sequences=True))(lstm_decoder)
            lstm_decoder = Bidirectional(LSTM(16, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, dropout=self.dropoutRate, activity_regularizer=actReg, return_sequences=True))(lstm_decoder)
            
            # Time distrubuted layer
            lstm_input_x = TimeDistributed(Dense(n_timesteps, activation='tanh', kernel_initializer='glorot_uniform', activity_regularizer=actReg))(lstm_decoder)
            lstm_input_x = GaussianDropout(self.gaussianDropoutRate)(lstm_input_x)

            # Attention layer  if sequences are too long (used in combo with autoencoder)
            lstm_input_x = attention(return_sequences=False)(lstm_input_x)
            lstm_input_x = GaussianDropout(self.gaussianDropoutRate)(lstm_input_x)  
        else:
            lstm_input_x = Bidirectional(LSTM(16, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, kernel_initializer='glorot_uniform', dropout=self.dropoutRate, activity_regularizer=actReg, return_sequences=True))(lstm_input)
            lstm_input_x = Bidirectional(LSTM(6, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, kernel_initializer='glorot_uniform', dropout=self.dropoutRate*1.00, activity_regularizer=actReg, return_sequences=False))(lstm_input_x)
             # Time distrubuted layer
            lstm_input_x = TimeDistributed(Dense(n_timesteps, activation='tanh', recurrent_dropout=0, kernel_initializer='glorot_uniform', activity_regularizer=actReg))(lstm_input_x)
            lstm_input_x = GaussianDropout(self.gaussianDropoutRate)(lstm_input_x)         

        # Flatten 3D to 2D
        lstm_input_x = Flatten()(lstm_input_x)
        lstm_input_x = Dense(4, activation='tanh', kernel_initializer='glorot_uniform')(lstm_input_x)
        lstm_input_x = GaussianDropout(self.gaussianDropoutRate)(lstm_input_x)


         # Static Layers (Multi-Layer Perceptron MLP Layers)
        static_input = tf.keras.Input(shape=(static_input_shape), name='static_input')
        static_input_x = Dense(24, activation='selu', kernel_initializer=LecunNormal(), activity_regularizer=actReg, name='static_input_x')(static_input)
        static_input_x = GaussianDropout(self.gaussianDropoutRate)(static_input_x)
        static_input_x = Dense(16, activation='selu', kernel_initializer=LecunNormal())(static_input_x)
        static_input_x = GaussianDropout(self.gaussianDropoutRate)(static_input_x)
        static_input_x = Dense(4, activation='selu', kernel_initializer=LecunNormal())(static_input_x)
        static_input_x = GaussianDropout(self.gaussianDropoutRate)(static_input_x)

        # Combine layers - Transformer + MLP
        #combined_layers = Concatenate(axis= 1, name='conc_temporal_static')([temporal_input_x, lstm_input_x, static_input_x])

        # Combine layers - Transformer + MLP
        combined_layers = Concatenate(axis= 1, name='conc_temporal_static')([temporal_input_x, static_input_x])

        # Single layer output 
        output = Dense(1, activation='sigmoid', name='output')(combined_layers)
        
        opt = Adamax(learning_rate=self.learning_rate)
                    
        # Compile Model
        #model = tf.keras.Model(inputs=[temporal_input, lstm_input, static_input], outputs=[output])
        model = tf.keras.Model(inputs=[temporal_input, static_input], outputs=[output])
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy', 'accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        return model


    # define RNN (LSTM) model
    def biLstmModel(self, n_timesteps, n_features, staticShape, autoEncoder=True):
        # Weights
        reg = tf.keras.regularizers.l1_l2(l1=0.00008, l2=0.00008)
        actReg = tf.keras.regularizers.l2(l2=0.000001)
        #actReg = tf.keras.regularizers.l2(l2=0.000008)

        #recurrent_initializer="he_uniform",
        #recurrent_initializer="he_uniform",
        #kernel_initializer=tf.keras.initializers.VarianceScaling(), recurrent_initializer=tf.keras.initializers.RandomNormal()

        # LSTM (Bidirectional LSTM Layers)
        lstm_input = tf.keras.Input(shape=(n_timesteps, n_features), name='lstm_input')

        if(autoEncoder == True):
            # Encoder
            lstm_encoder = Bidirectional(LSTM(16, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, kernel_initializer='glorot_uniform', dropout=self.dropoutRate, activity_regularizer=actReg, return_sequences=True))(lstm_input)
            lstm_encoder = Bidirectional(LSTM(6, activation='tanh',  recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, kernel_initializer='glorot_uniform', dropout=self.dropoutRate*1.00, activity_regularizer=actReg, return_sequences=False))(lstm_encoder)
                       
            #Decoder
            lstm_decoder = RepeatVector(n_timesteps)(lstm_encoder)
            lstm_decoder = Bidirectional(LSTM(6, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, dropout=self.dropoutRate*1.00, activity_regularizer=actReg, return_sequences=True))(lstm_decoder)
            lstm_decoder = Bidirectional(LSTM(16, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, dropout=self.dropoutRate, activity_regularizer=actReg, return_sequences=True))(lstm_decoder)
            
            # Time distrubuted layer
            lstm_input_x = TimeDistributed(Dense(n_timesteps, activation='tanh', kernel_initializer='glorot_uniform', activity_regularizer=actReg))(lstm_decoder)
            lstm_input_x = GaussianDropout(self.gaussianDropoutRate)(lstm_input_x)

            # Attention layer  if sequences are too long (used in combo with autoencoder)
            lstm_input_x = attention(return_sequences=False)(lstm_input_x)
            lstm_input_x = GaussianDropout(self.gaussianDropoutRate)(lstm_input_x)  
        else:
            lstm_input_x = Bidirectional(LSTM(16, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, kernel_initializer='glorot_uniform', dropout=self.dropoutRate, activity_regularizer=actReg, return_sequences=True))(lstm_input)
            lstm_input_x = Bidirectional(LSTM(6, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, kernel_initializer='glorot_uniform', dropout=self.dropoutRate*1.00, activity_regularizer=actReg, return_sequences=False))(lstm_input_x)
             # Time distrubuted layer
            lstm_input_x = TimeDistributed(Dense(n_timesteps, activation='tanh', recurrent_dropout=0, kernel_initializer='glorot_uniform', activity_regularizer=actReg))(lstm_input_x)
            lstm_input_x = GaussianDropout(self.gaussianDropoutRate)(lstm_input_x)         

        # Flatten 3D to 2D
        lstm_input_x = Flatten()(lstm_input_x)
        lstm_input_x = Dense(4, activation='tanh', kernel_initializer='glorot_uniform')(lstm_input_x)
        lstm_input_x = GaussianDropout(self.gaussianDropoutRate)(lstm_input_x)
            
        # Static Layers (Multi-Layer Perceptron MLP Layers)
        static_input = tf.keras.Input(shape=(staticShape), name='static_input')
        static_input_x = Dense(24, activation='selu', kernel_initializer=LecunNormal(), activity_regularizer=actReg, name='static_input_x')(static_input)
        static_input_x = GaussianDropout(self.gaussianDropoutRate)(static_input_x)
        static_input_x = Dense(16, activation='selu', kernel_initializer=LecunNormal())(static_input_x)
        static_input_x = GaussianDropout(self.gaussianDropoutRate)(static_input_x)
        static_input_x = Dense(4, activation='selu', kernel_initializer=LecunNormal())(static_input_x)
        static_input_x = GaussianDropout(self.gaussianDropoutRate)(static_input_x)
                   
        # Combine layers - LSTM + MLP
        combined_layers = Concatenate(axis= 1, name='conc_temporal_static')([lstm_input_x, static_input_x])
        #combined_layers = GaussianDropout(gaussianDropoutRate)(combined_layers)
                     
        # Single layer output 
        output = Dense(1, activation='sigmoid', name='output')(combined_layers)
            
        # configure optimizer with gradient norm clipping; large updates to weights during training 
        # can cause a numerical overflow or underflow often referred to as “exploding gradients.” clip_norm or clipvalue
        #opt = Adamax(learning_rate=learning_rate, clipvalue=0.5)
        #opt = Adamax(learning_rate=learning_rate, clipnorm=2.0)
        opt = Adamax(learning_rate=self.learning_rate)
                    
        # Compile Model
        model = Model(inputs=[lstm_input, static_input], outputs=[output])
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy', 'accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        return model



# Add attention layer to the deep learning network
class attention(Layer):
    def __init__(self, return_sequences=True, **kwargs):
        self.return_sequences = return_sequences
        super(attention,self).__init__(**kwargs)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'return_sequences': self.return_sequences
        })
        return config
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        if(self.return_sequences):
            return context
        context = K.sum(context, axis=1)
        return context