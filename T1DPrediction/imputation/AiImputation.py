import sys
import datetime
from datetime import timedelta
import argparse
import pandas as pd
import numpy as np
from numpy import array
from numpy import inf
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from enum import Enum

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

class ImputationMethods(Enum):
    SIMPLEIMPUTE = 1 
    FILLIMPUTE = 2
    REMOVEIMPUTE = 3
    LOCFIMPUTE = 4
    INTERPOLATE = 5

class Imputate(object):
    
    ######################################################################################################################################
    # This function imputes missing timepoints (observations)
    # based on due_num with incremental values of 3.
    # For example: 3 month visit, 6 month, 9 month...
    # Uses backfill and forward fill imputation methods:
    # Last observation carried forward (LOCF)
    # Next observation carried backward (NOCB)
    # These methods rely on the assumption that adjacent observations are similar to one another (no presence of strong seasonality)
    ######################################################################################################################################
    def imputeMissingTimepoints(self, dfd, numTimepoints, timepointStart, timePointIncrement, uid):
        df = dfd.copy()
        uidIds = df[uid].value_counts().keys().tolist()
        counts = df[uid].value_counts().tolist()
        countRemoved = 0
        dueNumStart = timepointStart
        maxDueNum = int(timePointIncrement * numTimepoints)

        # Impute records that do not have enough timepoints
        # For now, replace with next row or previous row
        # depending on if it exists
        for(id, count) in zip(uidIds, counts):
            rows = df.loc[df[uid] == id]
            n_missingTimepoints = int(numTimepoints - count)

            if n_missingTimepoints > 0:
                x = 0
                rows[0:].due_num.values[0]
                dueNum = dueNumStart
                prevDueNum = dueNum
                prevRow = None
                newRow = None
                nRows = len(rows)
                lRows = len(rows)

                # loop through number of timepoints
                while x < numTimepoints:
                    # Is the due num greater than the max due num? This is bad
                    if dueNum > maxDueNum:
                        bad = True

                    # is the row at this index null?
                    if x >= lRows:
                         # create new row from previous row
                         newRow = prevRow
                         # set due num on new row
                         newRow.due_num = dueNum
                         # Add new row to data frame
                         df = df.append(newRow, ignore_index=True)
                         # Increment number of rows    
                         nRows += 1 
                         # Set previous row to new row
                         prevRow = newRow
                    else:
                        currDueNum = int(rows.iloc[x].due_num)
                        prevRow = rows.iloc[x]
                        # Is the current due number of row not in correct position
                        if currDueNum != dueNum:
                           # create new row from previous row
                           newRow = prevRow
                           # set due num on new row
                           newRow.due_num = dueNum
                           # Add new row to data frame
                           df = df.append(newRow, ignore_index=True)
                           # Increment number of rows    
                           nRows += 1  
                           # Set previous row to new row
                           prevRow = newRow
                           # Set due num to currDueNum to move to next due number in order
                           # This was causing a bug, needed to comment out. The due num is auto incremented by 3
                           #dueNum = currDueNum
                    # Do the number of nows now equal the number of timepoints?
                    if nRows == numTimepoints:
                        break
                    # Increment due number by 3
                    dueNum += timePointIncrement     
                    
                    # Increment x    
                    x += 1  
            if n_missingTimepoints < 0:
                df.drop(rows.index, inplace=True)
                countRemoved += 1
          
        return df

    ##############################################################################
    # Function to perform the simple "most frequent" imputation
    ##############################################################################
    def simpleImputation(self, dfData, outcome): 
        # Define imputer. Imputation transformer for completing missing values.
        # If most_frequent, then replace missing using the most frequent value along each column. 
        # Can be used with strings or numeric data. 
        # If there is more than one such value, only the smallest is returned.

        imputer = SimpleImputer(strategy='most_frequent')
        idf = pd.DataFrame(imputer.fit_transform(dfData))

        idf.columns = dfData.columns
        
        print(idf.isnull().sum())

        x_data = idf.loc[:, idf.columns != outcome]
        y = idf[outcome]
        y_data = y.values
        y_data = y_data.astype('int32')

        return idf, x_data, y_data

    ##############################################################################
    # Function to perform the simple "median" imputation
    ##############################################################################
    def simpleImputation(self, dfData): 
        # Define imputer. Imputation transformer for completing missing values.
        # If most_frequent, then replace missing using the most frequent value along each column. 
        # Can be used with strings or numeric data. 
        # If there is more than one such value, only the smallest is returned.
        dfData.replace('?',np.NaN,inplace=True)
        #dfData = dfData.apply(pd.to_numeric, errors='coerce') # convert all columns of DataFrame to numeric
      
       
        imp=SimpleImputer(missing_values=np.NaN, strategy='median')
        idf=pd.DataFrame(imp.fit_transform(dfData))
        idf.columns=dfData.columns
        idf.index=dfData.index
        return idf

       
    ##############################################################################
    # This function removes (imputes) missing timepoints (observations)
    # based on due_num with incremental values of 3.
    # For example: 3 month visit, 6 month, 9 month...
    # Could introduce bias based on compliance, recommend using imputeMissingTimepoints
    ##############################################################################
    def filterTimepoints(self, dfd, numTimepoints, uid):
        df = dfd.copy()
        uidIds = df[uid].value_counts().keys().tolist()
        counts = df[uid].value_counts().tolist()
        countRemoved = 0
        # Remove records that do not have enough timepoints
        for(id, count) in zip(uidIds, counts):
            #idxs = df[df["registration_id"] == registrationId].index
            rows = df.loc[df[uid] == id]
            if count != numTimepoints:
                df.drop(rows.index, inplace=True)
                countRemoved += 1
          
        return df

    """
    interpolation: Clairvoyance: A Pipeline Toolkit for Medical Time Series. 
    Daniel Jarrett, Jinsung Yoon, Ioana Bica, Zhaozhi Qian, Ari Ercole, and Mihaela van der Schaar (2021). 
    Clairvoyance: A Pipeline Toolkit for Medical Time Series. 
    In International Conference on Learning Representations. Available at: https://openreview.net/forum?id=xnC8YwKUE3k.
    """
    def interpolation(self, x, t, imputation_model_name):
        """Interpolate temporal features.
        Args:
            x: temporal features to be interpolated
            t: time information
            imputation_model_name: temporal imputation model (e.g. linear, quadratic, cubic, spline)
        Returns:
            x: interpolated temporal features
        """
        # Parameters
        no, seq_len, dim = x.shape

        # For each patient temporal feature
        for i in tqdm(range(no)):

            temp_x = x[i, :, :]
            temp_t = t[i, :, 0]
            # Only for non-padded data
            idx_x = np.where(temp_x[:, 0] != -1)[0]

            temp_t_hat = temp_t[idx_x]

            # (1) Linear
            if imputation_model_name == "linear":
                temp_x_hat = temp_x[idx_x, :]
                # Convert data type to Dataframe
                temp_x_hat = pd.DataFrame(temp_x_hat)
                # Set time to index for interpolation
                temp_x_hat.index = temp_t_hat

                temp_x_hat = temp_x_hat.interpolate(method=imputation_model_name, limit_direction="both")
                x[i, idx_x, :] = np.asarray(temp_x_hat)

            # (2) Spline, Quadratic, Cubic
            elif imputation_model_name in ["spline", "quadratic", "cubic"]:

                for j in range(dim):
                    temp_x_hat = temp_x[idx_x, j]
                    # Convert data type to Dataframe
                    temp_x_hat = pd.DataFrame(temp_x_hat)
                    # Set time to index for interpolation
                    temp_x_hat.index = temp_t_hat
                    # Interpolate missing values
                    # Spline
                    if imputation_model_name == "spline":
                        if len(idx_x) - temp_x_hat.isna().sum()[0] > 3:
                            temp_x_hat = temp_x_hat.interpolate(
                                method=imputation_model_name, order=3, fill_value="extrapolate"
                            )
                        else:
                            temp_x_hat = temp_x_hat.interpolate(method="linear", limit_direction="both")
                    # Quadratic
                    elif imputation_model_name == "quadratic":
                        if len(idx_x) - temp_x_hat.isna().sum()[0] > 2:
                            temp_x_hat = temp_x_hat.interpolate(method=imputation_model_name, fill_value="extrapolate")
                        else:
                            temp_x_hat = temp_x_hat.interpolate(method="linear", limit_direction="both")
                    # Cubic
                    elif imputation_model_name == "cubic":
                        if len(idx_x) - temp_x_hat.isna().sum()[0] > 3:
                            temp_x_hat = temp_x_hat.interpolate(method=imputation_model_name, fill_value="extrapolate")
                        else:
                            temp_x_hat = temp_x_hat.interpolate(method="linear", limit_direction="both")

                    x[i, idx_x, j] = np.asarray(temp_x_hat)[0]

        return x

"""Interpolation on temporal data
- modes: 'linear', 'quadratic', 'cubic', 'spline'
- Reference: pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
"""
class Interpolate(object):

    def interpolateMissingTimepoints(self, df, n_start, n_timesteps, n_steps, n_freq, time_index, group_by):
        
        def f(x):
            r = pd.date_range(start=start, end=end, freq=n_freq)
            return x.reindex(r).interpolate(method='linear', limit_direction='both')

         # Create a new date
        rsDate = pd.to_datetime('19700101', format='%Y%m%d', errors='coerce')

        start = rsDate + timedelta(days=1)
        end = rsDate + timedelta(days=n_timesteps)

        # Create new column 'ti' from time_index (due_num)
        dfImputed = (df.assign(ti=rsDate + pd.to_timedelta(((df[time_index]-n_start)/n_steps)+1, unit='D')) 
                    .set_index('ti')
                    # Apply resampling for every n_steps time indexes 
                    .groupby([group_by]).apply(f)
                    # Reset index (ungroups data)
                    .reset_index(group_by, drop=True)
                    # Reset index on data set
                    .reset_index()
                )

        dfImputed.drop('index', 1, inplace=True)

        return dfImputed



