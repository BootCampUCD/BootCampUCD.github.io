

# Import Libraries

# Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Feature Selection and Encoding
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

# Machine learning
from sklearn import datasets, model_selection, tree, preprocessing, metrics, linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, SGDClassifier

# Grid and Random Search
import scipy.stats as st
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Managing Warnings
import warnings

import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from numpy import *
import numpy as np
from flask import Flask, render_template
import pymongo
import json
import math

import pandas as pd
from bson.json_util import dumps
from pprint import pprint

# 1 we change sql to mongo
# 2 we change this file to connect using pymongo and sql alchemy
# dont connnect database in the js file.

# flask app
app = Flask(__name__)

# create a mongodb connection
conn = "mongodb://localhost:27017"

# assign it to a label
client = pymongo.MongoClient(conn)

# connect to the database
db = client.covid_new
"It WORKS!!!"

# (pandas read in csv)
db.covid1a_db.drop()
df = pd.read_csv("Data/data1.csv")
dfjson = json.loads(df.to_json(orient="records"))
db.covid1a_db.insert_many(dfjson)

# * * * ML Code * * *

# Retrieve and load data

data = pd.read_csv("Data/data1.csv")
data_df = data.drop(["County", "Confirmed cases", "Confirmed Deaths"], axis=1)

# Let’s plot the distribution of each feature


def plot_distribution(data_df, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(data_df.shape[1]) / cols)
    for i, column in enumerate(data_df.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if data_df.dtypes[column] == np.object:
            g = sns.countplot(y=column, data=data_df)
            substrings = [s.get_text()[:18] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)
            plt.xticks(rotation=25)
        else:
            g = sns.distplot(data_df[column])
            plt.xticks(rotation=25)


plot_distribution(data_df, cols=3, width=20,
                  height=20, hspace=0.45, wspace=0.5)
plt.savefig("Images/ml_features.png")

# Feature Encoding: Machine Learning algorithms perform Linear Algebra on Matrices, which means all features need have numeric values. The process of converting Categorical Features into values is called Encoding. Let's perform both One-Hot and Label encoding.

# Min-Max normalizes/scales any list


def normalize(input_data):
    return ((np.array(input_data) - min(input_data)) / (max(input_data) - min(input_data)))


# One Hot Encodes all labels before Machine Learning
one_hot_cols = data_df.columns.tolist()
one_hot_cols.remove('State')
data_enc = pd.get_dummies(data_df, columns=one_hot_cols)

# Encode strings to integers using Label Encoding
le = LabelEncoder()
cols = ['State', 'FIPS', 'Population 2018', 'Median Household Income 2018 ($)', 'Unemployment Rate 2018 (%)', 'Poverty Rate 2018 (%)',
        'Confirmed cases per 100K people', 'Deaths per 100K people', 'Mortality Rate (%)', 'White (%)', 'Black (%)', 'Native American (%)', 'Asian (%)', 'Hispanic (%)', 'Dropout (%)', 'High School Diploma (%)', 'Some College_Associate Degree (%)', 'Bachelor Degree or Higher (%)']
for col in cols:
    data_df[col] = le.fit_transform(data_df[col])

# Correlation among attributes
corr = data_df.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(corr, annot=True)
plt.savefig("Images/corr_attr.png")


# Feature Importance: Random forest consists of a number of decision trees. Every node in the decision trees is a condition on a single feature, designed to split the dataset into two so that similar response values end up in the same set. The measure based on which the (locally) optimal condition is chosen is called impurity. When training a tree, it can be computed how much each feature decreases the weighted impurity in a tree. For a forest, the impurity decrease from each feature can be averaged and the features are ranked according to this measure. This is the feature importance measure exposed in sklearn’s Random Forest implementations.

# Using Random Forest to gain an insight on Feature Importance
feats = RandomForestClassifier()
feats.fit(data_df.drop('State', axis=1), data_df['State'])

plt.style.use('seaborn-whitegrid')
importance = feats.feature_importances_
importance = pd.DataFrame(importance, index=data_df.drop(
    'State', axis=1).columns, columns=["Importance"])
importance.sort_values(by='Importance', ascending=True).plot(
    kind='barh', figsize=(20, len(importance)/2))
plt.savefig("Images/random_forest_feat.png")


# PCA: Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. This transformation is defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components.

# We can use PCA to reduce the number of features to use in our ML algorithms, and graphing the variance gives us an idea of how many features we really need to represent our dataset fully.

# PCA's components graphed in 2D and 3D
# Apply Scaling
std_scaling = preprocessing.StandardScaler().fit(data_df.drop('State', axis=1))
X = std_scaling.transform(data_df.drop('State', axis=1))
y = data_df['State']

# Formatting
targets = [0, 1]
colors = ['blue', 'red']
lw = 2
alpha = 0.3
# 2 Components PCA
plt.style.use('seaborn-whitegrid')
plt.figure(2, figsize=(20, 8))

plt.subplot(1, 2, 1)
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
for color, i, target in zip(colors, [0, 1], targets):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1],
                color=color,
                alpha=alpha,
                lw=lw,
                label=target)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('1st 2 PCA directions')

# 3 Components PCA
ax = plt.subplot(1, 2, 2, projection='3d')

pca = PCA(n_components=3)
X_reduced = pca.fit(X).transform(X)
for color, i, target_name in zip(colors, [0, 1], targets):
    ax.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1], X_reduced[y == i, 2],
               color=color,
               alpha=alpha,
               lw=lw,
               label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
ax.set_title("1st 3 PCA directions")
ax.set_xlabel("1st eigenvector")
ax.set_ylabel("2nd eigenvector")
ax.set_zlabel("3rd eigenvector")

# rotate the axes
ax.view_init(30, 10)
plt.savefig("Images/PCA.png")

# OPTIONS:
# - data_enc
# - data_df

# Change the dataset to test how would the algorithms perform under a differently encoded dataset.

selected_data = data_df

# Splitting Data into Training and Testing Datasets: We need to split the data back into the training and testing datasets.

# Splitting the Training and Test data sets
train = selected_data.loc[0:2959, :]
test = selected_data.loc[17:, :]

# Removing Samples with Missing data: We could have removed rows with missing data during feature cleaning, but we're choosing to do it at this point. It's easier to do it this way, right after we split the data into Training and Testing. Otherwise we would have had to keep track of the number of deleted rows in our data and take that into account when deciding on a splitting boundary for our joined data.

# Given missing fields are a small percentange of the overall dataset,
# we have chosen to delete them.
train = train.dropna(axis=0)
test = test.dropna(axis=0)

# Rename datasets before we conduct machine learning algorithims
X_train_w_label = train
X_train = train.drop(['State'], axis=1)
y_train = train['State'].astype('int64')
X_test = test.drop(['State'], axis=1)
y_test = test['State'].astype('int64')

# Machine Learning Algorithms: Data Review: Let's take one last peek at our data before we start running the Machine Learning algorithms.
X_train.shape

# Setting a random seed will guarantee we get the same results
# every time we run our training and testing.


random.seed(1)


# The following algorithms are used:

# KNN Logistic Regression Random Forest Naive Bayes Decision Tree
# Function that runs the requested algorithm and returns the accuracy metrics
def fit_ml_algo(algo, X_train, y_train, X_test, cv):
    model = algo.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    if (isinstance(algo, (LogisticRegression,
                          KNeighborsClassifier,
                          GaussianNB,
                          DecisionTreeClassifier,
                          RandomForestClassifier))):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = "Not Available"
    acc = round(model.score(X_test, y_test) * 100, 2)
    train_pred = model_selection.cross_val_predict(algo,
                                                   X_train,
                                                   y_train,
                                                   cv=cv,
                                                   n_jobs=-1)
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
    return train_pred, test_pred, acc, acc_cv, probs


model = LogisticRegression(solver='liblinear', C=0.05,
                           multi_class='ovr', random_state=0)

model.fit(X_train, y_train)


# ????LogisticRegression(C=0.05, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=100, multi_class='ovr', n_jobs=None, penalty='l2', random_state=0, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

# Logistic Regression - Random Search for Hyperparameters
def report(results, n_top=5):
    for i in range(1, n_top + 1):
        samples = np.flatnonzero(results['rank_test_score'] == i)
        for sample in samples:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][sample],
                  results['std_test_score'][sample]))
            print("Parameters: {0}".format(results['params'][sample]))


# Logistic Regression
train_pred_log, test_pred_log, acc_log, acc_cv_log, probs_log = fit_ml_algo(
    LogisticRegression(n_jobs=-1), X_train, y_train, X_test, 10)


# K-Nearest Neighbors
train_pred_knn, test_pred_knn, acc_knn, acc_cv_knn, probs_knn = fit_ml_algo(
    KNeighborsClassifier(n_neighbors=3, n_jobs=-1), X_train, y_train, X_test, 10)

# Specifying a variable to KNeighborsClassifier
knc = KNeighborsClassifier()

# Ploting and checking error rate for different neighbors
error_rate = []

for i in range(1, 40):
    knc = KNeighborsClassifier(n_neighbors=i)
    knc.fit(X_train, y_train)
    pred_i = knc.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate)
plt.savefig("Images/KNN_err.png")

# Decision Tree Classifier
train_pred_dt, test_pred_dt, acc_dt, acc_cv_dt, probs_dt = fit_ml_algo(
    DecisionTreeClassifier(), X_train, y_train, X_test, 10)

# Random Forest Classifier - Random Search for Hyperparameters

# Utility function to report best scores


def report(results, n_top=5):
    for i in range(1, n_top + 1):
        samples = np.flatnonzero(results['rank_test_score'] == i)
        for sample in samples:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][sample],
                  results['std_test_score'][sample]))
            print("Parameters: {0}".format(results['params'][sample]))


# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=10, min_samples_leaf=2,
                             min_samples_split=17, criterion='gini', max_features=8)
train_pred_rf, test_pred_rf, acc_rf, acc_cv_rf, probs_rf = fit_ml_algo(
    rfc, X_train, y_train, X_test, 50)

# END of ML Code

# identifying the path to the web page
@app.route("/")
def index():
    "Still WORKING!!!"
    object = list(db.covid_new.find())
    return render_template("indexFP.html", object=object)

    # identifying the path to the web page


@app.route("/COVID")
def people(input1):
    # object = list(db.classDB.find())
    object = list(db.covid1a_db.find("State String"))
    # return "does THIS work???"
    return dumps(object)
    # return render_template("../HTML/index-FPD3-ML.html", objectX=object)

    # display the web page index.html with the data
    # return render_template("index.html")


# starts the web server
if __name__ == "__main__":
    app.run(debug=True)
