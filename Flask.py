

# Import Libraries

# Visualization
# training and testing data split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix  # for confusion matrix
from sklearn import metrics  # accuracy measure
from sklearn.tree import DecisionTreeClassifier  # Decision Tree
from sklearn.naive_bayes import GaussianNB  # Naive bayes
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from sklearn import svm  # support vector Machine
from sklearn.linear_model import LogisticRegression  # logistic regression
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron
from pprint import pprint
from bson.json_util import dumps
import pandas as pd
import math
import json
import pymongo
from flask import Flask, render_template, jsonify
import numpy as np
from numpy import *
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import random
import warnings
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
import scipy.stats as st
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, model_selection, tree, preprocessing, metrics, linear_model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.feature_selection import RFE, RFECV
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline

# Feature Selection and Encoding

# Machine learning

# Grid and Random Search

# Managing Warnings


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

# added for code on converting data to GeoJson format for mapping
db2 = client.covid1a_db

# * * * ML Code * * *

# Retrieve and load data
data = pd.read_csv("Data/data1.csv")

# *** PLTS REGRESS COVID ML CODE

# (ML CODE)

# Retrieve and load covid data

covid = pd.read_csv("covid.csv")
covid = covid.rename(columns={"Admin2": "County"})
covid["FIPS"].fillna(999999999999, inplace=True)
covid["FIPS"] = list(map(int, covid["FIPS"]))


# Clean files from government data
def govdata(file, index):
    data = pd.read_csv(file)
    data.columns = data.iloc[index]
    data = data.drop([i for i in range(index + 1)], axis=0)
    data.index -= index + 1
    data = data.rename(columns={"FIPStxt": "FIPS"})
    data["FIPS"] = list(map(int, data["FIPS"]))
    return data


# Retriev data files from USDA data
unemployment = govdata("unemployment.csv", 6)
poverty = govdata("poverty.csv", 3)
population = govdata("population.csv", 1)
education = govdata("education.csv", 3)


# Retrieve race data
race = pd.read_csv("racial.csv")

# List FIPS values

fips_list = list(race["FIPS"])

# All lists to be created from data sources
county_list, state_list, confirmed_list, death_list, mortality_list = [], [], [], [], []
white, black, native, asian, hispanic = [], [], [], [], []
income_list, unemp_list, poverty_list, pop_list = [], [], [], []
less_high, highschool, somecollege, bachelor = [], [], [], []
confirmed_pop_list, death_pop_list = [], []


# Loops through all FIPS values
for fips in fips_list:

    # Finds rows from each data frame with current FIPS value
    covid_row = covid.loc[covid["FIPS"] == fips]
    unemp_row = unemployment.loc[unemployment["FIPS"] == fips]
    poverty_row = poverty.loc[poverty["FIPS"] == fips]
    pop_row = population.loc[population["FIPS"] == fips]
    education_row = education.loc[education["FIPS"] == fips]
    race_row = race.loc[race["FIPS"] == fips]

    # Case info from Johns Hopkins research
    county_list.append(list(covid_row["County"])[0])
    state_list.append(list(covid_row["Province_State"])[0])
    if list(covid_row["Confirmed"])[0] == 0:
        mortality_list.append(0)
    else:
        mortality_list.append(
            list((covid_row["Deaths"] / covid_row["Confirmed"]) * 100)[0])
    confirmed_list.append(int(list(covid_row["Confirmed"])[0]))
    death_list.append(int(list(covid_row["Deaths"])[0]))

    confirmed_pop_list.append(list(covid_row["Confirmed"])[
                              0] / float(list(pop_row["POP_ESTIMATE_2018"])[0].replace(",", "")) * 100000)
    death_pop_list.append(list(covid_row["Deaths"])[
                          0] / float(list(pop_row["POP_ESTIMATE_2018"])[0].replace(",", "")) * 100000)

    # median household income and unemployment rate
    income_list.append(
        int(list(unemp_row["Median_Household_Income_2018"])[0].replace(",", "")))
    unemp_list.append(float(list(unemp_row["Unemployment_rate_2018"])[0]))

    # percent population under poverty line
    poverty_list.append(float(list(poverty_row["PCTPOVALL_2018"])[0]))

    # population
    pop_list.append(
        int(list(pop_row["POP_ESTIMATE_2018"])[0].replace(",", "")))

    # education
    less_high.append(float(list(
        education_row["Percent of adults with less than a high school diploma, 2014-18"])[0]))
    highschool.append(float(list(
        education_row["Percent of adults with a high school diploma only, 2014-18"])[0]))
    somecollege.append(float(list(
        education_row["Percent of adults completing some college or associate's degree, 2014-18"])[0]))
    bachelor.append(float(list(
        education_row["Percent of adults with a bachelor's degree or higher, 2014-18"])[0]))

    # Racial data
    white.append(float(race_row["White"]))
    black.append(float(race_row["Black"]))
    native.append(float(race_row["Native"]))
    asian.append(float(race_row["Asian"]))
    hispanic.append(float(race_row["Hispanic"]))

# Creates a new data frame from all of the lists
cleaned_data = pd.DataFrame({
    "State": state_list, "FIPS": fips_list, "County": county_list,
    "Population 2018": pop_list, "Median Household Income 2018 ($)": income_list,
    "Unemployment Rate 2018 (%)": unemp_list, "Poverty Rate 2018 (%)": poverty_list,
    "Confirmed cases": confirmed_list, "Confirmed Deaths": death_list,
    "Confirmed cases per 100K people": confirmed_pop_list,
    "Deaths per 100K people": death_pop_list, "Mortality Rate (%)": mortality_list,
    "White (%)": white, "Black (%)": black,
    "Native American (%)": native, "Asian (%)": asian,
    "Hispanic (%)": hispanic, "Dropout (%)": less_high,
    "High School Diploma (%)": highschool, "Some College/Associate's Degree (%)": somecollege,
    "Bachelor's Degree or Higher (%)": bachelor
})
cleaned_data.to_csv('data2.csv')

# Min-Max normalizes/scales any list


def normalize(input_data):
    return ((np.array(input_data) - min(input_data)) / (max(input_data) - min(input_data)))

# Create a chloropleth map based USA counties' map template


def plotMap(input_data):
    end_points = list(np.percentile(
        input_data, [5 + (5 * i) for i in range(19)]))
    i = 0
    while i < len(end_points):
        if end_points.count(end_points[i]) > 1 or end_points[i] == min(input_data) or end_points[i] == max(input_data):
            del end_points[i]
        else:
            i += 1

    colorScale, numColors = [], len(end_points) + 1
    for i in range(numColors):
        colorScale.append("rgb(" + str(200 - ((100 / (numColors - 1)) * i)) + ", " + str(251 - (
            (203 / (numColors - 1)) * i)) + ", " + str(255 - ((148 / (numColors - 1)) * i)) + ")")

    fig = ff.create_choropleth(fips=fips_list, values=input_data,
                               colorscale=colorScale, binning_endpoints=end_points)
    fig.layout.template = None

    title = 'USA COVID Mortality Rate %'
    legend_title = 'COVID Mortality Rate %'
    fig.show()


# Plot shows COVID-19 mortality rate by county
plotMap(mortality_list)


def plot_hbar(input_data, col, n, hover_data=[]):
    fig = px.bar(df.sort_values(col).tail(n),
                 x=col, y="State", color='green',
                 text=col, orientation='h', width=700, hover_data=hover_data,
                 color_discrete_sequence=px.colors.qualitative.Dark2)

    fig.show()


# Plot shows confirmed cases per 100K people by county
plotMap(confirmed_pop_list)

# Create a linear regression line


def regLine(x, y):
    coeff = np.polyfit(x, y, 1)
    return [((coeff[0] * x[i]) + coeff[1]) for i in range(len(x))]

# correlation coefficient


def coeff(x, y):
    return np.corrcoef(x, y)[0][1]


# Plotting the confirmed cases per 100k people against mortality rate
plt.rcParams["figure.figsize"] = (10, 7)

plt.xlabel("Confirmed cases per 100K people")
plt.ylabel("Mortality Rate (%)")
plt.scatter(confirmed_pop_list, mortality_list, 1)
plt.plot(confirmed_pop_list, regLine(confirmed_pop_list, mortality_list))
plt.show()

print("Correlation coefficient: " +
      str(coeff(confirmed_pop_list, mortality_list)))

# Plot shows deaths per 100K people by county
plotMap(death_pop_list)

# Plot shows poverty rate by county
plotMap(poverty_list)

# Plot shows unemployment rate by county
plotMap(unemp_list)

# plot to show correlation of unemployment rate against deaths per 100K people
plt.rcParams["figure.figsize"] = (8, 8)

x, y = (unemp_list), (death_pop_list)
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Deaths per 100K people")
plt.scatter(x, y)
plt.plot(x, regLine(x, y))
plt.show()

print("Correlation coefficient: " + str(coeff(x, y)))

# Plot shows median household income by county
plotMap(income_list)

# Show correlation of median household income with deaths per 100K people
plt.rcParams["figure.figsize"] = (8, 8)

x, y = (income_list), (death_pop_list)
plt.xlabel("Deaths per 100K people")
plt.ylabel("Unemployment Rate (%)")
plt.scatter(x, y)
plt.plot(x, regLine(x, y))
plt.show()

print("Correlation coefficient: " + str(coeff(x, y)))

# Plot correlation of races with deaths per 100K people
plt.rcParams["figure.figsize"] = (18, 8)

plt.subplot(1, 5, 1)
plt.xlabel("White (%)")
plt.ylabel("Deaths per 100K people")
plt.scatter(white, death_pop_list)
plt.plot(white, regLine(white, death_pop_list))

# show for Black
plt.subplot(1, 5, 2)
plt.xlabel("Black (%)")
plt.ylabel("Deaths per 100K people")
plt.scatter(black, death_pop_list)
plt.plot(black, regLine(black, death_pop_list))


plt.subplot(1, 5, 3)
plt.xlabel("Asian (%)")
plt.ylabel("Deaths per 100K people")
plt.scatter(asian, death_pop_list)
plt.plot(asian, regLine(asian, death_pop_list))


plt.subplot(1, 5, 4)
plt.xlabel("Native American (%)")
plt.ylabel("Deaths per 100K people")
plt.scatter(native, death_pop_list)
plt.plot(native, regLine(native, death_pop_list))


plt.subplot(1, 5, 5)
plt.xlabel("Hispanic (%)")
plt.ylabel("Deaths per 100K people")
plt.scatter(hispanic, death_pop_list)
plt.plot(hispanic, regLine(hispanic, death_pop_list))

plt.show()

print("Correlation coefficient of White: " + str(coeff(white, death_pop_list)))
print("Correlation coefficient of Black: " + str(coeff(black, death_pop_list)))
print("Correlation coefficient of Native American: " +
      str(coeff(native, death_pop_list)))
print("Correlation coefficient of Asian: " + str(coeff(asian, death_pop_list)))
print("Correlation coefficient of Hispanic: " +
      str(coeff(hispanic, death_pop_list)))


# Plot correlation of education with deaths per 100K people
plt.rcParams["figure.figsize"] = (18, 8)

plt.subplot(1, 4, 1)
plt.xlabel("Dropout (%)")
plt.ylabel("Deaths per 100K people")
plt.scatter(less_high, death_pop_list)
plt.plot(less_high, regLine(less_high, death_pop_list))


plt.subplot(1, 4, 2)
plt.xlabel("High School Diploma (%)")
plt.ylabel("Deaths per 100K people")
plt.scatter(highschool, death_pop_list)
plt.plot(highschool, regLine(highschool, death_pop_list))


plt.subplot(1, 4, 3)
plt.xlabel("Some College/Associate's Degree (%)")
plt.ylabel("Deaths per 100K people")
plt.scatter(somecollege, death_pop_list)
plt.plot(somecollege, regLine(somecollege, death_pop_list))


plt.subplot(1, 4, 4)
plt.xlabel("Bachelor's Degree or Higher (%)")
plt.ylabel("Deaths per 100K people")
plt.scatter(native, death_pop_list)
plt.plot(native, regLine(native, death_pop_list))

plt.show()

print("Correlation coefficient of Dropout: " +
      str(coeff(less_high, death_pop_list)))
print("Correlation coefficient of Highschool Diploma: " +
      str(coeff(highschool, death_pop_list)))
print("Correlation coefficient of Some College/Associate's Degree: " +
      str(coeff(somecollege, death_pop_list)))
print("Correlation coefficient of Bachelor's Degree: " +
      str(coeff(bachelor, death_pop_list)))

# Plotting relationships between economic factors
plt.rcParams["figure.figsize"] = (18, 8)

plt.subplot(1, 3, 1)
plt.xlabel("Median Household Income ($)")
plt.ylabel("Poverty Rate (%)")
plt.scatter(income_list, poverty_list)

plt.subplot(1, 3, 2)
plt.xlabel("Median Household Income ($)")
plt.ylabel("Unemployment Rate (%)")
plt.scatter(income_list, unemp_list)

plt.subplot(1, 3, 3)
plt.xlabel("Poverty Rate (%)")
plt.ylabel("Unemployment Rate (%)")
plt.scatter(poverty_list, unemp_list)

plt.show()

# Plotting other relationships (only significant ones plotted)

plt.rcParams["figure.figsize"] = (18, 28)

plt.subplot(4, 3, 1)
plt.xlabel("Poverty Rate (%)")
plt.ylabel("White (%)")
plt.scatter(poverty_list, white)

plt.subplot(4, 3, 2)
plt.xlabel("Poverty Rate (%)")
plt.ylabel("Black (%)")
plt.scatter(poverty_list, black)

plt.subplot(4, 3, 3)
plt.xlabel("Poverty Rate (%)")
plt.ylabel("Drop out (%)")
plt.scatter(poverty_list, less_high)

plt.subplot(4, 3, 4)
plt.xlabel("Poverty Rate (%)")
plt.ylabel("bachelor's degrees or higher (%)")
plt.scatter(poverty_list, bachelor)

plt.subplot(4, 3, 5)
plt.xlabel("Median Household Income ($)")
plt.ylabel("Asian (%)")
plt.scatter(income_list, asian)

plt.subplot(4, 3, 6)
plt.xlabel("Median Household Income ($)")
plt.ylabel("Drop out (%)")
plt.scatter(income_list, less_high)

plt.subplot(4, 3, 7)
plt.xlabel("Median Household Income ($)")
plt.ylabel("High school Diploma (%)")
plt.scatter(income_list, highschool)

plt.subplot(4, 3, 8)
plt.xlabel("Median Household Income ($)")
plt.ylabel("Bachelor's degrees or higher (%)")
plt.scatter(income_list, bachelor)

plt.subplot(4, 3, 9)
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Drop out (%)")
plt.scatter(unemp_list, less_high)

plt.subplot(4, 3, 10)
plt.xlabel("Asian (%)")
plt.ylabel("High school Diploma (%)")
plt.scatter(asian, highschool)

plt.subplot(4, 3, 11)
plt.xlabel("Asian (%)")
plt.ylabel("Bachelor's degrees or higher (%)")
plt.scatter(asian, bachelor)

plt.show()

# Plotting combined index against deaths per 100,000 people
plt.rcParams["figure.figsize"] = (16, 8)

allLists = [poverty_list, unemp_list, income_list, white, black,
    hispanic, asian, native, less_high, highschool, somecollege, bachelor]
allCoeff = []


# All 2959 counties combined index against natural log of number of deaths per 100,000 people
y = ma.log(death_pop_list)
allCoeff.clear()
for i in range(len(allLists)):
    allCoeff.append(coeff(normalize(allLists[i]), y))

combinedIndex = np.array([0 for i in range(len(y))])
for i in range(len(allLists)):
    combinedIndex = combinedIndex + (allCoeff[i] * normalize(allLists[i]))

plt.subplot(1, 2, 2)
plt.xlabel("Combined Socio-Economic Index")
plt.ylabel("Natural Log of COVID-19 Deaths per 100,000 people")
plt.title("Correlation coefficient for all lists: " +
          str(coeff(combinedIndex, y)))
plt.scatter(combinedIndex, y)
plt.plot(combinedIndex, regLine(combinedIndex, y))


plt.show()


# importing all the required ML packages

# Feature Selection and Encoding


df_encode = cleaned_data.apply(LabelEncoder().fit_transform)

drop_elements = ['State', 'FIPS', 'County', 'Confirmed cases',
    'Confirmed Deaths', 'Deaths per 100K people']
y = df_encode["Deaths per 100K people"]
X = df_encode.drop(drop_elements, axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2)

print("## 1.5. Correlation Matrix")
display(cleaned_data.corr())
print("we see that some columns are highly correlated.")

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
pca = PCA(n_components=None)
x_train_pca = pca.fit_transform(X_train_std)
a = pca.explained_variance_ratio_
a_running = a.cumsum()
a_running

# Classification Model
# Perceptron Method
ppn = Perceptron(eta0=1, random_state=1)
ppn.fit(X_train, y_train)


y_pred = ppn.predict(X_test)
accuracy_score(y_pred, y_test)

score_ppn = cross_val_score(ppn, X, y, cv=5)
score_ppn.mean()

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
# y_pred = gaussian.predict(X_test)
score_gaussian = gaussian.score(X_test, y_test)
print('The accuracy of Gaussian Naive Bayes is', score_gaussian)

# Support Vector Classifier (SVM/SVC)
svc = SVC(gamma=scale)
svc.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)
score_svc = svc.score(X_test, y_test)
print('The accuracy of SVC is', score_svc

svc_radical=svm.SVC(kernel='rbf', C=1, gamma=auto)
svc_radical.fit(X_train, y_train.values.ravel())
score_svc_radical=svc_radical.score(X_test, y_test)
print('The accuracy of Radical SVC Model is', score_svc_radical)

# Logistic Regression
logreg=LogisticRegression()
logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)
score_logreg=logreg.score(X_test, y_test)
print('The accuracy of the Logistic Regression is', score_logreg)

# Random Forest Classifier
randomforest=RandomForestClassifier()
randomforest.fit(X_train, y_train)
# y_pred = randomforest.predict(X_test)
score_randomforest=randomforest.score(X_test, y_test)
print('The accuracy of the Random Forest Model is', score_randomforest)


# K-Nearest Neighbors
knn=KNeighborsClassifier()
knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
score_knn=knn.score(X_test, y_test)
print('The accuracy of the KNN Model is', score_knn)

# cross validation
from sklearn.model_selection import KFold  # for K-fold cross validation
from sklearn.model_selection import cross_val_score  # score evaluation
from sklearn.model_selection import cross_val_predict  # prediction
# k=10, split the data into 10 equal parts
kfold=KFold(n_splits=10, random_state=22)
xyz=[]
accuracy=[]
std=[]
classifiers=['Naive Bayes', 'Linear Svm', 'Radial Svm',
    'Logistic Regression', 'Decision Tree', 'KNN', 'Random Forest']
models=[GaussianNB(), svm.SVC(kernel='linear'), svm.SVC(kernel='rbf'), LogisticRegression(), DecisionTreeClassifier(),
        KNeighborsClassifier(n_neighbors=9), RandomForestClassifier(n_estimators=100)]
for i in models:
    model=i
    cv_result=cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
models_dataframe=pd.DataFrame({'CV Mean': xyz, 'Std': std}, index=classifiers)
models_dataframe











# *** END   PLTS REGRESS COVID ML CODE


# *** COVID W SOCIO ECONOMIC ML CODE
data_df=data.drop(["County", "Confirmed cases", "Confirmed Deaths"], axis=1)

# Let’s plot the distribution of each feature


def plot_distribution(data_df, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    plt.style.use('seaborn-whitegrid')
    fig=plt.figure(figsize=(width, height))
    fig.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=wspace, hspace=hspace)
    rows=math.ceil(float(data_df.shape[1]) / cols)
    for i, column in enumerate(data_df.columns):
        ax=fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if data_df.dtypes[column] == np.object:
            g=sns.countplot(y=column, data=data_df)
            substrings=[s.get_text()[:18] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)
            plt.xticks(rotation=25)
        else:
            g=sns.distplot(data_df[column])
            plt.xticks(rotation=25)


plot_distribution(data_df, cols=3, width=20,
                  height=20, hspace=0.45, wspace=0.5)
plt.savefig("Images/ml_features.png")

# Feature Encoding: Machine Learning algorithms perform Linear Algebra on Matrices, which means all features need have numeric values. The process of converting Categorical Features into values is called Encoding. Let's perform both One-Hot and Label encoding.

# Min-Max normalizes/scales any list


def normalize(input_data):
    return ((np.array(input_data) - min(input_data)) / (max(input_data) - min(input_data)))


# One Hot Encodes all labels before Machine Learning
one_hot_cols=data_df.columns.tolist()
one_hot_cols.remove('State')
data_enc=pd.get_dummies(data_df, columns=one_hot_cols)

# Encode strings to integers using Label Encoding
le=LabelEncoder()
cols=['State', 'FIPS', 'Population_2018', 'Median Household Income 2018 ($)', 'Unemployment Rate 2018 (%)', 'Poverty Rate 2018 (%)',
        'Confirmed cases per 100K people', 'Deaths per 100K people', 'Mortality Rate (%)', 'White (%)', 'Black (%)', 'Native American (%)', 'Asian (%)', 'Hispanic (%)', 'Dropout (%)', 'High School Diploma (%)', 'Some College_Associate Degree (%)', 'Bachelor Degree or Higher (%)']
for col in cols:
    data_df[col]=le.fit_transform(data_df[col])

# Correlation among attributes
corr=data_df.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(corr, annot=True)
plt.savefig("Images/corr_attr.png")


# Feature Importance: Random forest consists of a number of decision trees. Every node in the decision trees is a condition on a single feature, designed to split the dataset into two so that similar response values end up in the same set. The measure based on which the (locally) optimal condition is chosen is called impurity. When training a tree, it can be computed how much each feature decreases the weighted impurity in a tree. For a forest, the impurity decrease from each feature can be averaged and the features are ranked according to this measure. This is the feature importance measure exposed in sklearn’s Random Forest implementations.

# Using Random Forest to gain an insight on Feature Importance
feats=RandomForestClassifier()
feats.fit(data_df.drop('State', axis=1), data_df['State'])

plt.style.use('seaborn-whitegrid')
importance=feats.feature_importances_
importance=pd.DataFrame(importance, index=data_df.drop(
    'State', axis=1).columns, columns=["Importance"])
importance.sort_values(by='Importance', ascending=True).plot(
    kind='barh', figsize=(20, len(importance)/2))
plt.savefig("Images/random_forest_feat.png")


# PCA: Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. This transformation is defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components.

# We can use PCA to reduce the number of features to use in our ML algorithms, and graphing the variance gives us an idea of how many features we really need to represent our dataset fully.

# PCA's components graphed in 2D and 3D
# Apply Scaling
std_scaling=preprocessing.StandardScaler().fit(data_df.drop('State', axis=1))
X=std_scaling.transform(data_df.drop('State', axis=1))
y=data_df['State']

# Formatting
targets=[0, 1]
colors=['blue', 'red']
lw=2
alpha=0.3
# 2 Components PCA
plt.style.use('seaborn-whitegrid')
plt.figure(2, figsize=(20, 8))

plt.subplot(1, 2, 1)
pca=PCA(n_components=2)
X_r=pca.fit(X).transform(X)
for color, i, target in zip(colors, [0, 1], targets):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1],
                color=color,
                alpha=alpha,
                lw=lw,
                label=target)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('1st 2 PCA directions')

# 3 Components PCA
ax=plt.subplot(1, 2, 2, projection='3d')

pca=PCA(n_components=3)
X_reduced=pca.fit(X).transform(X)
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

selected_data=data_df

# Splitting Data into Training and Testing Datasets: We need to split the data back into the training and testing datasets.

# Splitting the Training and Test data sets
train=selected_data.loc[0:2959, :]
test=selected_data.loc[17:, :]

# Removing Samples with Missing data: We could have removed rows with missing data during feature cleaning, but we're choosing to do it at this point. It's easier to do it this way, right after we split the data into Training and Testing. Otherwise we would have had to keep track of the number of deleted rows in our data and take that into account when deciding on a splitting boundary for our joined data.

# Given missing fields are a small percentange of the overall dataset,
# we have chosen to delete them.
train=train.dropna(axis=0)
test=test.dropna(axis=0)

# Rename datasets before we conduct machine learning algorithims
X_train_w_label=train
X_train=train.drop(['State'], axis=1)
y_train=train['State'].astype('int64')
X_test=test.drop(['State'], axis=1)
y_test=test['State'].astype('int64')

# Machine Learning Algorithms: Data Review: Let's take one last peek at our data before we start running the Machine Learning algorithms.
X_train.shape

# Setting a random seed will guarantee we get the same results
# every time we run our training and testing.


random.seed(1)


# The following algorithms are used:

# KNN Logistic Regression Random Forest Naive Bayes Decision Tree
# Function that runs the requested algorithm and returns the accuracy metrics
def fit_ml_algo(algo, X_train, y_train, X_test, cv):
    model=algo.fit(X_train, y_train)
    test_pred=model.predict(X_test)
    if (isinstance(algo, (LogisticRegression,
                          KNeighborsClassifier,
                          GaussianNB,
                          DecisionTreeClassifier,
                          RandomForestClassifier))):
        probs=model.predict_proba(X_test)[:, 1]
    else:
        probs="Not Available"
    acc=round(model.score(X_test, y_test) * 100, 2)
    train_pred=model_selection.cross_val_predict(algo,
                                                   X_train,
                                                   y_train,
                                                   cv=cv,
                                                   n_jobs=-1)
    acc_cv=round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
    return train_pred, test_pred, acc, acc_cv, probs


model=LogisticRegression(solver='liblinear', C=0.05,
                           multi_class='ovr', random_state=0)

model.fit(X_train, y_train)


# ????LogisticRegression(C=0.05, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=100, multi_class='ovr', n_jobs=None, penalty='l2', random_state=0, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

# Logistic Regression - Random Search for Hyperparameters
def report(results, n_top=5):
    for i in range(1, n_top + 1):
        samples=np.flatnonzero(results['rank_test_score'] == i)
        for sample in samples:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][sample],
                  results['std_test_score'][sample]))
            print("Parameters: {0}".format(results['params'][sample]))


# Logistic Regression
train_pred_log, test_pred_log, acc_log, acc_cv_log, probs_log=fit_ml_algo(
    LogisticRegression(n_jobs=-1), X_train, y_train, X_test, 10)


# K-Nearest Neighbors
train_pred_knn, test_pred_knn, acc_knn, acc_cv_knn, probs_knn=fit_ml_algo(
    KNeighborsClassifier(n_neighbors=3, n_jobs=-1), X_train, y_train, X_test, 10)

# Specifying a variable to KNeighborsClassifier
knc=KNeighborsClassifier()

# Ploting and checking error rate for different neighbors
error_rate=[]

for i in range(1, 40):
    knc=KNeighborsClassifier(n_neighbors=i)
    knc.fit(X_train, y_train)
    pred_i=knc.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate)
plt.savefig("Images/KNN_err.png")

# Decision Tree Classifier
train_pred_dt, test_pred_dt, acc_dt, acc_cv_dt, probs_dt=fit_ml_algo(
    DecisionTreeClassifier(), X_train, y_train, X_test, 10)

# Random Forest Classifier - Random Search for Hyperparameters

# Utility function to report best scores


def report(results, n_top=5):
    for i in range(1, n_top + 1):
        samples=np.flatnonzero(results['rank_test_score'] == i)
        for sample in samples:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][sample],
                  results['std_test_score'][sample]))
            print("Parameters: {0}".format(results['params'][sample]))


# Random Forest Classifier
rfc=RandomForestClassifier(n_estimators=10, min_samples_leaf=2,
                             min_samples_split=17, criterion='gini', max_features=8)
train_pred_rf, test_pred_rf, acc_rf, acc_cv_rf, probs_rf=fit_ml_algo(
    rfc, X_train, y_train, X_test, 50)

# END of ML Code

# identifying the path to the web page
@app.route("/")
def index():
    "Still WORKING!!!"
    object=list(db.covid_new.find())
    return render_template("indexFP.html", object=object)

# identifying the path to the web page


@app.route("/COVID", methods=["GET"])
def names():
    object=list(db.covid1a_db.find())
    return dumps(object)
    # return render_template("indexPF.html", object=object)

    # display the web page index.html with the data
    # return render_template("index.html")


# working on converting to GeoJson format for plotting on a map
# @app.route('/x', methods=['GET'])
# def toGeojson():
#     stateData = []
#     for name in db2.find({"State: }):
#         stateData.append({
#             "type": "Feature",
#             "State": {
#                 "type": "LineString",
#             }
#         })
#     return dumps(stateData)
    # return jsonify(points)


# starts the web server
if __name__ == "__main__":
    app.run(debug=True)
