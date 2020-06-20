

# Import Libraries
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
db = client.statesInfo_db
"It WORKS!!!"

# (pandas read in csv)
db.statesPop_db.drop()
df = pd.read_csv("./Data/data1.csv")
dfjson = json.loads(df.to_json(orient="records"))
db.statesPop_db.insert_many(dfjson)

# * * * ML Code * * *

# Retrieve and load data
data = pd.read_csv("/Data/data1.csv")
data_df = data.drop(["County", "Confirmed cases", "Confirmed Deaths"], axis=1)
data_df.head()

# describe data
data_df.describe()

# Min-Max normalizes/scales any list


def normalize(input_data):
    return ((np.array(input_data) - min(input_data)) / (max(input_data) - min(input_data)))


# Encode strings to integers using Label Encoding
le = LabelEncoder()
cols = ['State', 'FIPS', 'Population 2018', 'Median Household Income 2018 ($)', 'Unemployment Rate 2018 (%)', 'Poverty Rate 2018 (%)',
        'Confirmed cases per 100K people', 'Deaths per 100K people', 'Mortality Rate (%)', 'White (%)', 'Black (%)', 'Native American (%)', 'Asian (%)', 'Hispanic (%)', 'Dropout (%)', 'High School Diploma (%)', 'Some College_Associate Degree (%)', 'Bachelor Degree or Higher (%)']
for col in cols:
    data_df[col] = le.fit_transform(data_df[col])

data_df.head()

# Correlation among attributes
corr = data_df.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(corr, annot=True)
plt.show()

# Import StandardScaler to scale our continuous data
scaler = StandardScaler()
scaler.fit(data_df.drop('State', axis=1))
scaled_features = scaler.transform(data_df.drop("State", axis=1))
scaled_features
data_feat = pd.DataFrame(scaled_features, columns=data_df.columns[:-1])
data_feat.head()

# Importing train test split for splitting into train and test datasets

# Dividing predictors and predicted variables
X = data_feat
y = data_df['Deaths per 100K people']

# spliting the data into test (30 percent) and train sets (70 percent) with 101 random state
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

# Importing KNeighborsCLassifier

# Specifying a variable to KNeighborsClassifier
knc = KNeighborsClassifier()

# Fitting on the training dataset
knc.fit(X_train, y_train)

# Predicting on test dataset
pred = knc.predict(X_test)
pred

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

# Ploting and checking error rate for different neighbors
error_rate = []

for i in range(1, 40):
    knc = KNeighborsClassifier(n_neighbors=i)
    knc.fit(X_train, y_train)
    pred_i = knc.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate)
data_df.info()
sns.jointplot(x='Unemployment Rate 2018 (%)',
              y='Poverty Rate 2018 (%)', data=data_df)
unemp = pd.get_dummies(data_df['Unemployment Rate 2018 (%)'], drop_first=True)
feats = ['State']
final_data = pd.get_dummies(data_df, columns=feats, drop_first=True)
final_data.head()
X = final_data.drop('Unemployment Rate 2018 (%)', axis=1)
y - final_data['Unemployment Rate 2018 (%)']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
pred = dtree.predict(X_test)
print(classification_report(y_test, pred))
confusion_matrix(y_test, pred)
rtree = RandomForestClassifier()
rtree.fit(X_train, y_train)
prediction = rtree.predict(X_test)
print(classification_report(y_test, prediction))
confusion_matrix(y_test, prediction)


# END of ML Code

# identifying the path to the web page
@app.route("/")
def index():
    "Still WORKING!!!"
    object = list(db.statesInfo_db.find())
    return render_template("../HTML/index-FPD3-ML.html", object=object)

    # identifying the path to the web page


@app.route("/StatesPop")
def people(input1, input2, input3):
    object = list(db.statesPop_db.find())
    return dumps(object)
    # return render_template("../HTML/index-FPD3-ML.html", objectX=object)

    # display the web page index.html with the data
    # return render_template("index.html")


# starts the web server
if __name__ == "__main__":
    app.run(debug=True)
