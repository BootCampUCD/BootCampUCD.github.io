


# Import Libraries
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
df = pd.read_csv("./Data/P3-State-Lat-Long.csv")
dfjson = json.loads(df.to_json(orient="records"))
db.statesPop_db.insert_many(dfjson)


# identifying the path to the web page
@app.route("/")
def index():
    "Still WORKING!!!"
    object = list(db.statesInfo_db.find())
    return render_template("index-P3-1.html", object=object)

    # identifying the path to the web page


@app.route("/StatesPop")
def people():
    object = list(db.statesPop_db.find())
    return dumps(object)
    # return render_template("index-P3-2.html", objectX=object)

    # display the web page index.html with the data
    # return render_template("index.html")


# starts the web server
if __name__ == "__main__":
    app.run(debug=True)
