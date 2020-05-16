# Import dependencies
import pandas as pd
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine,inspect
from flask import Flask, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy

# Define the app as a flask app
app = Flask(__name__)

#################################################
# Database Setup
#################################################
# Identify the database path
app.config["SQLALCHEMY_DATABASE_URI"] = "postgres:///db/project_2"
# Make it so that it doesn't track modifications
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Define database as SQLALCHEMY of the flask app
db = SQLAlchemy(app)

# reflect an existing database into a new model
Base = automap_base()
# reflect the tables

# Define the engine
rds_connection_string = "postgres:Abc*1234@127.0.0.1:5432/project_2"
engine = create_engine(f'postgres://{rds_connection_string}')
conn = engine.connect()
session = Session(engine)
Base.prepare(engine,reflect = True)
# ROUTE CREATION

# Define the home
@app.route("/")
def index():
    """Return the homepage."""
    return render_template("index-P3-1.html")

# Define the route to "/unemployment"
@app.route("/unemployment")
def unemployment():
    # Run a query to find unemployment data by year and state (ANY ANALYSIS CAN BE CODED HERE before RETURN)
    unemployment = pd.read_sql("SELECT * FROM unemployment", engine)
    return jsonify(unemployment.tolist())

# Define the route to "/unemployment_state"
@app.route("/unemployment_state")
def unemployment():
    # Run a query to find unemployment data by year and state (ANY ANALYSIS CAN BE CODED HERE before RETURN)
    unemployment_state = pd.read_sql("SELECT * FROM unemployment_state", engine)
    return jsonify(unemployment_state.tolist())

# Define the route to "/mortgage_interest"
@app.route("/mortgage_interest")
def mortgage_interest():
    # Run a query to find mortgage interest data by year and state (ANY ANALYSIS CAN BE CODED HERE before RETURN)
    mortgage_interest = pd.read_sql("SELECT * FROM mortgage_interest", engine)
    return jsonify(mortgage_interest.tolist())

# Define the route to "/GDP_states"
@app.route("/GDP_states")
def GDP_states():
    # Run a query to find GDP data by year and state (ANY ANALYSIS CAN BE CODED HERE before RETURN)
    GDP_states = pd.read_sql("SELECT * FROM GDP_states", engine)
    return jsonify(GDP_states.tolist())
    
# Define the route to "lat_lon"
@app.route("/lat_lon")
def lat_lon():
    # Run a query to find latitude and longitude data by year and state (ANY ANALYSIS CAN BE CODED HERE before RETURN)
    lat_lon = pd.read_sql("SELECT * FROM lat_lon", engine)
    return jsonify(lat_lon.tolist())

# Define the route to "state_pop"
@app.route("/state_pop")
def state_pop():
    # Run a query to find state population data in millions by year and state (ANY ANALYSIS CAN BE CODED HERE before RETURN)
    state_pop_in_m = pd.read_sql("SELECT * FROM state_pop", engine)
    return jsonify(state_pop.tolist())

# Define the route to "/state_pop_30m"
@app.route("/state_pop_30m")
def state_pop_30m():
    # Run a query to find state population data which are 30 million by year and state (ANY ANALYSIS CAN BE CODED HERE before RETURN)
    state_pop_30m = pd.read_sql("SELECT * FROM state_pop_30m", engine)
    return jsonify(state_pop_30m.tolist())

    
# Run the application
if __name__ == "__main__":
    app.run()