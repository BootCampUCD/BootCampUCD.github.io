# Import dependencies
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
app.config["SQLALCHEMY_DATABASE_URI"] = "postgres:///db/project_2.sql"
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
    return render_template("index-p3-1.html")

    
# Run the application
if __name__ == "__main__":
    app.run()