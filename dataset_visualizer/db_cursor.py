""" Database Connection details"""

import mysql.connector
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

mydb = mysql.connector.connect(
    host="10.18.0.1",
    user="engine211",
    password="mmfinal",
    database='NEWS_ARTICLES'
)
my_cursor = mydb.cursor(dictionary=True)


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://engine211:mmfinal@10.18.0.1/NEWS_ARTICLES'
app.config['SECRET_KEY'] = 'mysecret'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

db = SQLAlchemy(app=app)

