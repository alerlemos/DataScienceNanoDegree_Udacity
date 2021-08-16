# Disaster Response Pipeline Project

## Overview of the project

The second projetct in the <a href='real messages that were sent during disaster events'> Udacity Data Science Nano Degree </a>required the development of a disaster response pipeline. To categorize each message, it was necessary to use NLP techniques and build a ML model.

This project was divided into 4 sections:
- Loading and cleaning the data;
- Creating the ML Pipeline;
- Evaluating the model obtained.

## Loading and cleaning the data
The data provided was diveded into two csv files: messages and categories.

The messages file contained real messages that were sent during disaster events while the categories file contained wich category the message belonged to. 
The files were loaded and then merged to obtain a single dataframe with all the information. The categories column was used to create other columns, each representing one category.

## Creating the ML Pipeline
In this part of the project, a pipeline was created to deal with all the transformations required to treat the data and make it usable for the development of the machine learning model. Here, the data was tokenized, normalized and inputted to the desired classifier (it was used the MultiOutputClassifier with RandomForest).

## Evaluating the model obtained
After the model was trained, i


## Libraries used:
- pandas
- scikit-learn
- sqlalchemy
- nltk
- pickle
- numpy
- flask
- joblib
- plotly
- json

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
