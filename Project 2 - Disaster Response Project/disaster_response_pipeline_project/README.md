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
The files were loaded and then merged to obtain a single dataframe with all the information. The categories column was used to create other columns, each representing one category. The data was stored in a SQL databased using the sqlalchemy library.

## Creating the ML Pipeline
In this part of the project, a pipeline was created to deal with all the transformations required to treat the data and make it usable for the development of the machine learning model. Here, the data was loaded from the SQL table, tokenized, normalized and inputted to the desired classifier (it was used the MultiOutputClassifier with RandomForest).

## Evaluating the model obtained
After the model was trained, it was evaluated to see how well it performed when dealing with data that was not used in the training stage.


## Libraries used:
- <a href = 'https://pandas.pydata.org'>pandas</a>
- <a href = 'https://scikit-learn.org/stable/'>scikit-learn</a>
- <a href = 'https://www.sqlalchemy.org'>sqlalchemy</a>
- <a href = 'https://www.nltk.org'>nltk</a>
- <a href = 'https://docs.python.org/3/library/pickle.html'>pickle</a>
- <a href = 'https://numpy.org'>numpy</a>
- <a href = 'https://flask.palletsprojects.com/en/2.0.x/'>flask</a>
- <a href = 'https://joblib.readthedocs.io/en/latest/'>joblib</a>
- <a href = 'https://plotly.com'>plotly</a>
- <a href = 'https://docs.python.org/3/library/json.html'>json</a>

## Instructions:
1. Clone the git repository:
```
git clone https://github.com/alerlemos/DataScienceNanoDegree_Udacity.git
```
2. Go to the folder of the project: 'Project 2 - Disaster Response Project/disaster_response_pipeline_project/'

3. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

4. Run the following command in the app's directory to run your web app.
    `python run.py`

5. Go to http://0.0.0.0:3001/

## Preview of the program

## Authors
<a href = 'https://github.com/alerlemos'>Alexandre Rosseto Lemos</a>

## Acknowledgements
* [Udacity](https://www.udacity.com/) for providing this amazing Data Science Nanodegree Program
* [Figure Eight](https://www.figure-eight.com/) for providing the relevant dataset to train the model
