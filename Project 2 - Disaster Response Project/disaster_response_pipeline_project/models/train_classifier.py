import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import pickle
from sklearn.metrics import accuracy_score,precision_score,recall_score
import numpy as np

def load_data(database_filepath):
    '''
    Info:
        This function reads the data from the sql table and returns the features
        and label columns
    ----------
    Input:
        database_filepath: path to the table containing the data (type: String)
    ----------
    Output:
        X: Dataframe with the features (type: pandas DataFrame)
        X: Dataframe with the labels (type: pandas DataFrame)
        category_names: Names of the categories (type: list)
    '''
    # Creating the path to the db file
    path  = 'sqlite:///' + database_filepath
    # Creating the engine
    engine = create_engine(path)

    # Reading the data from the sql table
    df = pd.read_sql_table('Data_clean', engine)

    # Obtaining the columns names
    columns = list(df.columns)

    # Splitting the data into training and test sets
    X_columns = ['message', 'original', 'genre']
    Y_columns = []
    for col in columns:
        if col in X_columns:
            pass
        else:
            Y_columns.append(col)

    Y_columns.remove('id')

    X = df['message']
    Y = df[Y_columns]

    # Obtaining the names of the categories
    category_names = list(Y.columns)

    return X, Y, category_names



def tokenize(text):
    '''
    Info:
        This function reads a text, tokenize and normalize it
    ----------
    Input:
        text: Text to be treated (type: String)
    ----------
    Output:
        clean_tokens: Text after the tokenizing and normalization proccess
    '''
    # Tokenizing the text
    tokens = word_tokenize(text)

    # Initializing the Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Obtaining the clean tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Info:
        This function creates a pipeline model
    ----------
    Input:
        None
    ----------
    Output:
        pipeline: Pipeline created
    '''
    # Building the pipeline model
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Info:
        This function evaluates the model creted. Then, it uses grid GridSearch
        to improve the results by finding better parameters.
    ----------
    Input:
        model: Model to be saved
        X_test: Path where the model will be saved
        Y_test:
        category_names:
    ----------
    Output:
        None
    '''
    # Making the first prediction
    y_pred1 = model.predict(X_test)

    columns = list(Y_test.columns)

    y_pred_1 = pd.DataFrame(data = y_pred1, columns = columns)

    # Obtaining the first resutls
    acc_list = []
    prec_list = []
    rec_list = []
    for col in columns:
        acc_list.append(accuracy_score(Y_test[col], y_pred_1[col]))
        prec_list.append(precision_score(Y_test[col], y_pred_1[col], average = 'macro'))
        rec_list.append(recall_score(Y_test[col], y_pred_1[col], average = 'macro'))

    acc1 = np.mean(acc_list)
    prec1 = np.mean(prec_list)
    rec1 = np.mean(rec_list)
    print(f'Model without GridSearch:\nAccuracy = {acc1}\nPrecision = {prec1}\nRecall = {rec1}')


def save_model(model, model_filepath):
    '''
    Info:
        This function saves the trained model
    ----------
    Input:
        model: Model to be saved
        model_filepath: Path where the model will be saved
    ----------
    Output:
        None
    '''
    # Saving the model
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
