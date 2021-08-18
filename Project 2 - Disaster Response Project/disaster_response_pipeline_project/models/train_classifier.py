import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
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
nltk.download('stopwords') 
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

    # Removing stop words
    tokens = [t for t in tokens if t not in stopwords.words("english")]

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

    # Performing grid search to optimize the parameters
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
    }

    cv = GridSearchCV(pipeline, param_grid = parameters)

    return cv


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
    # Making the prediction
    y_pred = model.predict(X_test)

    # Obtaining the columns names
    columns = list(Y_test.columns)

    # Creating a dataframe with the predictions
    df_y_pred = pd.DataFrame(data = y_pred, columns = columns)

    # Obtaining the f1 score, precision and recall for each category
    for col in columns:
        print(f'Classification results for column: {col}')
        print(classification_report(Y_test[col], df_y_pred[col]))


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
