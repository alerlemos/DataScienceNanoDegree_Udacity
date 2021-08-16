import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Info:
        This function reads two csv files (messages and categories) and returns
        a DataFrame with the information of each file
    ----------
    Input:
        messages_filepath: path to the message file (type: String)
        categories_filepath: path to the categories file (type: String)
    ----------
    Output:
        df: Dataframe with the information about the messages and categories (type: pandas DataFrame)
    '''

    # Reading the messages csv
    messages = pd.read_csv(messages_filepath)

    # Reading the categories csv
    categories = pd.read_csv(categories_filepath)

    # Merging the two datasets using the id column
    df = messages.merge(categories)

    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)

    # Select the first row of the categories dataframe
    row = categories.iloc[0]

    # Use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x : x[0:(len(x) - 2)])

    # Renaming the columns of `categories`
    categories.columns = category_colnames

    # Converting category values to just numbers 0 or 1
    for column in categories:

        # Setting each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # Converting column from string to numeric
        categories[column] = categories[column].astype(int)

    # Replacing the categories column in df with new category columns
    df = pd.concat([df, categories], axis = 1)

    return df


def clean_data(df):
    '''
    Info:
        This function cleans the dataframe
    ----------
    Input:
        df: Dataframe with the data (type: pandas DataFrame)
    ----------
    Output:
        df: Dataframe with the clean data (type: pandas DataFrame)
    '''
    # Dropping the original categories column from `df`
    df.drop(columns = 'categories', inplace = True)

    # Removing duplicates
    df.drop_duplicates(inplace = True)

    return df


def save_data(df, database_filename):
    '''
    Info:
        This function saves the data into an sqlite database
    ----------
    Input:
        df: Dataframe with the data (type: pandas DataFrame)
        database_filename: Path where the data will be saved (type: String)
    ----------
    Output:
        None
    '''

    # Creating the engine
    engine = create_engine(database_filename)

    # Saving the data
    df.to_sql('Data_clean', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
