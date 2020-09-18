import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads the messages and categories datasets and merge them together.
    
    Returns:
     df (pandas dataframe): merged datasets. 
    """
    
    messages = pd.read_csv('disaster_messages.csv')
    #messages.head()
    #load categories
    categories = pd.read_csv('disaster_categories.csv')
    print('shape of disaster messages: ', messages.shape)
    print('shape of disaster categories: ', categories.shape)

    #merge the two datasets
    df = pd.merge(messages, categories, on='id')
    
    print('shape of merged data: ', df.shape)

    return df


def clean_data(df):
    """
    Cleans dataset and extracts classification categories.
    
    Parameters: 
         df (pandas dataframe): dataset containing messages and categories. 
          
    Returns: 
         df (pandas dataframe): cleaned dataset.
    """
    
    # create a dataframe of the individual category columns
    categories = df.categories.str.split(pat = ';',expand = True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[ : -2] )
    print(category_colnames)
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
    # set each value to be the last character of the string
      categories[column] = categories[column].apply(lambda x: x[-1] if int(x[-1]) < 2 else 1)
    
    # convert column from string to numeric
      categories[column] = pd.to_numeric(categories[column])

    df = df.drop(['categories'], axis=1)
    # concatenate the original dataframe with the new `categories` datafram
    df = pd.concat([df, categories],join='inner', axis=1)

    #remove duplicates
    df.drop_duplicates(inplace=True)

    return df



def save_data(df, database_filename):
    """Save the clean dataset into an sqlite database in the provided path"""
    
    database = 'sqlite:///' + database_filename
    engine = create_engine(database)
    df.to_sql('disaster_response', engine, index=False, if_exists = 'replace')  


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
