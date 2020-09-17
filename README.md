# Disaster Response Pipeline Project
## Introduction
When a disaster occurs, we usually see many posts on social media and news stations that either inform the people, warn, or request for help.

In this project,i will create a ML model that analyzes and classify messages from real disaster data, [Figure Eight](https://appen.com/), into different categories depending on their meaning. The model will be used by a web application to display the result. An emergency worker can input a new message through the web app and get classification results in several categories

This project can be used to collect messages that people post on different social media platforms when a disaster occurs such floods, storms, or fire and analyze the current situation. It can be helpful by speeding up the time of response of rescue teams and news stations. Also it can help by warning nearby people of the occurance of a disaster and help them evacuate quickly. Finally, it will allow rescue teams to prioritize their actions based on the situation.

## Project structure:
1. ETL pipeline (data directory):
`process_data.py` loads data from the csv files, cleans them, and store the cleaned data into an SQLite database

2. ML pipeline (model directory):
`train_classifier.py` loads data from the database, builds a text processing and machine learning pipeline, trains and tunes the model using GridSearchCV, and exports the final model as a pickle file

3. Flask Web App (app directory):
The web app has data visualizatio for the training set, and provides an interface to classify messages into categories using the pre-built model.

4. Screenshots:
You can find screenshots of the working web application

## Requirements:
To get the flask app working you need:
- python3
- packages in requirements.txt file

install the packages by running:
- `pip3 install wheel`
- `pip3 install -r requirements.txt`

## Instructions:
1. Run the following commands to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database, **go to the data directory** and run:  
            `python3 process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`  
        
    - To run ML pipeline that trains classifier and saves the model,**go to the model directory** and run:  
             `python3 train_classifier.py ../data/DisasterResponse.db classifier.pkl`

2. Run the following command in the **app directory** to run the web app.  
         `python3 run.py`

3. Go to http://0.0.0.0:3001/
