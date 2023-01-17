# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import pandas as pd
import string
import numpy as np
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from random import sample
import pickle

## list of punctuations and stopwords
PUNCTUATIONS = list(string.punctuation)
STOP_WORDS = ['am', 'once', 'few', 'as', 'our', 'm', 'hasn', 'who', 'above', 'itself', 'him', 'does', 'before', 'after', 'now', 'were', 'any', 'ma', 'on', 'y', 'be', 'more', 'will', 'yourself', 'against', 's', 'these', 'doing', 'himself', 'been', 'ourselves', 'aren', 'there', 'needn', 'wasn', "you've", 'same', 'its', 'are', 'she', "you'd", 'below', 'to', 'ain', 'and', 'her', 'theirs', 'under', 'whom', 'those', 'you', 'what', 'that', 've', 'all', 're', 'further', 'is', 'until', "you're", 'the', 'his', 'do', 'of', 'each', 'he', 'have', 'haven', 'was', 'isn', 'with', 'most', "should've", 'had', 'wouldn', 'off', 'nor', 'very', 'down', "that'll", 'up', 'hers', 'their', 'just', 'yourselves', 'by', 'shouldn', 'can', 'being', 'than', 'hadn', 'has', 'doesn', 'only', 'll', 'both', 'such', 'did', 'mightn', 'some', 'here', 'where', 'o', 'i', 'don', 'for', 'couldn', 'mustn', "you'll", 'if', 'because', "she's", 'into', 'about', 'why', 'how', 'd', 'too', 'your', 'when', 'own', 'ours', 'it', 'out', 'we', 'an', 'other', 'during', 'me', 'weren', 'so', "it's", 'or', 'while', 'again', 'yours', 'won', 'shan', 'at', 'between', 'over', 'they', 'should', 'themselves', 'in', 'having', 'myself', 'then', 'my', 'through', 'them', 'this', 'which', 'a', 'from', 't', 'herself']



# INSTRUCTIONS: You are responsible for making sure that this script outputs 

# 1) the evaluation scores of your system on the data in CSV_TEST (minimally 
# accuracy, if possible also recall and precision).

# 2) a csv file with the contents of a dataframe built from CSV_TEST that 
# contains 3 columns: the gold labels, your system's predictions, and the texts
# of the reviews.

TRIAL = False
# ATTENTION! the only change that we are supposed to do to your code
# after submission is to change 'True' to 'False' in the following line:
EVALUATE_ON_DUMMY = True

#path to load the model
MODEL_PATH=None
VECTORIZER_PATH='vectorizer.sav'

# the real thing:
CSV_TRAIN = "data/sentiment_train.csv"
CSV_VAL = "data/sentiment_val.csv"
CSV_TEST = "data/sentiment_test.csv" # you don't have this file; we do

if TRIAL:
    CSV_TRAIN = "data/sentiment_10.csv"
    CSV_VAL = "data/sentiment_10.csv"
    CSV_TEST = "data/sentiment_10.csv"
    print('You are using your SMALL dataset!')
elif EVALUATE_ON_DUMMY:
    CSV_TEST = "data/sentiment_dummy_test_set.csv"
    print('You are using the FULL dataset, and using dummy test data! (Ok for system development.)')
else:
    print('You are using the FULL dataset, and testing on the real test data.')


def preprocessing(text):
    """
       This module takes text in string as input, replace contractions,removes punctuation\\
               removes stop words, lemmatize and return the cleaned text.
    Parameters:
    text (str): text to be cleaned

    """

    #dictionary consisting of the contraction and the actual value
    Apos_dict={"'s":" is","n't":" not","'m":" am","'ll":" will",
           "'d":" would","'ve":" have","'re":" are"}

    #replace the contractions
    for key,value in Apos_dict.items():
        if key in text:
            text=text.replace(key,value)
    tokens = text.split(" ")
    #lemmatize
    #tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in PUNCTUATIONS]
    tokens = [lemmatizer.lemmatize(token) for token in tokens if (token not in STOP_WORDS) and (token not in PUNCTUATIONS)]
    return " ".join(tokens)

def calculate_accuracy(gold, prediction):
    n_total_predictions = len(prediction)
    correct_predictions = gold == prediction
    acc = np.sum(correct_predictions) / n_total_predictions
    return acc

if not MODEL_PATH:
    print("training the model")
    #read and preprocess train, test and validation datasets
    train_data = pd.read_csv(CSV_TRAIN)  # loading data
    validation_data = pd.read_csv(CSV_VAL)  # loading data

    print('n columns:')
    print("Training data: ",len(train_data.columns))
    print("Validation data: ",len(validation_data.columns))

    print('n rows:')
    print("Training data: ",len(train_data))
    print("Validation data: ",len(validation_data))
   
    # preprocessing- tokenizing removing stop words and expanding words and lemmatizing
    train_tokens = [preprocessing(each) for each in train_data["text"]] # preprocessing train data
    validation_tokens = [preprocessing(each) for each in validation_data["text"]] # preprocessing validation data

    #vectorizing the words
    vectorizer = CountVectorizer()
    train_X = vectorizer.fit_transform(train_tokens)
    train_Y = train_data["sentiment"].to_numpy()
    validation_X = vectorizer.transform(validation_tokens)
    validation_Y = validation_data["sentiment"].to_numpy()
    # training
    # initializing the model
    model = LogisticRegression()
    # training the model
    model = model.fit(train_X, train_Y)
    # predicting result on train dataset
    train_data['predicted_by_logistic_regression'] = model.predict(train_X)

    accuracy = calculate_accuracy(train_Y, train_data['predicted_by_logistic_regression'])
    print(f'Accuracy linear regression: {accuracy}')

    #validation
    validation_data['predicted_by_logistic_regression'] = model.predict(validation_X)

    accuracy_val = calculate_accuracy(validation_Y, validation_data['predicted_by_logistic_regression'])
    print(f'Validation-accuracy linear regression: {accuracy_val}')
    
    precision_val = precision_score(validation_Y, validation_data['predicted_by_logistic_regression'], pos_label='pos')
    recall_val = recall_score(validation_Y, validation_data['predicted_by_logistic_regression'], pos_label='pos')

    print(f'Precision on val-set: {precision_val}; Recall on val-set: {recall_val}')

    ##validation wrong result
    validation_wrong = validation_data[validation_data['predicted_by_logistic_regression'] != validation_data['sentiment']]
    validation_wrong.to_csv("validation_wrong.csv")

    print("saving the model")
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    filename = 'vectorizer.sav'
    pickle.dump(vectorizer, open(filename, 'wb'))
    

else:
    print("loading and evaluating the model")
    # load the model from disk
    model = pickle.load(open(MODEL_PATH, 'rb'))
    vectorizer=pickle.load(open(VECTORIZER_PATH, 'rb'))

## Testing
test_data = pd.read_csv(CSV_TEST)  # loading data
print("no column")
print("Test data: ", len(test_data.columns))
print("no. rows")
print("Test data: ", len(test_data))

# preprocessing
test_tokens = [preprocessing(each) for each in test_data["text"]] # preprocessing test data
#vectorizing
test_X = vectorizer.transform(test_tokens)
test_Y = test_data["sentiment"].to_numpy()
#prediction
test_data['predicted_by_logistic_regression'] = model.predict(test_X)

# metrics
accuracy_test = calculate_accuracy(test_Y, test_data['predicted_by_logistic_regression'])
print(f'Test-accuracy linear regression: {accuracy_test}')
precision_test = precision_score(test_Y, test_data['predicted_by_logistic_regression'], pos_label='pos')
recall_test = recall_score(test_Y, test_data['predicted_by_logistic_regression'], pos_label='pos')
print(f'Precision on test-set: {precision_test}; Recall on test-set: {recall_test}') 
#save the csv for CSV_TEST with gold annotation and predicted annotation
test_data.columns=["Gold Sentiment", "Text", "Predicted Sentiment"]
test_data.to_csv("final.csv")


