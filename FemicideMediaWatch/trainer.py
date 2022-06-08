import joblib
import os 
import requests
import pandas as pd
from itertools import chain
import re
from nltk.corpus import stopwords 
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize 
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from google.cloud import storage
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

STORAGE_LOCATION = 'models/femicide_model'
MODEL_NAME='femicide_model'
BUCKET_NAME ='wagon-data-871-rojas'

def imitating_guardian(text):
    return {"body": text}

def get_data():
    #client = storage.Client().bucket(BUCKET_NAME)
    #blob = client.blob(f"{STORAGE_LOCATION}/{model_name}.joblib")
    #blob.download_to_filename(f'{model_name}.joblib')
    df = pd.read_csv("FemicideMediaWatch/data/femicide_final_with_ceros.csv")
    df["fields"]= df.clean_text.apply(imitating_guardian)
    return df

def just_text(text):
    CLEANR = re.compile('<.*?>') 
    cleantext = re.sub(CLEANR, '', text['body'])
    return cleantext

def basic_cleaning(sentence):
    sentence = sentence.strip()
    #print(type(sentence))
    sentence = sentence.lower()
    sentence = ''.join(char for char in sentence if not char.isdigit())
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '') 
    return sentence

def remove_stopwords (text):
    stop_words = set(stopwords.words('english'))
    tokenized = word_tokenize(text)
    without_stopwords = [word for word in tokenized if not word in stop_words]
    return without_stopwords

def lemma(text):
    lemmatizer = WordNetLemmatizer() # Initiate lemmatizer
    lemmatized = [lemmatizer.lemmatize(word,pos="n") for word in text] # Lemmatize
    lemmatized_string = " ".join(lemmatized)
    return lemmatized_string

def cleaning_data(data):
    print(f"XXXXXXXXXXXXXXXXXX")
    print(f"lets start cleaning")
    data['body'] = data.fields.apply(just_text)
    print(f"XXXXXXXXXXXXXXXXXX")
    print(f"lets continued with the basic of cleaning")
    data['clean_text'] = data.body.apply(basic_cleaning)
    print(f"XXXXXXXXXXXXXXXXXX")
    print(f"lets continued with the hause cleaning and removing stopwords")
    data['clean_text'] = data.clean_text.apply(remove_stopwords)
    print(f"XXXXXXXXXXXXXXXXXX")
    print(f"lemmanizingggg")
    data['clean_text'] = data.clean_text.apply(lemma)
    print("fin")
    #data['clean_text'] = data['clean_text'].astype('str')
    return data.clean_text

def building_pipeline():
    feature_averager = FunctionTransformer(cleaning_data)

    vectorizer = CountVectorizer(ngram_range=(1, 1))
    #vectorizer = joblib.load("vectorice.joblib")
    #X_bow= vectorizer.fit_transform(X.clean_text)
    nvaive = MultinomialNB(alpha=0.1)

    pipe = make_pipeline(feature_averager,
                        vectorizer, 
                        nvaive)
    
    
    #cv_nb = cross_validate(MultinomialNB(),X_bow, X.cases , scoring = "recall")

    #print(cv_nb['test_score'].mean())
    return pipe

def grid_search(X, pipe):
    parameters = {
        'countvectorizer__ngram_range': ((1, 2),(1,1), (2,2), (1, 3)),
        'multinomialnb__alpha': (0.1,1)}

    """ 'countvectorizer__max_df': 1.0,
        'countvectorizer__max_features': None,
        'countvectorizer__min_df': (1,10) """
        
    grid_search = GridSearchCV(pipe, parameters, scoring = "recall",
                            cv = 5, n_jobs=-1, verbose=1)

    grid_search.fit(X, X.cases)

    print(f"Best Score = {grid_search.best_score_}")
    print(f"Best params = {grid_search.best_params_}")
    
def defining_dataset(df):
    df_solo_1 = df[df.topic ==1]
    y = df_solo_1['cases']
    X = pd.DataFrame(df_solo_1.fields)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    pipe=building_pipeline()
    #grid_search(df_solo_1, pipe)
    y_pred = pipe.fit(X, y)

    joblib.dump(y_pred, 'pipeline.joblib')
    
def main():
    df = get_data()
    defining_dataset(df)
    
if __name__ == '__main__':
    from FemicideMediaWatch.trainer import cleaning_data, lemma, remove_stopwords, basic_cleaning, just_text
    main()