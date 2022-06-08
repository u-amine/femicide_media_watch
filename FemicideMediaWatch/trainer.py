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
from sklearn import set_config; set_config(display='diagram')
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from google.cloud import storage

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

    vectorizer = CountVectorizer()
    #vectorizer = joblib.load("vectorice.joblib")

    nvaive = MultinomialNB()

    pipe = make_pipeline(feature_averager,
                        vectorizer, 
                        MultinomialNB())
    return pipe

def defining_dataset(df):
    df_solo_1 = df[df.topic ==1]
    y = df_solo_1['cases']
    X = pd.DataFrame(df_solo_1.fields)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    pipe=building_pipeline()
    y_pred = pipe.fit(X, y)
    joblib.dump(y_pred, 'pipeline.joblib')
    
def main():
    df = get_data()
    defining_dataset(df)
    
if __name__ == '__main__':
    from FemicideMediaWatch.trainer import cleaning_data, lemma, remove_stopwords, basic_cleaning, just_text
    main()