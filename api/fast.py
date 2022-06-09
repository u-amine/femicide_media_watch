
# $DELETE_BEGIN
from datetime import datetime, date
import requests
import pandas as pd
import numpy as np
from itertools import chain
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize 
import joblib

from FemicideMediaWatch.trainer import cleaning_data, lemma, remove_stopwords, basic_cleaning, just_text

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return dict(greeting="hello")

@app.get("/predict")
def predict(text):    
    text= [{'body': text}]  
    new_dataframe = pd.DataFrame([text], columns=["fields"], index=["0"])
    model = joblib.load('pipeline.joblib')
    results=model.predict(new_dataframe)
    print(results[0])  
    return dict(prediction=str(results[0]))

@app.get("/news_feed")
def predict(date_start):
    today= date.today()
    #date_start =datetime.strptime("2000-01-01", "%Y-%m-%d") #if date_start != "field required" else today.strptime("%Y-%m-%d")
    #date_end=datetime.strptime(date_end, "%Y-%m-%d") if date_end != "field required"else today.strptime("%Y-%m-%d")
    i=1
    url=f'https://content.guardianapis.com/search?&from-date=2020-01-01&to-date={today}&show-fields=body&page-size=50&page={i}&api-key=test'
    news = requests.get(url).json()
    total= round(int(news["response"]["total"])/50)
    print(total)
    if total > 100 : total=100
    print(total)
    news_list =[]
    i=1
    while i < total+1:
        news = requests.get(url).json()
        news_list.append(news["response"]["results"])
        i=i+1
    
    y = list(chain(*news_list))
    data= pd.DataFrame(y)
    print(f"data:shape: {data.shape}")
    model = joblib.load('pipeline.joblib')
    
    y_pred= model.predict(data) 

    final_list = data[y_pred]
    del final_list["fields"]
    del final_list["clean_text"]
    final_list= final_list.drop_duplicates(keep="first")
    #r=final_list.drop_duplicates()
    print(final_list.to_dict())
    return final_list.to_dict()


