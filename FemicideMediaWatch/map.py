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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from FemicideMediaWatch.topics_from_web import save_model, upload_model_to_gcp
import geopandas as gpd 
import geopy 
import matplotlib.pyplot as plt
from geopy.extra.rate_limiter import RateLimiter
import folium
from folium.plugins import FastMarkerCluster
import spacy
from spacy import displacy 
from geotext import GeoText
import ast
import pycountry

STORAGE_LOCATION = 'csv'
MODEL_STO= "model"
MODEL_NAME='femicide_model'
BUCKET_NAME ='wagon-data-871-rojas'

def get_data():
    news_list =[]
    i=1
    while i < 761:
        news = requests.get(f'https://content.guardianapis.com/search?q=(murder%20OR%20homicide%20OR%20femicide%20OR%20feminicide%20OR%20murdered%20OR%20dead%20OR%20death%20OR%20killed%20OR%20murdered%20OR%20shot%20OR%20stabbed%20OR%20struck%20OR%20strangled%20OR%20"lifeless")%20AND%20(woman%20OR%20girl%20OR%20"a young woman"%20OR%20"a teenage girl"%20OR%20"a girl"%20OR%20"body of a woman"%20OR%20prostitute%20OR%20"sex worker")&from-date=2000-01-01&show-fields=body,thumbnail&page-size=50&page={i}&api-key=test').json()
        news_list.append(news["response"]["results"])
        print(f"{i}: {len(news_list)}")
        print("_____")
        i=i+1
    y = list(chain(*news_list))
    data= pd.DataFrame(y)
    return data

#def read_csv():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    #df = pd.read_csv(f"gs://{BUCKET_NAME}/data/{BUCKET_TRAIN_DATA_PATH}", nrows=10000)
   # return df
    
def localization(text):
    nlp_wk = spacy.load("xx_ent_wiki_sm")
    doc = nlp_wk(text)
    location=[]
    for ent in doc.ents:
        if ent.label_ in ["LOC"]:
            return ent
    print("____")
    return None  

def local(text):
    for country in pycountry.countries:
    # Handle both the cases(Uppercase/Lowercase)
        if str(country.name).lower() in str(text).lower():
            print(country.name)
            return country.name

""" def cities(text):
    places = GeoText(text)
    if places.cities:
        return places.cities[0]
    elsif: places.countries
        return places.countries[0]
    return None """
    
def get_list_of_cases(data):
    #model= import_model(data)
    model = joblib.load('pipeline.joblib')
    y_pred= model.predict(data) 
    final_list = data[y_pred]
    del final_list["fields"]
    final_list= final_list.drop_duplicates(keep="first")
    return final_list

def save_csv(data, name):
    data.to_csv(f'{name}.csv', index=False)
    #data.to_csv(f'{STORAGE_LOCATION}/femicide-{name}.csv')
    #client = storage.Client()
    #bucket = client.bucket(BUCKET_NAME)
    #blob = bucket.blob(f"{STORAGE_LOCATION}/femicide-{name}.csv")
    #blob.upload_from_filename(f'femicide-{name}.csv')
    
def send_model():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_STO)
    blob.upload_from_filename('pipeline.joblib')

def import_model(data):
    client = storage.Client().bucket(BUCKET_NAME)
    blob = client.blob(f"{STORAGE_LOCATION}/final_home.csv")

def map(df):
    folium_map = folium.Map(location=[59.338315,18.089960],
                    zoom_start=2,
                    tiles="CartoDB dark_matter")
    FastMarkerCluster(data=list(zip(df["latitude"].values, df["longitude"].values))).add_to(folium_map)
    folium.LayerControl().add_to(folium_map)
    folium_map
    
def main():
    #print("sended model")
    #send_model()
   
   # print("geting data")
    #data = get_data()
   
    #save_csv(data, "raw")
    data=pd.read_csv("get_cases.csv")
    print(data.shape)
    print(data.columns)
    #final_list = pd.read_csv(f"gs://{BUCKET_NAME}/csv/get_cases.csv", nrows=30)
    #final_list = pd.read_csv("get_cases.csv", index_col=False)
    #final_list= final_list.reset_index(drop=True, inplace=True)
    data.drop(data.filter(regex="Unname"),axis=1, inplace=True)
    #data.fields= data.fields.apply(lambda x : ast.literal_eval(x))
    #print(f"get_list_of_cases :{data.shape}")
    #final_list= get_list_of_cases(data)
    #print(f"get_list_of_cases : {final_list.shape}")
    #print(final_list.body)
    #save_csv(final_list, "get_cases")
    
    print("localization data")
    final_list["localization"] = data.clean_text.apply(localization)
    save_csv(final_list, "localization data")
    locator = geopy.geocoders.Nominatim(user_agent="mygeocoder")
    geocode = RateLimiter(locator.geocode)
    print("address data")
    save_csv(final_list, "address data")
    final_list["address"] = final_list["localization"].apply(geocode)
    print(final_list["address"])
    final_list["coordinates"] = final_list["address"].apply(lambda loc: tuple(loc.point) if loc else None)
    print("long")
    save_csv(final_list, "long")
    final_list[["latitude", "longitude", "altitude"]] = pd.DataFrame(final_list["coordinates"].tolist(), index=final_list.index)
    print("long")
    final_list.latitude.isnull().sum()
    print("long")
    final_list = final_list[pd.notnull(final_list["latitude"])]
    save_csv(final_list, "final_list") 
    """df=pd.read_csv("final_list.csv")
    print(df.head(1))
    folium_map = folium.Map(location=[59.338315,18.089960],
                    zoom_start=2,
                    tiles="CartoDB dark_matter")
    FastMarkerCluster(data=list(zip(df["latitude"].values, df["longitude"].values))).add_to(folium_map)
    folium.LayerControl().add_to(folium_map)
    folium_map"""
    
if __name__ == '__main__':
    main()