from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from google.cloud import storage
import joblib
import os 
from FemicideMediaWatch.
STORAGE_LOCATION = 'models/femicide_model'
MODEL_NAME='femicide_model'
BUCKET_NAME ='wagon-data-871-rojas'

def get_model(model_name):
    client = storage.Client().bucket(BUCKET_NAME)
    blob = client.blob(f"{STORAGE_LOCATION}/{model_name}.joblib")
    blob.download_to_filename(f'{model_name}.joblib')
    print('=> pipeline downloaded from storage')

def import_model(model_name):
    model = joblib.load(f'{model_name}.joblib')
    return model

def print_topics(model, vectorizer):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-10 - 1:-1]])
        
def main():
    #get_model("model_we_build")
    #get_model("vectorice")
    model = import_model("model_we_build")
    vectorizer = import_model("vectorice")
    print_topics(model, vectorizer)

if __name__ == '__main__':
    model = joblib.load("../pipeline.joblib")
    print(model.predict("text"))