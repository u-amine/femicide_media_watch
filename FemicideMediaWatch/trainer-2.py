from google.cloud import storage
import pandas as pd
from sklearn import linear_model
import numpy as np
import joblib
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME ='wagon-data-871-rojas'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH ='guardian_full.csv'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'topics_prediction'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_data():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    df = pd.read_csv(f"gs://{BUCKET_NAME}/data/{BUCKET_TRAIN_DATA_PATH}", nrows=10000)
    return df

def topic_selection(data,vectorizer, lda_model):
    example_vectorized = vectorizer.transform([data])
    lda_vectors = lda_model.transform(example_vectorized)
    vectore_in_list= list(lda_vectors[0])
    idx = np.where(vectore_in_list == lda_vectors.max())
    return int(idx[0][0])

def print_topics(model, vectorizer):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-10 - 1:-1]])
        
def vocabularie(data):
    print("vocabularie")
    vectorizer = CountVectorizer()
    data_vectoriced= vectorizer.fit_transform(data["clean_text"].apply(lambda x: np.str_(x)))
    return vectorizer
    
def train_model(data):
    print("method that trains the model")
    vocabulario = vocabularie(data)
    lda_model = LatentDirichletAllocation(n_components=9)
    vectorizer = CountVectorizer()
    data_vectoriced= vectorizer.fit_transform(data["clean_text"].apply(lambda x: np.str_(x)))
    lda_model.fit(data_vectoriced)
    #print_topics(lda_model, vectorizer)
    #data['topic'] = data["clean_text"].apply(lambda x: np.str_(topic_selection(x,vectorizer, lda_model)))
    #new = data.to_csv(f'guardian_with_topics_final.csv', index=False, header=True)
    #client = storage.Client()

    #bucket = client.bucket(BUCKET_NAME)

    #blob = bucket.blob(STORAGE_dos)

    #blob.upload_from_filename('guardian_with_topics_final.csv')
    return lda_model


STORAGE_LOCATION = 'models/femicide_model'
STORAGE_dos='csv/guardian_with_topics_final.csv'


def upload_model_to_gcp(model_name):
    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(f"{STORAGE_LOCATION}/{model_name}.joblib")

    blob.upload_from_filename(f'{model_name}.joblib')



def save_model(reg, model_name):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    joblib.dump(reg, f'{model_name}.joblib')
    print(f"saved {model_name}.joblib locally")

    # Implement here
    upload_model_to_gcp(model_name)
    print(f"uploaded model_we_build.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}/{model_name}.joblib")


if __name__ == '__main__':
    # get training data from GCP bucket
    df = get_data()
    reg = train_model(df)
    vetorices = vocabularie(df)
    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model(reg, "model_we_build")
    save_model(vetorices, "vectorice")
