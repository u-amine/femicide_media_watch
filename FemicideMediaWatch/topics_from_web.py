from google.cloud import storage
import pandas as pd
from sklearn import linear_model
import numpy as np
import joblib
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd
from pprint import pprint
import seaborn as sns

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, LdaModel 

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


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

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    nlp = spacy.load("en_core_web_sm")
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def jaccard_similarity(topic_1, topic_2):
    """
    Derives the Jaccard similarity of two topics

    Jaccard similarity:
    - A statistic used for comparing the similarity and diversity of sample sets
    - J(A,B) = (A ∩ B)/(A ∪ B)
    - Goal is low Jaccard scores for coverage of the diverse elements
    """
    intersection = set(topic_1).intersection(set(topic_2))
    union = set(topic_1).union(set(topic_2))
                    
    return float(len(intersection))/float(len(union))

def train_model(data):
    print("starting the model")
    df = data.dropna(subset=['clean_text'])
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    
    # Convert to list
    data = df.clean_text.values.tolist()

    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]
    
    data_words = list(sent_to_words(data))
    print("cleaded data")
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    print("bigram and trigram created")
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    print("bigram mode and trigram mode created")
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words, stop_words)
    print("colection of data words of nonstops")
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    
    print("colection of data_words_bigrams")
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    print("data_lemmatized")
    corpus = data_lemmatized
    print("corpus")
    dirichlet_dict = corpora.Dictionary(corpus)
    bow_corpus = [dirichlet_dict.doc2bow(text) for text in corpus]

    print("bow_corpus")
    # Considering 1-15 topics, as the last is cut off
    num_topics = list(range(30)[1:])
    num_keywords = 10

    LDA_models = {}
    LDA_topics = {}
    for i in num_topics:
        LDA_models[i] = LdaModel(corpus=bow_corpus,
                                id2word=dirichlet_dict,
                                num_topics=i,
                                update_every=1,
                                chunksize=len(bow_corpus),
                                passes=20,
                                alpha='auto',
                                random_state=42)

        shown_topics = LDA_models[i].show_topics(num_topics=i, 
                                                num_words=num_keywords,
                                                formatted=False)
        LDA_topics[i] = [[word[0] for word in topic[1]] for topic in shown_topics]
    
    print("lda_model")
    LDA_stability = {}
    for i in range(0, len(num_topics)-1):
        jaccard_sims = []
        for t1, topic1 in enumerate(LDA_topics[num_topics[i]]): # pylint: disable=unused-variable
            sims = []
            for t2, topic2 in enumerate(LDA_topics[num_topics[i+1]]): # pylint: disable=unused-variable
                sims.append(jaccard_similarity(topic1, topic2))    
            
            jaccard_sims.append(sims)    
        
        LDA_stability[num_topics[i]] = jaccard_sims
    
    print("LDA_stability")             
    mean_stabilities = [np.array(LDA_stability[i]).mean() for i in num_topics[:-1]]
    coherences = [CoherenceModel(model=LDA_models[i], texts=corpus, dictionary=dirichlet_dict, coherence='c_v').get_coherence()\
                    for i in num_topics[:-1]]
    print("coherences")     
    coh_sta_diffs = [coherences[i] - mean_stabilities[i] for i in range(num_keywords)[:-1]] # limit topic numbers to the number of keywords
    coh_sta_max = max(coh_sta_diffs)
    coh_sta_max_idxs = [i for i, j in enumerate(coh_sta_diffs) if j == coh_sta_max]
    ideal_topic_num_index = coh_sta_max_idxs[0] # choose less topics in case there's more than one max
    ideal_topic_num = num_topics[ideal_topic_num_index]
    print("XXXXXXXX")
    print(f"ideal_topic_num: {ideal_topic_num}")
    print("XXXXXXXX")
    plt.figure(figsize=(20,10))
    ax = sns.lineplot(x=num_topics[:-1], y=mean_stabilities, label='Average Topic Overlap')
    ax = sns.lineplot(x=num_topics[:-1], y=coherences, label='Topic Coherence')

    ax.axvline(x=ideal_topic_num, label='Ideal Number of Topics', color='black')
    ax.axvspan(xmin=ideal_topic_num - 1, xmax=ideal_topic_num + 1, alpha=0.5, facecolor='grey')

    y_max = max(max(mean_stabilities), max(coherences)) + (0.10 * max(max(mean_stabilities), max(coherences)))
    ax.set_ylim([0, y_max])
    ax.set_xlim([1, num_topics[-1]-1])
                    
    ax.axes.set_title('Model Metrics per Number of Topics', fontsize=25)
    ax.set_ylabel('Metric Level', fontsize=20)
    ax.set_xlabel('Number of Topics', fontsize=20)
    plt.legend(fontsize=20)
    plt.show()
        # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    print("id2word")
    # Create Corpus
    texts = data_lemmatized
    print("texts")
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=ideal_topic_num, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
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
    print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}/{model_name}.joblib")


if __name__ == '__main__':
    df = get_data()
    reg = train_model(df)
    #vetorices = vocabularie(df)
    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model(reg, "model")
    #save_model(vetorices, "vectorice")
