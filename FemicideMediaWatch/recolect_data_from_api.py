import requests
import pandas as pd
from itertools import chain
import re
from nltk.corpus import stopwords 
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize 

def save_csv(df, name):
    df.to_csv(f'./data/{name}.csv')
    print(f"Document is saved as {name}.csv")
    return 

def just_text(text):
    CLEANR = re.compile('<.*?>') 
    cleantext = re.sub(CLEANR, '', text["body"])
    print(f"cleaded text 1")
    print("_____")
    return cleantext

def basic_cleaning(sentence):
    sentence = sentence.strip()
    sentence = sentence.lower()
    sentence = ''.join(char for char in sentence if not char.isdigit())
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '') 
    print(f"cleaded text 2")
    print("_____")
    return sentence

def remove_stopwords (text):
    stop_words = set(stopwords.words('english'))
    tokenized = word_tokenize(text)
    without_stopwords = [word for word in tokenized if not word in stop_words]
    print(f"removed stopwords")
    print("_____")
    return without_stopwords

def lemma(text):
    lemmatizer = WordNetLemmatizer() # Initiate lemmatizer
    lemmatized = [lemmatizer.lemmatize(word) for word in text] # Lemmatize
    lemmatized_string = " ".join(lemmatized)
    print(f"lemmatizer stopwords")
    print("_____")
    return lemmatized_string

def clean (text):
    """ for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ') # Remove Punctuation
    lowercased = text.lower() # Lower Case
    tokenized = word_tokenize(lowercased) # Tokenize
    words_only = [word for word in tokenized if word.isalpha()] # Remove numbers
    stop_words = set(stopwords.words('english')) # Make stopword list
    without_stopwords = [word for word in words_only if not word in stop_words] # Remove Stop Words
    lemma=WordNetLemmatizer() # Initiate Lemmatizer
    lemmatized = [lemma.lemmatize(word) for word in without_stopwords] # Lemmatize
    return lemmatized """

def cleaning_data(data):
    print(f"XXXXXXXXXXXXXXXXXX")
    print(f"lets start cleaning")
    print(f"XXXXXXXXXXXXXXXXXX")
    data['body'] = data.fields.apply(just_text)
    print(f"XXXXXXXXXXXXXXXXXX")
    print(f"lets continued with the basic of cleaning")
    print(f"XXXXXXXXXXXXXXXXXX")
    data['clean_text'] = data.body.apply(basic_cleaning)
    print(f"XXXXXXXXXXXXXXXXXX")
    print(f"lets continued with the hause cleaning and removing stopwords")
    print(f"XXXXXXXXXXXXXXXXXX")
    data['clean_text'] = data.clean_text.apply(remove_stopwords)
    print(f"XXXXXXXXXXXXXXXXXX")
    print(f"lemmanizingggg")
    print(f"XXXXXXXXXXXXXXXXXX")
    data['clean_text'] = data.clean_text.apply(lemma)
    data['clean_text'] = data['clean_text'].astype('str')
    print(f"saving the file!")
    print(f"XXXXXXXXXXXXXXXXXX")
    save_csv(data, "guardian_full")

def main():
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
    save_csv(data, "guardian")
    data_to_csv= cleaning_data(data)
    
if __name__ == '__main__':
    main
