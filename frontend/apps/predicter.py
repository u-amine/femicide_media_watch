import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

URL = "https://fmw-elg4b5acyq-ew.a.run.app/predict?text="
guardianAPI = 'https://content.guardianapis.com/search?q=murder&from-date=2022-01-01&show-fields=body,thumbnail&page-size=50&page=1&api-key=test'

def header1(url):
     st.markdown(f'<p style=color:#9B1146;font-size:24px;font-family:Verdana;border-radius:2%;">{url}</p>', unsafe_allow_html=True)


def blackText1(text):
    st.markdown(f'<h1 style=color:#0B0B0B;text-alignt:center;padding-bottom:10px;font-size:70px;font-family:Verdana;border-radius:2%;">{text}</h1>', unsafe_allow_html=True)

def missionText1(text):
    st.markdown(f'<p style=text-align: center;margin:40px;color:#0B0B0B;font-size:50px;font-family:Verdana;border-radius:2%;">{text}</p>', unsafe_allow_html=True)


def blackText(text):
     st.markdown(f'<h1 style=color:#0B0B0B;font-size:40px;font-family:Verdana;border-radius:2%;">{text}</h1>', unsafe_allow_html=True)

def app():

    #st.image("pics/femicide_predicter.png", width=700)
    blackText1("Femicide Media Watch")
    #st.header('Articles on femicide detected from a database of 30.000 articles by The Guardian')
    st.header('Evaluate whether a text or article deals with a case of Femicide:')
    missionText1("""In this section you can review the articles detected and check the accuracy of our prediction model.
                 This Archive is updated every day in the event that a Guardian article is detected""")
    st.text(" ")
    st.text(" ")
    txt = st.text_area('Insert text/article here', height=250)
    st.text(" ")
    st.text(" ")
    predict = st.button('Predict')
    if predict:
        with st.spinner('Loading...'):
            r = requests.get(url =f'{URL}"{txt}"')
            prediction = r.json().get('prediction')
            st.text(" ")
        if prediction == "True":
            st.text(" ")
            st.error('Yes, this article talks about a Femicide case.')
            
            articles = pd.read_csv('../final_list.csv')
            
            articleArray= []

            articleArray.append(articles.iloc[1])
            articleArray.append(articles.iloc[5])
            
            """ print(articles.iloc[7])
            for article in articles:
                with st.container():
                    col1, col2 = st.columns((2, 4))
                    print(article)
                    #col1.image(article.get('fields').get('thumbnail'), width=200)
                    col2.subheader(f"[{article.get('webTitle')}]({article.get('webUrl')})")
                    body = f"{article.get('clean_text')[:250]}..."
                    body = body.replace('<p>', '')
                    body = body.replace('</p>', '')
                    col2.markdown(body)
                st.markdown("***") """
        else:
            st.text(" ")
            st.info('No, this article does not talk about a Femicide case.')
    else:
        st.write('')
