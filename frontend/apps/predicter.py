import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

URL = "http://127.0.0.1:8000/predict?text="
guardianAPI = 'https://content.guardianapis.com/search?q=murder&from-date=2022-01-01&show-fields=body,thumbnail&page-size=50&page=1&api-key=test'


def blackText(text):
     st.markdown(f'<h1 style=color:#0B0B0B;font-size:40px;font-family:Verdana;border-radius:2%;">{text}</h1>', unsafe_allow_html=True)

def app():

    st.image("pics/femicide_predicter.png", width=700)
    blackText('Find out if a text or article talks about a Femicide case.')
    txt = st.text_area('Insert text here', height=250)
    predict = st.button('Predict')
    if predict:
        with st.spinner('Loading...'):
            r = requests.get(url =f'{URL}{txt}')
            #prediction = r.json().get('prediction')
            prediction = True
            prediction = False
        if prediction == True:
            st.info('Yes, this article talks about a Femicide case.')
            response = requests.get(guardianAPI)
            articles = response.json().get('response').get('results')
            articles = articles[5:8]

            # for article in articles:
            #     with st.container():
            #         col1, col2 = st.columns((2, 4))
            #         col1.image(article.get('fields').get('thumbnail'), width=200)
            #         col2.subheader(f"[{article.get('webTitle')}]({article.get('webUrl')})")
            #         body = f"{article.get('fields').get('body')[:250]}..."
            #         body = body.replace('<p>', '')
            #         body = body.replace('</p>', '')
            #         col2.markdown(body)
            #     st.markdown("***")
        else:
            st.info('No, this article does not talk about a Femicide case.')
    else:
        st.write('')
