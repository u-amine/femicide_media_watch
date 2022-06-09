import streamlit as st
import pandas as pd
import numpy as np
import requests

guardianAPI = 'https://content.guardianapis.com/search?q=murder&from-date=2022-01-01&show-fields=body,thumbnail&page-size=50&page=1&api-key=test'
URL = "https://fmw-elg4b5acyq-ew.a.run.app/news_feed?date_start=”222000-01-01"

def header(url):
     st.markdown(f'<p style=color:#9B1146;font-size:24px;font-family:Verdana;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

def blackText(text):
    st.markdown(f'<h1 style=color:#0B0B0B;text-alignt:center;padding-bottom:10px;font-size:70px;font-family:Verdana;border-radius:2%;">{text}</h1>', unsafe_allow_html=True)

def blackText2(text):
    st.markdown(f'<h1 style=color:#0B0B0B;text-alignt:center;padding-bottom:10px;font-size:40px;font-family:Verdana;border-radius:2%;">{text}</h1>', unsafe_allow_html=True)

def underlineText(text):
    st.markdown(f'<p style=color:#0B0B0B;font-size:20px;font-family:Verdana;border-radius:2%;text-decoration:underline;font-weight:bold;">{text}</p>', unsafe_allow_html=True)

def missionText1(text):
    st.markdown(f'<p style=text-align: center;margin:40px;color:#0B0B0B;font-size:50px;font-family:Verdana;border-radius:2%;">{text}</p>', unsafe_allow_html=True)

def numbers(text):
    st.markdown(f'< style=text-align: center;color:#0B0B0B;font-size:50px;font-family:Verdana;border-radius:2%;">{text}</p>', unsafe_allow_html=True)

def numbers(text):
    st.title(f'< style=text-align: center;color:red;font-size:10px;font-family:Verdana;border-radius:2%;">{text}</p>', unsafe_allow_html=True)
    
    
def app():
    blackText("Femicide Media Watch")
    st.text(" ")
    st.text(" ")
    
    st.header('Articles on femicide detected from a database of 30.000 articles by The Guardian')
    
    articles = pd.read_csv('../get_cases_with_thum.csv')
    
    with st.spinner('Please wait, we are evaluating and detecting Cases of Femicide from all the articles of TODAY from The Guardian...'):
        response = requests.get(URL)
        if len(response.json()) != 14:
            news = response.json().get('response').get('results')
        else:
            news=[]
        
    col1, col2, = st.columns(2)
    col1.title(len(articles))
    col1.markdown("Femicide articles detected from The Guardian")
    col2.title(len(news))
    col2.markdown("Femicide articles detected today from The Guardian")
    
    df = pd.read_csv('../final_list.csv', usecols= ['latitude','longitude'])

    st.map(df,zoom=1)
    
    st.header('List of Femicide articles detected from The Guardian')

    missionText1("""In this section you can review the articles detected and check the accuracy of our prediction model.
                 This Archive is updated every day in the event that a Guardian article is detected""")
        
    articleArray= []

    articleArray.append(articles.iloc[7])
    articleArray.append(articles.iloc[4])
    articleArray.append(articles.iloc[5])
    articleArray.append(articles.iloc[8])
    articleArray.append(articles.iloc[9])
    articleArray.append(articles.iloc[10])
    articleArray.append(articles.iloc[11])
    articleArray.append(articles.iloc[12])
    articleArray.append(articles.iloc[13])

    #st.write(articles)

    for article in articleArray:
        with st.container():
            col1, col2 = st.columns((2, 4))
            col1.image(article.get('thumbnail'), width=200)
            col2.subheader(f"[{article.get('webTitle')}]({article.get('webUrl')})")
            body = f"{article.get('body')[:250]}..."
            body = body.replace('<p>', '')
            body = body.replace('</p>', '')
            col2.markdown(body)
        st.markdown("***")

        """ with st.container():
            col1, col2 = st.columns((2, 4))
            #col1.image('https://media.socastsrm.com/wordpress/wp-content/blogs.dir/2272/files/2020/07/Tom-1.jpg', width=200)
            col1.image('pics/placeholder2.jpg', width=200)
            col2.subheader("[Palestinians racially abuse IDF soldier who thwarted stabbing attack](https://wsau.com/2022/06/01/underrated-soul-man/)")
            col2.markdown('''We continue our look back at the music of 50 years ago.....\n\nTake the grooves of Al Green...mix in the funk of James Brown....add the backing of the pros from Muscle Shoals and you have a tasty slice of Southern soul from Bobby Womack...his fourth LP...''')

        st.markdown("***") """
