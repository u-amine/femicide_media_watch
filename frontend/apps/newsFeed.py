import streamlit as st
import pandas as pd
import numpy as np
import requests

guardianAPI = 'https://content.guardianapis.com/search?q=murder&from-date=2022-01-01&show-fields=body,thumbnail&page-size=50&page=1&api-key=test'

def header(url):
     st.markdown(f'<p style=color:#9B1146;font-size:24px;font-family:Verdana;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

def blackText(text):
    st.markdown(f'<h1 style=color:#0B0B0B;font-size:30px;font-family:Verdana;border-radius:2%;">{text}</h1>', unsafe_allow_html=True)

def underlineText(text):
    st.markdown(f'<p style=color:#0B0B0B;font-size:20px;font-family:Verdana;border-radius:2%;text-decoration:underline;font-weight:bold;">{text}</p>', unsafe_allow_html=True)

def app():

    st.image("pics/news_feed.png", width=700)
    blackText('Stay up to date with the latest news on femicide. Globally.')
    col1, col2  = st.columns(2)
    col1.title('26')
    col1.markdown("Latest articles about femicide from the last year")
    col2.title('1')
    col2.markdown("New articles about femicide today")

    df = pd.read_csv('data/final_list.csv', usecols= ['latitude','longitude'])

    st.map(df)

    st.header('Latest articles')

    response = requests.get(guardianAPI)
    articles = response.json().get('response').get('results')
    articleArray= []
    articleArray.append(articles[8])
    articleArray.append(articles[23])
    articleArray.append(articles[24])

    #st.write(articles)

    for article in articleArray:
        with st.container():
            col1, col2 = st.columns((2, 4))
            col1.image(article.get('fields').get('thumbnail'), width=200)
            col2.subheader(f"[{article.get('webTitle')}]({article.get('webUrl')})")
            body = f"{article.get('fields').get('body')[:250]}..."
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
