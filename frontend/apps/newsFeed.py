import streamlit as st
import pandas as pd
import numpy as np

def header(url):
     st.markdown(f'<p style=color:#9B1146;font-size:24px;font-family:Verdana;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

def blackText(text):
    st.markdown(f'<h1 style=color:#0B0B0B;font-size:30px;font-family:Verdana;border-radius:2%;">{text}</h1>', unsafe_allow_html=True)

#def purpleText(text):
    st.markdown(f'<h1 style=color:#9B1146;font-size:20px;font-family:Verdana;border-radius:2%;">{text}</h1>', unsafe_allow_html=True)

def underlineText(text):
    st.markdown(f'<p style=color:#0B0B0B;font-size:20px;font-family:Verdana;border-radius:2%;text-decoration:underline;font-weight:bold;">{text}</p>', unsafe_allow_html=True)

def app():

    st.image("frontend/news_feed.png", width=700)
    blackText('Stay up to date with the latest news on femicide. Globally.')
    col1, col2  = st.columns(2)
    col1.title('26')
    col1.markdown("Latest articles about femicide from the last 30 days")
    col2.title('2')
    col2.markdown("New articles about femicide today")
    st.header('Latest articles')

    with st.container():
        col1, col2 = st.columns((2, 4))
        #col1.image('https://media.socastsrm.com/wordpress/wp-content/blogs.dir/2272/files/2020/07/Tom-1.jpg', width=200)
        col1.image('frontend/placeholder1.jpg', width=200)
        col2.subheader("[Underrated Soul Man](https://wsau.com/2022/06/01/underrated-soul-man/)")
        col2.markdown('''We continue our look back at the music of 50 years ago.....\n\nTake the grooves of Al Green...mix in the funk of James Brown....add the backing of the pros from Muscle Shoals and you have a tasty slice of Southern soul from Bobby Womack...his fourth LP...''')

    st.markdown("***")

    with st.container():
        col1, col2 = st.columns((2, 4))
        #col1.image('https://media.socastsrm.com/wordpress/wp-content/blogs.dir/2272/files/2020/07/Tom-1.jpg', width=200)
        col1.image('frontend/placeholder2.jpg', width=200)
        col2.subheader("[Palestinians racially abuse IDF soldier who thwarted stabbing attack](https://wsau.com/2022/06/01/underrated-soul-man/)")
        col2.markdown('''We continue our look back at the music of 50 years ago.....\n\nTake the grooves of Al Green...mix in the funk of James Brown....add the backing of the pros from Muscle Shoals and you have a tasty slice of Southern soul from Bobby Womack...his fourth LP...''')

    st.markdown("***")


    # with st.container():
    #     col1, col2 = st.columns([1, 3])
    #     #col1.image('https://media.socastsrm.com/wordpress/wp-content/blogs.dir/2272/files/2020/07/Tom-1.jpg', width=200)
    #     col1.image('frontend/placeholder1.jpg', width=200)
    #     col2.subheader("[Underrated Soul Man](https://wsau.com/2022/06/01/underrated-soul-man/)")
    #     col2.markdown('''We continue our look back at the music of 50 years ago.....\n\nTake the grooves of Al Green...mix in the funk of James Brown....add the backing of the pros from Muscle Shoals and you have a tasty slice of Southern soul from Bobby Womack...his fourth LP...''')

    # with st.container():
    #     col1, col2 = st.columns([1, 3])
    #     col1.image('https://worldisraelnews.com/wp-content/uploads/2022/06/download-e1654060997695-1.jpg', width=200)
    #     col2.subheader("[Palestinians racially abuse IDF soldier who thwarted stabbing attack](https://worldisraelnews.com/palestinians-racially-abuse-idf-soldier-who-thwarted-stabbing-attack/)")
    #     col2.markdown('''Angry mob surrounds soldier who they believe fired the shots that killed terrorist, tells him to go back to Ethiopia.\n\nBy World Israel News Staff\n\nA Palestinian woman was shot dead by IDF troops after charging at them while holding a knife near Al-Arroub''')

    # with st.container():
    #     col1, col2 = st.columns([1, 3])
    #     col1.image('https://remezcla.com/wp-content/uploads/2018/03/Protest_Artwork.jpg', width=200)
    #     col2.subheader("[Life imprisonment till death for Tripura editor in triple murder case](https://www.thenewsminute.com/article/life-imprisonment-till-death-tripura-editor-triple-murder-case-21852)")
    #     col2.markdown('''The News Minute | July 17, 2014 | 03:22 pm IST Agartala: A court in Tripura on Thursday awarded life imprisonment till death to a newspaper editor-cum-owner in a year-old triple murder case, a public prosecutor said. Kripankur Chakraborty, additional district ''')

    # with st.container():
    #     col1, col2 = st.columns([1, 3])
    #     col1.image('https://i2-prod.stokesentinel.co.uk/incoming/article7151631.ece/ALTERNATES/s1200/0_Untitled-collage.jpg', width=200)
    #     col2.subheader("[The disgraced darts champ and other criminals justice caught up with in May](https://www.stokesentinel.co.uk/news/stoke-on-trent-news/disgraced-darts-champ-cowboy-builder-7146015)")
    #     col2.markdown('''These are the people who have been jailed for crimes linked to North Staffordshire\n\nThis list shows the faces of the people jailed in May for crimes linked to North Staffordshire. Among those locked up include disgraced darts player Ted Hankey who sexually ...''')
