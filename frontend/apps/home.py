import streamlit as st
import pandas as pd
import numpy as np

def blackText(text):
    st.markdown(f'<p style=color:#0B0B0B;font-size:18px;font-family:Verdana;border-radius:2%;">{text}</p>', unsafe_allow_html=True)

def purpleText(text):
    st.markdown(f'<p style=color:#9B1146;font-size:18px;font-family:Verdana;border-radius:2%;">{text}</p>', unsafe_allow_html=True)

def underlineText(text):
    st.markdown(f'<p style=color:#0B0B0B;font-size:18px;font-family:Verdana;border-radius:2%;text-decoration:underline;font-weight:bold;">{text}</p>', unsafe_allow_html=True)

def app():

    #st.title('Femicide Media Watch')
    st.image("../frontend/pic.png", width=800)
    purpleText('Every year an average of 66.000 women are violently killed globally.')
    blackText('Many of these cases are not properly reported as femicide.')
    underlineText('We are on a mission to change that.')
    purpleText('                      ')
    purpleText('                      ')
    col1, col2  = st.columns(2)
    col1.button(label='FMW Predicter')
    col2.button(label='FMW News Feed')
    #col1.markdown(f'<p style=color:#0B0B0B;font-size:20px;font-family:Verdana;border-radius:2%;border-style:solid;">text</p>', unsafe_allow_html=True)
    #col2.markdown(f'<p style=color:#0B0B0B;font-size:20px;font-family:Verdana;border-radius:2%;border-style:solid;">text</p>', unsafe_allow_html=True)
    # col1.markdown("Latest articles about femicide from the last 30 days")
    # col1.title('26')
    # col2.markdown("New articles about femicide today")
    # col2.title('2')
    # st.header('Our mission with Femicide Media Watch')
    # st.markdown('''
    # There is one key barrier to understanding the global impact of femicide: there is no global definition of femicide.
    # That means that individual countries and institutions are only collecting data that fits within these, often very narrow, definitions.
    # For example, in the UK trans women are routinely omitted from datasets, and in Chile, until a few years ago, women murdered by people other than their partner,
    # or ex-partner, were not included at all. This results in data that is inaccessible, miscategorised and too varied in indicators.

    # \n
    # Subsequently, women’s organisations and  social justice groups, who are often underfunded and under-resourced,
    # have picked up the slack by collecting detailed femicide records; the work by these organisations will be supported
    # by this project that aims to provide a platform for their data, and have a positive effect on the reporting of femicide and gender violence.
    # ''')
    # st.subheader('Every year, an average of 66,000 women are violently killed globally.')
    # st.subheader('Find out if your text talks about femicide')
    # txt = st.text_area('Insert text here', height=200)
    # if st.button('Predict'):
    #     st.header('Related articles')
    #     with st.container():
    #         col1, col2 = st.columns([1, 3])
    #         col1.image('https://media.socastsrm.com/wordpress/wp-content/blogs.dir/2272/files/2020/07/Tom-1.jpg', width=250)
    #         col2.subheader("[Underrated Soul Man](https://wsau.com/2022/06/01/underrated-soul-man/)")
    #         col2.markdown('''We continue our look back at the music of 50 years ago.....\n\nTake the grooves of Al Green...mix in the funk of James Brown....add the backing of the pros from Muscle Shoals and you have a tasty slice of Southern soul from Bobby Womack...his fourth LP...''')

    #     with st.container():
    #         col1, col2 = st.columns([1, 3])
    #         col1.image('https://worldisraelnews.com/wp-content/uploads/2022/06/download-e1654060997695-1.jpg', width=250)
    #         col2.subheader("[Palestinians racially abuse IDF soldier who thwarted stabbing attack](https://worldisraelnews.com/palestinians-racially-abuse-idf-soldier-who-thwarted-stabbing-attack/)")
    #         col2.markdown('''Angry mob surrounds soldier who they believe fired the shots that killed terrorist, tells him to go back to Ethiopia.\n\nBy World Israel News Staff\n\nA Palestinian woman was shot dead by IDF troops after charging at them while holding a knife near Al-Arroub''')

    #     with st.container():
    #         col1, col2 = st.columns([1, 3])
    #         col1.image('https://remezcla.com/wp-content/uploads/2018/03/Protest_Artwork.jpg', width=250)
    #         col2.subheader("[Life imprisonment till death for Tripura editor in triple murder case](https://www.thenewsminute.com/article/life-imprisonment-till-death-tripura-editor-triple-murder-case-21852)")
    #         col2.markdown('''The News Minute | July 17, 2014 | 03:22 pm IST Agartala: A court in Tripura on Thursday awarded life imprisonment till death to a newspaper editor-cum-owner in a year-old triple murder case, a public prosecutor said. Kripankur Chakraborty, additional district ''')

    #     with st.container():
    #         col1, col2 = st.columns([1, 3])
    #         col1.image('https://i2-prod.stokesentinel.co.uk/incoming/article7151631.ece/ALTERNATES/s1200/0_Untitled-collage.jpg', width=250)
    #         col2.subheader("[The disgraced darts champ and other criminals justice caught up with in May](https://www.stokesentinel.co.uk/news/stoke-on-trent-news/disgraced-darts-champ-cowboy-builder-7146015)")
    #         col2.markdown('''These are the people who have been jailed for crimes linked to North Staffordshire\n\nThis list shows the faces of the people jailed in May for crimes linked to North Staffordshire. Among those locked up include disgraced darts player Ted Hankey who sexually ...''')

    # else:
    #     st.write('')
