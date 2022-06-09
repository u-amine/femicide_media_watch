import streamlit as st
import pandas as pd
import numpy as np

def header(url):
     st.markdown(f'<p style=color:#9B1146;font-size:24px;font-family:Verdana;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

def blackText(text):
    st.markdown(f'<h1 style=color:#0B0B0B;text-alignt:center;padding-bottom:10px;font-size:70px;font-family:Verdana;border-radius:2%;">{text}</h1>', unsafe_allow_html=True)

def purpleText(text):
    st.markdown(f'<p style=color:#9B1146;font-size:20px;font-family:Verdana;border-radius:2%;">{text}</p>', unsafe_allow_html=True)

def underlineText(text):
    st.markdown(f'<p style=color:#0B0B0B;font-size:20px;font-family:Verdana;border-radius:2%;text-decoration:underline;font-weight:bold;">{text}</p>', unsafe_allow_html=True)

def missionText1(text):
    st.markdown(f'<p style=text-align: center;margin-bottom:40px;color:#0B0B0B;font-size:50px;font-family:Verdana;border-radius:2%;">{text}</p>', unsafe_allow_html=True)


def app():
    
    blackText("Femicide Media Watch")
   # st.header("[Contribute](https://github.com/u-amine/femicide_media_watch) to our open source project")
    st.text(" ")
    st.text(" ")
    header("A little bit about our model")
    
    missionText1("""In this section you can review the articles detected and check the accuracy of our prediction model.
                 This Archive is updated every day in the event that a Guardian article is detected""")  
    missionText1("""In this section you can review the articles detected and check the accuracy of our prediction model.
                 This Archive is updated every day in the event that a Guardian article is detected""")  
    missionText1("""In this section you can review the articles detected and check the accuracy of our prediction model.
                 This Archive is updated every day in the event that a Guardian article is detected""")   
    
    st.image("pics/matrix.png", width=400)
    st.text(" ")
    st.text(" ")
    header("Next Steps, the idea behind the open source")
    st.text(" ")
    
    missionText1("""In this section you can review the articles detected and check the accuracy of our prediction model.
                 This Archive is updated every day in the event that a Guardian article is detected""")  
    missionText1("""In this section you can review the articles detected and check the accuracy of our prediction model.
                 This Archive is updated every day in the event that a Guardian article is detected""")  
    missionText1("""In this section you can review the articles detected and check the accuracy of our prediction model.
                 This Archive is updated every day in the event that a Guardian article is detected""")   
    
    st.text(" ")
    st.text(" ")
    header("The Potencial of Sharing")
    st.text(" ")
    col1, col2, = st.columns(2)
    col1.subheader('Integrate the python package into your project')
    col1.markdown("Femicide articles detected from The Guardian")
    col1.subheader('Colaborate with the Repository')
    col1.markdown("You can integrate a local Newspaper")
    
    st.text(" ")
    st.text(" ")
    

    st.text(" ")
    st.text(" ")
    st.subheader('Made with 🫀🧠 by: ')
    col1, col2 = st.columns(2)
    col1.write('[Catalina Rojas Ugarte](https://github.com/crojasu)')
    col2.write('[Amine Ünal](https://github.com/u-amine)')
    col1.write('[Soenke Bernhardi](https://github.com/sbernhardi)')
    col2.write('[Vera Montacuti](https://github.com/vmontacuti)')
