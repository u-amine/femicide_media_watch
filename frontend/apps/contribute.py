import streamlit as st
import pandas as pd
import numpy as np

def header(url):
     st.markdown(f'<p style=color:#9B1146;font-size:24px;font-family:Verdana;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

def blackText(text):
    st.markdown(f'<h1 style=color:#0B0B0B;font-size:30px;font-family:Verdana;border-radius:2%;">{text}</h1>', unsafe_allow_html=True)

def purpleText(text):
    st.markdown(f'<p style=color:#9B1146;font-size:20px;font-family:Verdana;border-radius:2%;">{text}</p>', unsafe_allow_html=True)

def underlineText(text):
    st.markdown(f'<p style=color:#0B0B0B;font-size:20px;font-family:Verdana;border-radius:2%;text-decoration:underline;font-weight:bold;">{text}</p>', unsafe_allow_html=True)

def app():

    st.image("pics/contrib.png", width=800)
    st.header("[Contribute](https://github.com/u-amine/femicide_media_watch) to our open source project")
    blackText('The team members')
    col1, col2 = st.columns(2)
    col1.write('[Catalina Rojas](https://github.com/crojasu)')
    col2.write('[Amine Ünal](https://github.com/u-amine)')
    col1.write('[Soenke Bernhardi](https://github.com/sbernhardi)')
    col2.write('[Vera Montacuti](https://github.com/vmontacuti)')
