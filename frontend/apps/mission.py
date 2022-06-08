import streamlit as st
import pandas as pd
import numpy as np

def header(url):
     st.markdown(f'<p style=color:#9B1146;font-size:24px;font-family:Verdana;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

def blackText(text):
    st.markdown(f'<p style=color:#0B0B0B;font-size:20px;font-family:Verdana;border-radius:2%;">{text}</p>', unsafe_allow_html=True)

def purpleText(text):
    st.markdown(f'<p style=color:#9B1146;font-size:20px;font-family:Verdana;border-radius:2%;">{text}</p>', unsafe_allow_html=True)

def underlineText(text):
    st.markdown(f'<p style=color:#0B0B0B;font-size:20px;font-family:Verdana;border-radius:2%;text-decoration:underline;font-weight:bold;">{text}</p>', unsafe_allow_html=True)

def app():

    st.image("../frontend/mission.png", width=800)
    purpleText('This is just a placeholder text')
    purpleText('This is just a placeholder text')
    purpleText('This is just a placeholder text')
    purpleText('This is just a placeholder text')
    purpleText('This is just a placeholder text')
    purpleText('This is just a placeholder text')
