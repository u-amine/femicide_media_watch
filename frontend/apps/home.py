import streamlit as st
import pandas as pd
import numpy as np

def missionTitle(text):
    st.markdown(f'<h1 style=text-align: center;color:#0B0B0B;font-size:70px;font-family:Verdana;border-radius:2%;font-weight:bold;">{text}</h1>', unsafe_allow_html=True)

def missionText1(text):
    st.markdown(f'<p style=text-align: center;color:#0B0B0B;font-size:50px;font-family:Verdana;border-radius:2%;">{text}</p>', unsafe_allow_html=True)


def app():

    st.image("pics/Title.png", width=700)

    missionTitle("Our Mission")


    missionText1("""We have set out to apply the wonders of modern Machine Learning to raise awareness on the thousands of women murdered every year just because of their gender – otherwise known as Femicide.
                 \n\n Since there is still no internationally accepted definition of the term femicide,many of these cases often get lost in the general news stream, not detectable as what they are.
                 \n\nThis is where Data Science comes into play. We, a team of four Data Scientists, have built an algorithm that is able to detect news articles reporting cases of femicide although without being explicitly classified as such.\n\n
                 \n\nThis tool aims to serve institutions and organizations that want to collect accurate and reliable data on femicide cases, and anyone who wants to get up-to-date information on the status of femicides around the world.

                There are two parts to our tool - check them out below👇🏾
""")

    st.markdown("***")


    col1, col2  = st.columns(2)
    col2.subheader('Femicide Media Predicter: Simply input any text and we will tell you if it talks about a femicide case or not')
    col1.subheader('Femicide Media Watch: Explore the latest articles on femicide cases around the world. Always up to date.')

    col2.button(label='Femicide Media Predicter')
    col1.button(label='Femicide Media Watch')
