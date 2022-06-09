import streamlit as st
from multiapp import MultiApp
from apps import home
from apps import predicter
from apps import newsFeed
from apps import contribute

st.set_page_config(layout="wide")

apps = MultiApp()

# Add application here
apps.add_app("Home", home.app)
apps.add_app("Media Watch", newsFeed.app)
apps.add_app("Media Predicter", predicter.app)
apps.add_app("The code behind", contribute.app)

apps.run()
