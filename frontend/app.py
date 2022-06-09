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
apps.add_app("FMW Predicter", predicter.app)
apps.add_app("FMW News Feed", newsFeed.app)
apps.add_app("Contribute Now!", contribute.app)

apps.run()
