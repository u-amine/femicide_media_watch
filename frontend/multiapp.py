"""Frameworks for running multiple Streamlit applications as a single app.
"""

import streamlit as st


class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        st.sidebar.title("Femicide Media Watch")
        app = st.sidebar.radio('',
                           self.apps,
                           format_func=lambda app: app['title'])
        app['function']()
