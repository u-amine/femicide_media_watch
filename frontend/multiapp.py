"""Frameworks for running multiple Streamlit applications as a single app.
"""

import streamlit as st

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func,
        })

    def run(self):
        #st.sidebar.title("Femicide Media Watch")
        st.sidebar.image("pics/Title.png", use_column_width=True)
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 200px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 200px;
                margin-left: -200px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        app = st.sidebar.radio('',
                           self.apps,
                           format_func=lambda app: app['title'])
        app['function']()
