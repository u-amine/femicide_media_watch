FROM python:3.8.6-buster

EXPOSE 8080

COPY .streamlit /.streamlit
COPY apps /apps
COPY pics /pics
COPY app.py /app.py
COPY multiapp.py /multiapp.py
COPY requirements.txt /requirements.txt

RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]
