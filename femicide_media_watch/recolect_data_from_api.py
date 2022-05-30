import requests
import sys
import urllib.parse
import csv
import json 

BASE_URI = "https://newsapi.org/v2/"
BASE_DOS="http://www.newsapi.ai/api/v1/article/getArticles?query=%7B%22%24query%22%3A%7B%22%24and%22%3A%5B%7B%22%24or%22%3A%5B%7B%22keyword%22%3A%22femicide%22%2C%22keywordLoc%22%3A%22body%22%7D%2C%7B%22keyword%22%3A%22feminicide%22%2C%22keywordLoc%22%3A%22body%22%7D%5D%7D%2C%7B%22lang%22%3A%22eng%22%7D%5D%7D%2C%22%24filter%22%3A%7B%22forceMaxDataTimeWindow%22%3A%2231%22%2C%22dataType%22%3A%5B%22news%22%5D%7D%7D&resultType=articles&articlesSortBy=date&articlesCount=1&articleBodyLen=-1&apiKey=5c49c401-d8f5-4356-ab12-8529c3741b48"
API="3a0496ae382c440fa364fb94add8e35a"

def api_call():
   # query='murderer'
    #url = urllib.parse.urljoin(BASE_URI, f"everything?q={query}&apiKey={API}")
    news = requests.get(BASE_DOS).json()
    print(news["articles"]["results"][0]["title"])
    
    # create the csv writer
    with open('noticiaas.csv', 'w') as file:
    # 2. Create a CSV writer
        writer = csv.writer(file)
        # 3. Write data to the file
        writer.writerow(news["articles"]["results"])
    return news

if __name__ == '__main__':
    news = api_call()
    #print(news["articles"]["results"][0]["body"])