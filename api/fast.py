from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Article(BaseModel):
    text: str

@app.post("/predict")
async def predict(article: Article):
    return {"prediction": True,
            "text": article.text}

@app.get("/latest")
async def predict():
    return {"articles": {
        "results": [
            {
                "uri": "7062666379",
                "lang": "eng",
                "isDuplicate": False,
                "date": "2022-06-01",
                "time": "09:17:00",
                "dateTime": "2022-06-01T09:17:00Z",
                "dateTimePub": "2022-06-01T09:17:00Z",
                "dataType": "news",
                "sim": 0,
                "url": "https://wsau.com/2022/06/01/underrated-soul-man/",
                "title": "Underrated Soul Man",
                "body": "We continue our look back at the music of 50 years ago.....\n\nTake the grooves of Al Green...mix in the funk of James Brown....add the backing of the pros from Muscle Shoals and you have a tasty slice of Southern soul from Bobby Womack...his fourth LP, \"Understanding\" released in 1972.\n\nWomack maybe doesnt get the press and recognition of those other guys...but he could bring it and this LP shows it.\n\nIt includes his big hit \"Woman's Got To Have It\" which went #1 on the R&B chart. It got a nice cover from James Taylor a few years later....\n\nCovers on this record include a version of Neil Diamond's \"Sweet Caroline\" which cracked the Top 40 Pop charts, and as soulful a version of a Beatles song ever put on vinyl with his take on \"And I Love Her\"\n\nWomack also wrote alot of this record including standouts like \"Simple Man\" which features a driving backing track from David Hood, Roger Hawkins, Jimmy Johnson, Barry Beckett and the rest of the Swampers from Muscle Shoals.\n\nYou can hear the Green and Brown influence on the track \"Got To Get You Back\". And the song that leads off the record, \"I Can Understand It\", really gets down...and became very popular in the underground and gay nightclubs springing up in the Northeast, which, of course, led to the \"disco\"craze and few years later.\n\nHe also does a slow jam called \"Hairy Hippie\" dedicated to his brother Harry, who played bass, and was later stabbed by his girlfriend. It became an unlikely hit.\n\nWomack's other brothers sing background on the record as do Patrice Holloway and actress Pam Grier.\n\nHe may not be remembered by as many folks as Reverend Al or the Godfather of Soul but he delivers a high energy, solid performance both live and on record. Check it out!",
                "source": {
                    "uri": "wsau.com",
                    "dataType": "news",
                    "title": "WSAU News/Talk 550 AM · 99.9 FM | Wausau, Stevens Point"
                },
                "authors": [
                    {
                        "uri": "tom_king@wsau.com",
                        "name": "Tom King",
                        "type": "author",
                        "isAgency": False
                    }
                ],
                "image": "https://media.socastsrm.com/wordpress/wp-content/blogs.dir/2272/files/2020/07/Tom-1.jpg",
                "eventUri": '',
                "sentiment": 0.2156862745098038,
                "wgt": 391771020,
                "relevance": 2
            }]}}
