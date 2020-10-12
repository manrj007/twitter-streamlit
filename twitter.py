import requests
import bs4
from bs4 import BeautifulSoup
from requests_oauthlib import OAuth1

auth_params = {
    'app_key':'zVuMpGebsDOlKvn2TxZbieF6A',   ## 'API Key'
    'app_secret':'ilxtHZJMKpE2txq3erWfORNzCaa6DQxJULEQVlt6sqvPVl9Dm5',  ## 'API secret'
    'oauth_token':'1059637136511520768-IsGRaBmmUQMAXgW97vBmqysqFM80db', ## 'Access token'
    'oauth_token_secret':'jgbnA1KOpNId91tsgZpvRkr1aVbtGfSDTKONH29O1OQrN'    ## 'Access token secret'
}

auth = OAuth1 (
    auth_params['app_key'],
    auth_params['app_secret'],
    auth_params['oauth_token'],
    auth_params['oauth_token_secret']
)

# url according to twitter API
url_rest = "https://api.twitter.com/1.1/search/tweets.json"


# getting rid of retweets in the extraction results and filtering all replies to the tweet often uncessary for the analysis
tweetSearch = input('Enter the search query')
q = '%40'+tweetSearch+' -filter:retweets -filter:replies' # Twitter handle of Amazon India

# count : no of tweets to be retrieved per one call and parameters according to twitter API
params = {'q': q, 'count': 1000, 'lang': 'en',  'result_type': 'recent'}
results = requests.get(url_rest, params=params, auth=auth)

tweets = results.json()
messages = [BeautifulSoup(tweet['text'], 'html5lib').get_text() for tweet in tweets['statuses']]
print(messages)

### Save the scraped messages
encoded_unicode = []
for i in messages:
    encoded_unicode.append(i.encode("utf8"))
a_file = open("textfile.txt", "wb")
for i in encoded_unicode:
    a_file.write(i)
