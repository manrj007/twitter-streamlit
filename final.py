import bs4
import json
import torch
import requests
import numpy as np
import transformers
import pandas as pd
import seaborn as sns
from matplotlib import rc
from textwrap import wrap
from pylab import rcParams
from torch import nn, optim
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from GoogleNews import GoogleNews
from collections import defaultdict
from requests_oauthlib import OAuth1
from torch.utils.data import Dataset, DataLoader
from IPython.display import set_matplotlib_formats
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup



RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
rcParams['figure.figsize'] = 12, 8
googlenews = GoogleNews(lang='en',period='d')
url_rest = "https://api.twitter.com/1.1/search/tweets.json"
# sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
# sns.set(style='whitegrid', palette='muted', font_scale=1.2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set_matplotlib_formats('retina')
# %matplotlib inline
# %config InlineBackend.figure_format='retina'


auth_params = {
    'app_key':'zVuMpGebsDOlKvn2TxZbieF6A',
    'app_secret':'ilxtHZJMKpE2txq3erWfORNzCaa6DQxJULEQVlt6sqvPVl9Dm5',
    'oauth_token':'1059637136511520768-IsGRaBmmUQMAXgW97vBmqysqFM80db',
    'oauth_token_secret':'jgbnA1KOpNId91tsgZpvRkr1aVbtGfSDTKONH29O1OQrN'
                }
auth = OAuth1 (
    auth_params['app_key'],
    auth_params['app_secret'],
    auth_params['oauth_token'],
    auth_params['oauth_token_secret']
)


PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
class_names = ['negative', 'neutral', 'positive']
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
MAX_LEN = 160

class SentimentClassifier(nn.Module):
      def __init__(self, n_classes):
          super(SentimentClassifier, self).__init__()
          self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
          self.drop = nn.Dropout(p=0.3)
          self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
      def forward(self, input_ids, attention_mask):
          _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask)
          output = self.drop(pooled_output)
          return self.out(output)


model = SentimentClassifier(len(class_names))
model.load_state_dict(torch.load('model/best_model_state.bin',map_location='cpu'))      ## remove the map_location when running in GPU
model = model.to(device)

# review_text = input('Enter the review you want to check:\n')

## Google News start

news_content = []
searchInput = input('Enter the search keyword:\n')
googlenews.search(searchInput)
for i in range(1,1+1):
    googlenews.getpage(i)
    for i in googlenews.result():
        news_content.append(i['desc'])
    googlenews.clear()

## End

### Twitter handle
q = '%40'+searchInput+' -filter:retweets -filter:replies' # Twitter handle of Amazon India

# count : no of tweets to be retrieved per one call and parameters according to twitter API
params = {'q': q, 'count': 1000, 'lang': 'en',  'result_type': 'recent'}
results = requests.get(url_rest, params=params, auth=auth)

tweets = results.json()
messages = [BeautifulSoup(tweet['text'], 'html.parser').get_text() for tweet in tweets['statuses']]
## End
### Save the scraped messages
encoded_unicode = []
for i in messages:
    encoded_unicode.append(i.encode("utf8"))
a_file = open("textfile.txt", "wb")
for i in encoded_unicode:
    a_file.write(i)
print(len(messages))
# messages.extend(news_content)
# news_content.extend(messages)

for i in messages:
    encoded_review = tokenizer.encode_plus(i,
                                           max_length=MAX_LEN,
                                           add_special_tokens=True,
                                           return_token_type_ids=False,
                                           pad_to_max_length=True,
                                           return_attention_mask=True,
                                           return_tensors='pt',
                                           )

    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)

    print(f'Review text: {i}')
    print(f'Sentiment  : {class_names[prediction]}')
