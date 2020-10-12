import bs4
import json
import time
import torch
import datetime
import requests
import numpy as np
import transformers
import pandas as pd
import seaborn as sns
import streamlit as st
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

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6
from bokeh.plotting import figure


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
rcParams['figure.figsize'] = 12, 8
googlenews = GoogleNews(lang='en')
url_rest = "https://api.twitter.com/1.1/search/tweets.json"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
class tqdm:
    def __init__(self, iterable, title=None):
        if title:
            st.write(title)
        self.prog_bar = st.progress(0)
        self.iterable = iterable
        self.length = len(iterable)
        self.i = 0

    def __iter__(self):
        for obj in self.iterable:
            yield obj
            self.i += 1
            current_prog = self.i / self.length
            self.prog_bar.progress(current_prog)


st.title('Public sentiments')
st.sidebar.title('User Inputs')

searchInput = st.sidebar.text_input('search query')
val = len(searchInput)
if val>0:
    agree = st.sidebar.checkbox('frequency')
    if agree:
        option = st.sidebar.selectbox('How would you like to be contacted?',('1h','1d','7d','1y'))
        googlenews.setperiod(option)
    else:
        st.sidebar.markdown('Select the time range for the search')
        dt1 = st.sidebar.date_input('from date',datetime.date.today())
        dt2 = st.sidebar.date_input('till date',datetime.date.today())
        if dt1>dt2:
            st.sidebar.error('SELECT A VALID "FROM" DATE')
        else:
            googlenews.setTimeRange(dt1,dt2)
    with st.spinner('Getting data...'):
        googlenews.search(searchInput)
        news_content = []

        ## ''' Google News start '''
        for i in range(1,1+1):
            googlenews.getpage(i)
            for i in googlenews.result():
                news_content.append(i['desc'])
            googlenews.clear()

        ## ''' Twitter handle '''
        q = '%40'+'#'+searchInput+' -filter:retweets -filter:replies'
        # count : no of tweets to be retrieved per one call and parameters according to twitter API
        params = {'q': q, 'count': 1000, 'lang': 'en',  'result_type': 'recent'}
        results = requests.get(url_rest, params=params, auth=auth)
        tweets = results.json()
        messages = [BeautifulSoup(tweet['text'], 'html.parser').get_text() for tweet in tweets['statuses']]
        # End
    st.success('Done!')

    finalList = []
    finalList.extend(news_content)
    finalList.extend(messages)



    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    class_names = ['negative', 'neutral', 'positive']
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    MAX_LEN = 160

    model = SentimentClassifier(len(class_names))
    model.load_state_dict(torch.load('model/best_model_state.bin',map_location='cpu'))      ## remove the map_location when running in GPU
    model = model.to(device)



    # for i in tqdm(range(200), title='tqdm style progress bar'):
    #     time.sleep(0.05)


    posCnt= 0
    netCnt= 0
    negCnt= 0
    for i in tqdm(finalList):
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

        if class_names[prediction] == 'negative':
            negCnt = negCnt+1
        elif class_names[prediction]== 'positive':
            posCnt = posCnt +1
        elif class_names[prediction]== 'neutral':
            netCnt = netCnt+1

    posCnt= round(posCnt/len(finalList)*100,2)
    netCnt= round(netCnt/len(finalList)*100,2)
    negCnt= round(negCnt/len(finalList)*100,2)

    cnames = ['positive', 'negative', 'neutral']
    cnt = [posCnt,negCnt,netCnt]

    chart_data = pd.DataFrame(columns=["label", "count"])
    chart_data['label'] = cnames
    chart_data['count'] = cnt
    st.bar_chart(cnt)
    # ax = chart_data.plot(kind='bar',x='label', y='count', rot=0)
    # st.bar_chart(ax)


else:
    st.error('Eneter a valid Search Query!...')
