# %matplotlib inline
# %config InlineBackend.figure_format='retina'
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from IPython.display import set_matplotlib_formats
import requests
import json
from GoogleNews import GoogleNews
set_matplotlib_formats('retina')

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
googlenews = GoogleNews(lang='en',period='d')


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
model.load_state_dict(torch.load('model/best_model_state.bin',map_location='cpu'))
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
for i in news_content:
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
