import requests
import json
from GoogleNews import GoogleNews
googlenews = GoogleNews(lang='en',period='d')


news_content = []
searchInput = input('Enter the search keyword:\n')
googlenews.search(searchInput)
for i in range(1,1+1):
    googlenews.getpage(i)
    for i in googlenews.result():
        news_content.append(i['desc'])
    googlenews.clear()

#
# testData = [y for x in news_content for y in x]
print(news_content)
