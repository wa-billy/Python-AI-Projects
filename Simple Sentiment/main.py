from textblob import TextBlob
# from newspaper import Article
# import nltk

# url = 'https://www.cnbc.com/2025/10/02/cnbc-daily-open-unpleasant-news-from-the-us-appears-to-be-making-investors-cheery.html'
# article = Article(url)

# article.download()
# article.parse()
# article.nlp()

# text = article.summary
# print(text)

with open('Simple Sentiment/text.txt') as f:
    text = f.read()

blob = TextBlob(text)
sentiment = blob.sentiment.polarity
print(sentiment)