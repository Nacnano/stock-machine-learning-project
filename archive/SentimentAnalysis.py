from textblob import TextBlob
import pandas as pd 
import json
from tqdm import tqdm

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity 

def analyze_news_sentiment(news_data):
    sentiments = []
    for news in tqdm(news_data):
        headline = news['headline']
        short_description = news['short_description']
        combined_text = headline + ' ' + short_description
        sentiment = analyze_sentiment(combined_text)
        sentiments.append(sentiment)
    return sentiments

def read_json_file(filepath):
    news_data = [] 
    for line in open(filepath, 'r'): 
        try: 
            news_data += [json.loads(line)] 
        except: 
            continue # Skip errors in JSON file.
    return news_data

filepath = "data/News_Category_Dataset_v3/News_Category_Dataset_v3.json"
news_data = read_json_file(filepath)
polarities = analyze_news_sentiment(news_data)

df = pd.DataFrame(news_data)
df['polarity'] = polarities

output_csv_file = "data/News_Category_Dataset_v3/news_sentiment.csv"
df.to_csv(output_csv_file, index=False)
