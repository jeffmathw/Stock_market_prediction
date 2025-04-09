from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F  


tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1).detach().numpy()[0]  

    sentiment_labels = ["negative", "neutral", "positive"]
    sentiment_scores = {label: float(prob) for label, prob in zip(sentiment_labels, probabilities)}

    # Compute compound score: positive - negative
    compound_score = sentiment_scores["positive"] - sentiment_scores["negative"]
    
    return sentiment_scores, compound_score


columns = ['datetime', 'title', 'source', 'link', 'negative_score', 'neutral_score', 'positive_score', 'compound_score']
df = pd.DataFrame(columns=columns)

c = 0
for page in range(1, 430):
    url = f'https://markets.businessinsider.com/news/amzn-stock?p={page}'
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, "lxml")
    print(c)
    c+=1

    articles = soup.find_all("div", class_="latest-news__story")
    for article in articles:
        datetime = article.find("time", class_="latest-news__date").get("datetime")
        title = article.find("a", class_="news-link").text.strip()
        source = article.find("span", class_="latest-news__source").text.strip()
        link = article.find("a", class_="news-link").get("href")

        sentiment_scores, compound_score = analyze_sentiment(title)

        df = pd.concat([pd.DataFrame([[datetime, title, source, link, 
                                       sentiment_scores["negative"], 
                                       sentiment_scores["neutral"], 
                                       sentiment_scores["positive"],
                                       compound_score]],
                                     columns=df.columns), df], ignore_index=True)
        c += 1

print(f'{c} headlines scraped')
df.to_csv('/Users/jeffrinmathew/Desktop/folder/AMZNstocknews.csv', index=False)


df = pd.read_csv('/Users/jeffrinmathew/Desktop/folder/AMZNstocknews.csv')


df['datetime'] = pd.to_datetime(df['datetime']).dt.date

# Group by date and calculate the mean compound score for each day
mean_df = df.groupby('datetime')['compound_score'].mean()
print(mean_df)

# Plot the bar chart
plt.figure(figsize=(12, 6))
mean_df.plot(kind='bar', color=['red' if x < 0 else 'green' for x in mean_df])

# Formatting the graph
plt.xlabel("Date")
plt.ylabel("Average Compound Sentiment Score (-1 to 1)")
plt.title("Daily Average Sentiment Score Over Time of ")
plt.xticks(rotation=45)  
plt.axhline(y=0, color='black', linestyle='--')  
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()



