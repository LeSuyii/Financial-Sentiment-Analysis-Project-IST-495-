import os
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd
import yfinance as yf
import numpy as np
import datetime
import matplotlib.pyplot as plt

# Load the FinBERT model for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# Function to get stock data for the past 5 days (1 week)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="5d")  # Get 5-day stock history
    df.reset_index(inplace=True)  # Reset index to use it as a column
    df.columns = df.columns.str.capitalize()  # Capitalize column names
    return df

# Function to fetch news content for a specific date
def fetch_news_content_for_date(ticker, target_date):
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)  # Send a request to the FinViz URL
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')  # Parse the HTML content
        news_table = soup.find(id='news-table')  # Find the news table by ID
        if news_table:
            rows = news_table.findAll('tr')  # Get all rows in the news table
            news_data = []
            for row in rows:
                try:
                    time = row.td.text.strip() if row.td else 'N/A'  # Extract time
                    headline = row.a.text.strip() if row.a else 'N/A'  # Extract headline
                    news_url = row.a['href'] if row.a else 'N/A'  # Extract news URL
                    news_date = time.split(' ')[0]  # Extract date from time
                    if news_date == target_date:  # Check if the date matches the target date
                        news_data.append((time, headline, news_url))
                except AttributeError as e:
                    print(f"AttributeError: {e}")
                    continue
            return news_data  # Return the news data
        return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news content: {e}")
        return []

# Function to fetch the content of a news article
def fetch_news_article_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)  # Send a request to the article URL
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')  # Parse the HTML content
        paragraphs = soup.findAll('p')  # Find all paragraph elements
        content = ' '.join([p.text for p in paragraphs if p.text])  # Concatenate all paragraph texts
        return content if content else None  # Return the content if not empty
    except requests.exceptions.RequestException as e:
        print(f"Error fetching article content: {e}")
        return None

# Function to analyze sentiment using BERT model
def analyze_sentiment_bert(text, chunk_size=512):
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]  # Split text into chunks
    sentiments = []
    scores = []
    for chunk in text_chunks:
        result = sentiment_pipeline(chunk)  # Analyze sentiment of each chunk
        sentiments.append(result[0]['label'])  # Extract sentiment label
        scores.append(result[0]['score'])  # Extract sentiment score
    return sentiments, scores

# Function to calculate article sentiment scores
def calculate_article_sentiment(sentiments, scores):
    positive_score = sum(score for sentiment, score in zip(sentiments, scores) if sentiment.lower() == 'positive')
    negative_score = sum(score for sentiment, score in zip(sentiments, scores) if sentiment.lower() == 'negative')
    return positive_score, negative_score

# Function to calculate stock trend sentiment
def calculate_stock_trend_sentiment(stock_data):
    stock_data['Price Change'] = stock_data['Close'].pct_change()  # Calculate price change percentage
    stock_trend_sentiment = stock_data['Price Change'].rolling(window=3).mean().fillna(0).mean()  # 3-day moving average of price change
    return stock_trend_sentiment

# Function to process multiple stock tickers from a CSV file
def process_multiple_tickers(csv_file, target_date):
    if not os.path.isfile(csv_file):
        print(f"File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)  # Read CSV file
    tickers = df['Ticker'].tolist()  # Get list of tickers
    data = {}

    for ticker in tickers:
        try:
            print(f"Processing {ticker}...")
            stock_data = get_stock_data(ticker)  # Get stock data
            stock_trend_sentiment = calculate_stock_trend_sentiment(stock_data)  # Calculate stock trend sentiment
            print(f"Stock trend sentiment for {ticker}: {stock_trend_sentiment}")

            news_data = fetch_news_content_for_date(ticker, target_date)  # Fetch news for the target date
            if not news_data:
                print(f"No news found for {ticker} on {target_date}.")
                data[ticker] = (0, stock_trend_sentiment, [])
                continue

            all_sentiments = []
            all_scores = []
            for news_time, headline, news_url in news_data:
                if news_url == 'N/A':
                    continue
                content = fetch_news_article_content(news_url)  # Fetch news article content
                if content:
                    sentiments, scores = analyze_sentiment_bert(content)  # Analyze sentiment of the article
                    all_sentiments.extend(sentiments)
                    all_scores.extend(scores)

            positive_score, negative_score = calculate_article_sentiment(all_sentiments, all_scores)  # Calculate sentiment scores
            article_sentiment = positive_score - negative_score
            print(f"Article sentiment for {ticker}: {article_sentiment}")

            news_titles_dates = [(headline, news_time) for news_time, headline, news_url in news_data]  # Store news headlines and dates
            data[ticker] = (article_sentiment, stock_trend_sentiment, news_titles_dates)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    return data

# Main program
csv_file = '/Users/csn/PycharmProjects/Fin3/finviz (4).csv'
target_date = 'Jul-24-24'  # Define the target date
data = process_multiple_tickers(csv_file, target_date)

# Output results
for ticker, (article_sentiment, stock_trend_sentiment, news_titles_dates) in data.items():
    sentiment_label = 'Positive' if article_sentiment > 0 else 'Negative'
    trend_label = 'Positive' if stock_trend_sentiment > 0 else 'Negative'
    print(f"{ticker}: Sentiment - {sentiment_label}, Stock Trend - {trend_label}")
    for title, date in news_titles_dates:
        print(f"    News Title: {title}, Date: {date}")

# Save output to CSV file
df = pd.DataFrame(data.items(), columns=['Ticker', 'Details'])
df[['Article Sentiment', 'Stock Trend Sentiment', 'News Titles and Dates']] = pd.DataFrame(df['Details'].tolist(), index=df.index)
df.drop(columns=['Details'], inplace=True)
df.to_csv('results.csv', index=False)
print("CSV file created successfully: results.csv")
