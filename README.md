# Financial-Sentiment-Analysis-Project-IST-495-

# Stock News Sentiment Analysis

This project is designed to fetch the latest news for a given stock ticker from FinViz, analyze the sentiment of the news articles using a pre-trained BERT model, and plot the stock price changes around the news publication date.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Project Structure](#project-structure)
- [License](#license)

## Introduction

Stock News Sentiment Analysis is a tool that helps investors and traders analyze the sentiment of news articles related to specific stocks. By leveraging the power of natural language processing (NLP) with a pre-trained BERT model, the tool can provide insights into the potential impact of news on stock prices.

## Features

- Fetches the latest news for a given stock ticker from FinViz.
- Analyzes the sentiment of news articles using a pre-trained BERT model.
- Plots the stock price changes around the news publication date.

## Requirements

To run this project, you need the following Python packages:

- `requests`
- `beautifulsoup4`
- `transformers`
- `yfinance`
- `matplotlib`

## Installation

Follow these steps to install the required packages and set up the project:

1. **Clone the repository:**

   ```sh
   git clone https://github.com/your-username/stock-news-sentiment-analysis.git
   cd stock-news-sentiment-analysis
2. Create a virtual environment (optional but recommended):
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install the required packages:
   pip install -r requirements.txt




Usage

To run the sentiment analysis script, follow these steps:

1. Navigate to the project directory:
cd stock-news-sentiment-analysis

2. Run the script:
   python sentiment_analysis.py


Example

The following example demonstrates how to fetch the latest news for Tesla (TSLA), analyze the sentiment of the news articles, and plot the stock price changes:

1. Set the ticker symbol in the script (default is TSLA):
ticker = 'TSLA'

3. Run the script:
python sentiment_analysis.py
The script will output the latest news articles for the specified ticker, analyze their sentiment, and plot the stock price changes around the news publication date.

Project Structure

The project directory structure is as follows:
stock-news-sentiment-analysis/
├── .gitignore
├── README.md
├── requirements.txt
└── sentiment_analysis.py

.gitignore: Specifies files and directories to be ignored by Git.
README.md: Contains the documentation for the project.
requirements.txt: Lists the required Python packages.
sentiment_analysis.py: The main script that performs sentiment analysis on stock news.

License

This project is licensed under the MIT License. See the LICENSE file for details.


### `requirements.txt`
Create a requirements.txt file listing the project dependencies:

```plaintext
requests
beautifulsoup4
transformers
yfinance
matplotlib

Make sure your project directory structure is as follows:
stock-news-sentiment-analysis/
├── .gitignore
├── README.md
├── requirements.txt
└── sentiment_analysis.py


Example .gitignore
__pycache__/
*.pyc
.venv/
.env
.DS_Store

sentiment_analysis.py
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import datetime
import yfinance as yf
import matplotlib.pyplot as plt

# Specify the model explicitly for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", revision="af0f99b")

# Function to fetch the latest news content from FinViz for a given stock ticker
def fetch_latest_news_content(ticker):
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)  # Send a request to the FinViz URL
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')  # Parse the HTML content
        news_table = soup.find(id='news-table')  # Find the news table by ID
        news_list = []

        if news_table:
            rows = news_table.findAll('tr')  # Get all rows in the news table
            if rows:
                # Get the date of the most recent news
                last_news_date = rows[0].td.text.strip().split(' ')[0]
                for row in rows:
                    time = row.td.text.strip()  # Extract time
                    headline = row.a.text.strip()  # Extract headline
                    news_url = row.a['href']  # Extract news URL
                    news_date = time.split(' ')[0]  # Extract date from time

                    # Only get news from the most recent date
                    if news_date == last_news_date:
                        news_list.append((time, headline, news_url))
                    else:
                        break

        return news_list  # Return the list of news
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
        content = ' '.join([p.text for p in paragraphs])  # Concatenate all paragraph texts
        return content if content else None  # Return the content if not empty
    except requests.exceptions.RequestException as e:
        print(f"Error fetching article content: {e}")
        return None

# Function to analyze sentiment using the BERT model
def analyze_sentiment_bert(text, chunk_size=512):
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]  # Split text into chunks
    sentiments = []
    scores = []
    for chunk in text_chunks:
        result = sentiment_pipeline(chunk)  # Analyze sentiment of each chunk
        sentiments.append(result[0]['label'])  # Extract sentiment label
        scores.append(result[0]['score'])  # Extract sentiment score
    return sentiments, scores

# Function to split text into lines for readability
def split_text_into_lines(text, line_length=150):
    return '\n'.join([text[i:i + line_length] for i in range(0, len(text), line_length)])

# Function to fetch stock price changes around the news publication date
def fetch_stock_price_change(ticker, news_time):
    try:
        stock = yf.Ticker(ticker)  # Create a Ticker object
        news_date = datetime.datetime.strptime(news_time, "%b-%d-%y %I:%M%p")  # Parse the news time
        start_date = news_date - datetime.timedelta(days=7)  # Define the start date (7 days before the news date)
        end_date = news_date + datetime.timedelta(days=7)  # Define the end date (7 days after the news date)
        hist = stock.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))  # Fetch historical stock data
        return hist
    except Exception as e:
        print(f"Error fetching stock price data: {e}")
        return None

def main():
    ticker = 'TSLA'
    news_list = fetch_latest_news_content(ticker)  # Fetch the latest news for the ticker

    if news_list:
        for time, headline, news_url in news_list:
            print(f"Time: {time}")
            print(f"Headline: {headline}")
            print(f"URL: {news_url}")
            content = fetch_news_article_content(news_url)  # Fetch the content of the news article
            if content:
                print(f"Content:\n{split_text_into_lines(content)}")

                sentiments, scores = analyze_sentiment_bert(content)  # Analyze the sentiment of the article

                # Calculate the overall sentiment scores
                positive_score = sum(score for sentiment, score in zip(sentiments, scores) if sentiment == 'POSITIVE')
                negative_score = sum(score for sentiment, score in zip(sentiments, scores) if sentiment == 'NEGATIVE')
                total_score = positive_score + negative_score

                if total_score > 0:
                    print(f"Positive Score: {positive_score:.2f}")
                    print(f"Negative Score: {negative_score:.2f}")

                    # Fetch and display stock price change
                    hist = fetch_stock_price_change(ticker, time)  # Fetch stock price data around the news date
                    if hist is not None:
                        hist['Close'].plot(title=f'Stock Price Around News Date for {ticker}', figsize=(10, 6))
                        plt.axvline(x=datetime.datetime.strptime(time, "%b-%d-%y %I:%M%p"), color='red', linestyle='--', label='News Time')
                        plt.legend()
                        plt.show()
                    else:
                        print("Failed to fetch stock price data.")
                else:
                    print("No sentiment detected in the text.")
            else:
                print("Failed to fetch news content.")
    else:
        print(f"No news found for {ticker}.")

if __name__ == '__main__':
    main()
