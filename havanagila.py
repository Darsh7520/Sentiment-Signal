import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from transformers import pipeline
import feedparser
import urllib.parse
from datetime import datetime, timedelta
import re

# --- 1. MODEL LOADING ---
def load_models():
    """
    Loads FinBERT (Financial) and RoBERTa (Social/Nuance)
    """
    print("Loading sentiment models...")
    try:
        finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        roberta = pipeline("sentiment-analysis", 
                           model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                           tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                           top_k=None)
        print("Models loaded successfully!")
        return finbert, roberta
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

finbert, roberta = load_models()

# --- 2. SENTIMENT SCORING ENGINE ---
def get_sentiment_score(text):
    if not text: return 0
    text = text[:512]
    
    # FinBERT Score (News Focused)
    fb_res = finbert(text)[0]
    fb_map = {"positive": 1, "neutral": 0, "negative": -1}
    fb_score = fb_map.get(fb_res['label'], 0) * fb_res['score']
    
    # RoBERTa Score (Nuance/Context Focused)
    rob_res = roberta(text)[0]
    rob_pos = next((x['score'] for x in rob_res if x['label'] == 'positive'), 0)
    rob_neg = next((x['score'] for x in rob_res if x['label'] == 'negative'), 0)
    rob_score = rob_pos - rob_neg
    
    # Average them
    return (fb_score + rob_score) / 2

# --- 3. HISTORICAL NEWS FETCHER (Google News RSS with date filter) ---
def fetch_historical_news(ticker, days=90):
    """
    Fetch historical news from Google News RSS with date filtering.
    Returns list of {date, title, source} sorted by date.
    """
    news_items = []
    
    # Clean ticker for search (remove .NS, .BO suffixes for broader search)
    clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
    
    # Google News RSS with time filter (when:Nd for N days)
    try:
        query = urllib.parse.quote(f"{clean_ticker} stock")
        rss_url = f"https://news.google.com/rss/search?q={query}+when:{days}d&hl=en-IN&gl=IN&ceid=IN:en"
        
        print(f"Fetching news from Google News RSS for {clean_ticker}...")
        feed = feedparser.parse(rss_url)
        
        for entry in feed.entries:
            # Parse the publication date
            try:
                pub_date = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %Z")
            except:
                try:
                    # Alternative format
                    pub_date = datetime(*entry.published_parsed[:6])
                except:
                    continue
            
            # Extract source from title (format: "Title - Source")
            title = entry.title
            source = "Unknown"
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0]
                source = parts[1] if len(parts) > 1 else "Unknown"
            
            news_items.append({
                "date": pub_date.date(),
                "title": title,
                "source": source,
                "datetime": pub_date
            })
            
    except Exception as e:
        print(f"Error fetching news: {e}")
    
    print(f"Fetched {len(news_items)} news articles from the past {days} days")
    return sorted(news_items, key=lambda x: x['datetime'])

# --- 4. AGGREGATE DAILY SENTIMENT ---
def calculate_daily_sentiment(news_items):
    """
    Calculate sentiment scores for each news item and aggregate by date.
    Returns DataFrame with date and average sentiment.
    """
    if not news_items:
        return pd.DataFrame()
    
    print("\nAnalyzing sentiment for each headline...")
    
    # Score each news item
    scored_items = []
    for i, item in enumerate(news_items):
        score = get_sentiment_score(item['title'])
        scored_items.append({
            'date': item['date'],
            'title': item['title'],
            'source': item['source'],
            'sentiment': score
        })
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Analyzed {i + 1}/{len(news_items)} headlines...")
    
    # Create DataFrame and aggregate by date
    df = pd.DataFrame(scored_items)
    daily_sentiment = df.groupby('date').agg({
        'sentiment': 'mean',
        'title': 'count'  # Number of articles
    }).rename(columns={'title': 'article_count'})
    
    daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
    
    return daily_sentiment, df

# --- 5. BACKTESTING ENGINE (Real Sentiment) ---
def generate_backtest(ticker, daily_sentiment, days=90):
    """
    Backtest using real historical sentiment scores.
    """
    # Fetch Real Price
    end = datetime.now()
    start = end - timedelta(days=days)
    price_df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
    
    if price_df.empty:
        return pd.DataFrame()
    
    # Merge sentiment with price data
    price_df = price_df.copy()
    
    # Normalize price index to date only (no timezone)
    price_df.index = pd.to_datetime(price_df.index).normalize()
    if price_df.index.tz is not None:
        price_df.index = price_df.index.tz_localize(None)
    
    # Map sentiment data to price dates
    if not daily_sentiment.empty:
        # Create a sentiment lookup dictionary
        sentiment_dict = {}
        for idx, row in daily_sentiment.iterrows():
            date_key = pd.to_datetime(idx).normalize()
            if hasattr(date_key, 'tz') and date_key.tz is not None:
                date_key = date_key.tz_localize(None)
            sentiment_dict[date_key] = row['sentiment']
        
        # Map sentiment to price dates
        price_df['sentiment'] = price_df.index.map(lambda x: sentiment_dict.get(x, np.nan))
        # Forward fill sentiment for days without news
        price_df['sentiment'] = price_df['sentiment'].ffill().fillna(0)
    else:
        price_df['sentiment'] = 0
    
    # Trading Strategy based on REAL sentiment
    cash = 100000
    position = 0
    portfolio = []
    signals = []
    
    buy_threshold = 0.15
    sell_threshold = -0.15
    
    for i in range(len(price_df)):
        price = float(price_df['Close'].iloc[i])
        score = float(price_df['sentiment'].iloc[i])
        
        if score > buy_threshold and position == 0:
            position = cash / price
            cash = 0
            signals.append("Buy")
        elif score < sell_threshold and position > 0:
            cash = position * price
            position = 0
            signals.append("Sell")
        else:
            signals.append(None)
            
        val = cash + (position * price)
        portfolio.append(val)
        
    price_df['Strategy'] = portfolio
    price_df['BuyHold'] = (100000 / float(price_df['Close'].iloc[0])) * price_df['Close']
    price_df['Signal'] = signals
    
    return price_df

# --- 6. MAIN EXECUTION ---
def run_analysis(ticker="TATASTEEL.NS", days=90):
    print("\n" + "="*70)
    print(f"📊 SENTINANCE HISTORICAL BACKTEST: {ticker}")
    print("="*70)
    
    # --- FETCH HISTORICAL NEWS ---
    news = fetch_historical_news(ticker, days)
    
    if not news:
        print("No historical news found. Cannot proceed with backtest.")
        return
    
    # --- CALCULATE DAILY SENTIMENT ---
    daily_sentiment, scored_df = calculate_daily_sentiment(news)
    
    # --- DISPLAY TOP NEWS BY SENTIMENT ---
    print("\n" + "-"*70)
    print("TOP 5 MOST POSITIVE NEWS:")
    print("-"*70)
    top_positive = scored_df.nlargest(5, 'sentiment')
    for _, row in top_positive.iterrows():
        print(f"  [{row['date']}] Score: {row['sentiment']:+.3f}")
        print(f"    {row['title'][:70]}...")
    
    print("\n" + "-"*70)
    print("TOP 5 MOST NEGATIVE NEWS:")
    print("-"*70)
    top_negative = scored_df.nsmallest(5, 'sentiment')
    for _, row in top_negative.iterrows():
        print(f"  [{row['date']}] Score: {row['sentiment']:+.3f}")
        print(f"    {row['title'][:70]}...")
    
    # --- AGGREGATE STATS ---
    avg_sentiment = scored_df['sentiment'].mean()
    print("\n" + "-"*70)
    print(f"📈 OVERALL SENTIMENT STATS ({len(scored_df)} articles):")
    print(f"   Average Sentiment: {avg_sentiment:+.3f}")
    print(f"   Positive Articles: {len(scored_df[scored_df['sentiment'] > 0.1])}")
    print(f"   Negative Articles: {len(scored_df[scored_df['sentiment'] < -0.1])}")
    print(f"   Neutral Articles:  {len(scored_df[abs(scored_df['sentiment']) <= 0.1])}")
    
    if avg_sentiment > 0.15:
        verdict = "BUY"
    elif avg_sentiment < -0.15:
        verdict = "SELL"
    else:
        verdict = "HOLD"
    
    print(f"   VERDICT: {verdict}")
    print("-"*70)
    
    # --- BACKTESTING ---
    print("\n🧪 Running backtest with REAL historical sentiment...")
    data = generate_backtest(ticker, daily_sentiment, days)
    
    if not data.empty:
        # Calculate final returns
        strategy_return = ((data['Strategy'].iloc[-1] - 100000) / 100000) * 100
        buyhold_return = ((data['BuyHold'].iloc[-1] - 100000) / 100000) * 100
        
        print(f"\n📊 BACKTEST RESULTS ({days} days):")
        print(f"   AI Sentiment Strategy Return: {strategy_return:+.2f}%")
        print(f"   Buy & Hold Return:            {buyhold_return:+.2f}%")
        print(f"   Alpha Generated:              {strategy_return - buyhold_return:+.2f}%")
        
        # Count trades
        buy_count = len(data[data['Signal'] == 'Buy'])
        sell_count = len(data[data['Signal'] == 'Sell'])
        print(f"   Total Trades:                 {buy_count} buys, {sell_count} sells")
        
        # --- MATPLOTLIB PLOT ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
        plt.style.use('dark_background')
        
        # Plot 1: Strategy Performance
        ax1.plot(data.index, data['Strategy'], label='AI Sentiment Strategy', color='#00FF00', linewidth=2)
        ax1.plot(data.index, data['BuyHold'], label='Buy & Hold', color='gray', linestyle='--', linewidth=1.5)
        
        # Plot buy/sell markers
        buys = data[data['Signal'] == 'Buy']
        sells = data[data['Signal'] == 'Sell']
        
        ax1.scatter(buys.index, buys['Strategy'], marker='^', color='lime', s=100, label='Buy Signal', zorder=5)
        ax1.scatter(sells.index, sells['Strategy'], marker='v', color='red', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title(f'Real Sentiment Strategy vs Benchmark - {ticker} ({days} Days)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add return annotation
        ax1.annotate(f'Sentiment Strategy: {strategy_return:+.2f}%', 
                     xy=(0.02, 0.95), xycoords='axes fraction',
                     fontsize=11, color='#00FF00', fontweight='bold')
        ax1.annotate(f'Buy & Hold: {buyhold_return:+.2f}%', 
                     xy=(0.02, 0.90), xycoords='axes fraction',
                     fontsize=11, color='gray')
        
        # Plot 2: Daily Sentiment
        if not daily_sentiment.empty:
            colors = ['#00FF00' if x > 0 else '#FF4444' for x in daily_sentiment['sentiment']]
            ax2.bar(daily_sentiment.index, daily_sentiment['sentiment'], color=colors, alpha=0.7, width=1)
            ax2.axhline(y=0, color='white', linestyle='-', linewidth=0.5)
            ax2.axhline(y=0.15, color='green', linestyle='--', linewidth=0.5, alpha=0.5, label='Buy Threshold')
            ax2.axhline(y=-0.15, color='red', linestyle='--', linewidth=0.5, alpha=0.5, label='Sell Threshold')
            ax2.set_title('Daily Sentiment Score (from Real News)', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Sentiment', fontsize=12)
            ax2.legend(loc='upper left', fontsize=9)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    else:
        print("No price data available for backtesting.")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)

# --- RUN ---
if __name__ == "__main__":
    ticker = input("Enter the stock ticker (e.g., TCS.NS): ").strip() or "TATASTEEL.NS"
    days_input = input("Enter number of past days to fetch (default 90): ").strip()
    try:
        days = int(days_input) if days_input else 90
    except ValueError:
        print("Invalid number of days, using default 90.")
        days = 90
    run_analysis(ticker=ticker, days=days)