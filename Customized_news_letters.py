import streamlit as st
import feedparser
from newspaper import Article
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch

# Summarization model
@st.cache_resource
def load_summarizer():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)

# --- User Profiles ---
user_profiles = {
    "Alex Parker": {
        "interests": ["AI", "cybersecurity", "blockchain", "startups", "programming"],
        "sources": [
            "https://techcrunch.com/feed/",
            "https://www.wired.com/feed/rss",
            "https://arstechnica.com/feed/",
            "https://www.technologyreview.com/feed/"
        ]
    },
    "Priya Sharma": {
        "interests": ["global markets", "startups", "fintech", "cryptocurrency", "economics"],
        "sources": [
            "https://www.bloomberg.com/feed/podcast",
            "https://www.ft.com/?format=rss",
            "https://www.forbes.com/investing/feed/",
            "https://www.coindesk.com/arc/outboundfeeds/rss/"
        ]
    },
    "Marco Rossi": {
        "interests": ["football", "F1", "NBA", "Olympic sports", "esports"],
        "sources": [
            "https://www.espn.com/espn/rss/news",
            "http://feeds.bbci.co.uk/sport/rss.xml",
            "https://www.skysports.com/rss/12040",
            "https://theathletic.com/feed/"
        ]
    },
    "Lisa Thompson": {
        "interests": ["movies", "celebrity news", "TV shows", "music", "books"],
        "sources": [
            "https://variety.com/feed/",
            "https://www.rollingstone.com/music/music-news/feed/",
            "https://www.billboard.com/feed/",
            "https://www.hollywoodreporter.com/t/feed/"
        ]
    },
    "David Martinez": {
        "interests": ["space exploration", "AI", "biotech", "physics", "renewable energy"],
        "sources": [
            "https://www.nasa.gov/rss/dyn/breaking_news.rss",
            "https://www.sciencedaily.com/rss/all.xml",
            "https://www.nature.com/subjects/science.rss",
            "https://arstechnica.com/science/feed/"
        ]
    }
}

# --- Article Fetching ---
def fetch_articles(rss_urls, limit=5):
    articles = []
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries[:limit]:
            try:
                art = Article(entry.link)
                art.download()
                art.parse()
                articles.append({
                    "title": art.title,
                    "text": art.text,
                    "url": art.source_url if hasattr(art, 'source_url') else entry.link
                })
            except:
                continue
    return articles

# --- Relevance Scoring ---
def score_articles_by_interests(articles, interests):
    scored_articles = []
    for article in articles:
        content = article["title"] + " " + article["text"]
        tfidf = TfidfVectorizer().fit_transform([content] + interests)
        score = cosine_similarity(tfidf[0:1], tfidf[1:]).mean()
        scored_articles.append((score, article))
    scored_articles.sort(reverse=True, key=lambda x: x[0])
    return [a for score, a in scored_articles[:5]]

# --- Summarizer ---
# def summarize(text, max_len=100):
#     try:
#         return summarizer(text[:1024], max_length=max_len, min_length=30, do_sample=False)[0]['summary_text']
#     except Exception:
#         return text[:200] + "..."

def summarize(text, max_len=100):
    summarizer = load_summarizer()
    try:
        return summarizer(text[:1024], max_length=max_len, min_length=30, do_sample=False)[0]['summary_text']
    except Exception:
        return text[:200] + "..."


# --- Streamlit UI ---
st.set_page_config(page_title="AI Newsletter Generator", layout="centered")
st.title("📰 AI-Powered Personalized Newsletter")
selected_user = st.selectbox("Choose a User Profile", list(user_profiles.keys()))

if st.button("Generate Newsletter"):
    with st.spinner(f"Generating newsletter for {selected_user}..."):
        profile = user_profiles[selected_user]
        articles = fetch_articles(profile["sources"])
        relevant = score_articles_by_interests(articles, profile["interests"])
        
        st.markdown(f"## ✨ {selected_user}'s Newsletter")
        for article in relevant:
            st.markdown(f"### [{article['title']}]({article['url']})")
            st.write(summarize(article['text']))
            st.markdown("---")
        