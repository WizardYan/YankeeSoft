# --- In your "Social Hub" page (after the quick-launch UI) ---
import feedparser
from datetime import datetime
from time import mktime
import streamlit as st

st.divider()
st.subheader("News RSS — subscriptions & keyword filter")

# --- Session defaults ---
if "rss_feeds" not in st.session_state:
    st.session_state.rss_feeds = [
        # sample science/news feeds; edit as you like
        "https://www.nature.com/nature.rss",
        "https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=science",
        "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
        "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    ]
if "rss_keywords" not in st.session_state:
    st.session_state.rss_keywords = ""

# --- Sidebar controls ---
with st.sidebar:
    st.markdown("### RSS Subscriptions")
    new_feed = st.text_input("Add feed URL", key="rss_new_url", placeholder="https://…/rss.xml")
    cols = st.columns([3,1])
    with cols[0]:
        if st.button("➕ Add feed"):
            url = new_feed.strip()
            if url and url not in st.session_state.rss_feeds:
                st.session_state.rss_feeds.append(url)
                st.success("Added.")
    with cols[1]:
        pass

    if st.session_state.rss_feeds:
        rm_feed = st.selectbox("Remove a feed", st.session_state.rss_feeds, key="rss_rm_pick")
        if st.button("🗑 Remove selected"):
            st.session_state.rss_feeds.remove(rm_feed)
            st.success("Removed.")

    st.markdown("### Keyword Filter")
    st.session_state.rss_keywords = st.text_input(
        "Comma-separated keywords (any match)",
        value=st.session_state.rss_keywords,
        key="rss_kw_text",
        placeholder="e.g., AI, neuroscience, grid, opioid"
    )
    max_items = st.slider("Max articles", 20, 300, 100, 10)

@st.cache_data(ttl=900)
def fetch_all(feeds: list[str], max_items: int = 200):
    items = []
    for f in feeds:
        try:
            d = feedparser.parse(f)
            for e in d.entries[: min(len(d.entries), max_items)]:
                # Robust fields
                link = getattr(e, "link", "")
                title = getattr(e, "title", "(no title)")
                summary = getattr(e, "summary", "")
                published = None
                if getattr(e, "published_parsed", None):
                    try:
                        published = datetime.fromtimestamp(mktime(e.published_parsed))
                    except Exception:
                        published = None
                items.append({
                    "feed": d.feed.title if getattr(d, "feed", None) and getattr(d.feed, "title", None) else f,
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "published": published
                })
        except Exception:
            # Skip bad feeds silently
            continue
    # de-dupe by link
    seen = set(); dedup = []
    for it in items:
        if it["link"] and it["link"] not in seen:
            seen.add(it["link"]); dedup.append(it)
    # sort newest first
    dedup.sort(key=lambda x: x["published"] or datetime.min, reverse=True)
    return dedup

feeds = st.session_state.rss_feeds
keywords = [k.strip().lower() for k in st.session_state.rss_keywords.split(",") if k.strip()]

if feeds:
    with st.spinner("Fetching feeds…"):
        articles = fetch_all(feeds, max_items=max_items)
    if keywords:
        def hit(a):
            blob = f"{a['title']} {a['summary']}".lower()
            return any(k in blob for k in keywords)
        articles = [a for a in articles if hit(a)]

    st.caption(f"Showing {len(articles)} articles from {len(feeds)} feed(s).")
    for a in articles:
        with st.container(border=True):
            st.markdown(f"**{a['title']}**  \n{a['feed']}"
                        + (f" · {a['published'].strftime('%Y-%m-%d %H:%M')}" if a['published'] else ""))
            if a["summary"]:
                st.write(a["summary"], unsafe_allow_html=True)
            st.markdown(f"[Open article]({a['link']})")
else:
    st.info("Add at least one RSS feed in the sidebar to get started.")
