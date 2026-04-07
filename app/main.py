import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Eco-Style Recommender", page_icon="🌿", layout="wide")

# --- FILE PATHING ---
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "..", "data", "handm_scored.csv")

@st.cache_data
def load_data():
    return pd.read_csv(file_path)

df = load_data()

# --- AI ENGINE ---
@st.cache_resource 
def build_engine(data):
    # We return both the matrix AND the vectorizer so we can transform new search queries
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['productName'] + " " + data['details'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, tfidf_matrix, cosine_sim

# Unpack the engine components
tfidf, tfidf_matrix, cosine_sim = build_engine(df)

# --- SIDEBAR FILTERS ---
st.sidebar.header("🌿 Search Settings")
min_score = st.sidebar.slider("Min Sustainability Score", 0, 100, 70)
num_recs = st.sidebar.number_input("Number of recommendations", 1, 10, 3)

# --- HEADER ---
st.title("🌿 Eco-Style Recommender")
st.markdown("Find sustainable alternatives to your favorite fashion pieces.")

# --- SEARCH INTERFACE ---
col_search, col_drop = st.columns(2)

with col_search:
    search_query = st.text_input("🔍 Search by Style", placeholder="e.g. Blue Denim Jacket")

with col_drop:
    product_list = ["None"] + list(df['productName'].unique())
    selected_dropdown = st.selectbox("🎯 Or Pick from List", product_list)

# Determine the Target Index
idx = None
# 1. If the user picked something from the dropdown, use that
if selected_dropdown != "":
    idx = df[df['productName'] == selected_dropdown].index[0]
# 2. If the dropdown is empty but there is text in the search box, use the search
elif search_query:
    query_vec = tfidf.transform([search_query.lower()])
    search_sim = cosine_similarity(query_vec, tfidf_matrix)
    idx = search_sim.argmax()


if search_query:
    # Inference: Transform the text search into a vector
    query_vec = tfidf.transform([search_query.lower()])
    search_sim = cosine_similarity(query_vec, tfidf_matrix)
    idx = search_sim.argmax()
elif selected_dropdown != "None":
    idx = df[df['productName'] == selected_dropdown].index[0]

# --- DISPLAY RESULTS ---
if idx is not None:
    target = df.iloc[idx]
    
    st.divider()
    
    # 1. Selected Item & Explainable AI
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"Current Style: {target['productName']}")
        st.caption(f"Category: {target['mainCatCode']}")
        
        with st.expander("✨ Why this score?"):
            score = target['sustainability_score']
            if score >= 80:
                st.success("🌟 This item is an **Eco-Champion**.")
            elif score >= 50:
                st.warning("⚖️ This item is a **Mixed Bag**.")
            else:
                st.error("⚠️ This item is **Conventional**.")
            st.write(f"**Composition:** {target['materials']}")
            
    with col2:
        st.metric("Sustainability Score", f"{target['sustainability_score']}/100")

    st.divider()
    
    # 2. Recommendation Logic
    st.subheader("🌿 Highly Similar Sustainable Alternatives")
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    recs = []
    seen_names = {target['productName']}
    
    for i, score in sim_scores[1:]:
        row = df.iloc[i]
        # Only keep if it hits the sustainability threshold
        if row['sustainability_score'] >= min_score and row['productName'] not in seen_names:
            recs.append(row)
            seen_names.add(row['productName'])
        
        # Collect a few extra so we have a pool to sort by price
        if len(recs) >= (num_recs + 2): 
            break
             
    # --- NEW: SORT BY PRICE ---
    # We check for 'white_price' because that's the column in our H&M data
    if 'white_price' in df.columns:
        recs = sorted(recs, key=lambda x: x['white_price'])
    
    # Trim back down to the user's requested number
    recs = recs[:num_recs]

    # 3. Display Recommendations 
    if recs:
        rec_cols = st.columns(len(recs))
        for i, item in enumerate(recs):
            with rec_cols[i]:
                st.info(f"**{item['productName']}**")
                st.progress(int(item['sustainability_score']))
                # Show the price so the user sees the "Value Nudge"
                st.write(f"Price: **${item['white_price']}** | Score: **{item['sustainability_score']}**")