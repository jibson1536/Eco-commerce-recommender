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
    tfidf = TfidfVectorizer(stop_words='english')
    # Use fillna('') to prevent errors if details are missing
    combined_text = data['productName'] + " " + data['details'].fillna('')
    tfidf_matrix = tfidf.fit_transform(combined_text)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, tfidf_matrix, cosine_sim

tfidf, tfidf_matrix, cosine_sim = build_engine(df)

# --- SIDEBAR FILTERS ---
st.sidebar.header("🌿 Search Settings")
min_score = st.sidebar.slider("Min Sustainability Score", 0, 100, 70)
num_recs = st.sidebar.number_input("Number of recommendations", 1, 10, 3)

# --- HEADER ---
st.title("🌿 Eco-Style Recommender")
st.markdown("Find sustainable alternatives to your favorite fashion pieces.")

# --- SIMULTANEOUS SEARCH INTERFACE ---
col_search, col_drop = st.columns(2)

with col_search:
    search_query = st.text_input("🔍 Search by Style", placeholder="e.g. Blue Denim Jacket", key="text_input")

with col_drop:
    product_options = ["Select a product..."] + sorted(list(df['productName'].unique()))
    selected_dropdown = st.selectbox("🎯 Or Pick from List", product_options, key="dropdown_input")

# Logic to determine which input to follow
idx = None

# If user picks from dropdown, it takes priority
if selected_dropdown != "Select a product...":
    match = df[df['productName'] == selected_dropdown]
    if not match.empty:
        idx = match.index[0]
# Otherwise, use the text search
elif search_query:
    query_vec = tfidf.transform([search_query.lower()])
    search_sim = cosine_similarity(query_vec, tfidf_matrix)
    idx = search_sim.argmax()

# Welcome message if nothing is selected
if idx is None:
    st.info("Start by typing a style above or picking an item from the list!")
    st.stop() 

# --- DISPLAY RESULTS ---
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
        st.write(f"**Composition:** {target.get('materials', 'No material data available')}")
        
with col2:
    st.metric("Sustainability Score", f"{target['sustainability_score']}/100")

st.divider()

# 2. Recommendation Logic
st.subheader("🌿 Highly Similar Sustainable Alternatives")

# Get similarities for the selected index
sim_scores = list(enumerate(cosine_sim[idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

recs = []
seen_names = {target['productName']}

for i, score in sim_scores[1:]:
    row = df.iloc[i]
    if row['sustainability_score'] >= min_score and row['productName'] not in seen_names:
        recs.append(row)
        seen_names.add(row['productName'])
    
    # Get a few extra candidates to allow for price sorting
    if len(recs) >= (num_recs + 5): 
        break
            
# Sort by price 
price_col = 'white_price' if 'white_price' in df.columns else 'price'
if price_col in df.columns:
    recs = sorted(recs, key=lambda x: x[price_col])

recs = recs[:num_recs]

# 3. Display Recommendations 
if recs:
    rec_cols = st.columns(len(recs))
    for i, item in enumerate(recs):
        with rec_cols[i]:
            st.info(f"**{item['productName']}**")
            st.progress(int(item['sustainability_score']))
            
            # Safe price retrieval
            curr_price = item.get(price_col, 'N/A')
            st.write(f"Price: **${curr_price}** | Score: **{item['sustainability_score']}**")
            st.caption(f"Category: {item['mainCatCode']}")
else:
    st.warning("No sustainable alternatives found. Try lowering the 'Min Sustainability Score'!")