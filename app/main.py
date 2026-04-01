import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Eco-Style Recommender", page_icon="🌿")

# Get the directory that main.py is in
base_path = os.path.dirname(__file__)
# Go up one level (to the root) and then into data/
file_path = os.path.join(base_path, "..", "data", "handm_scored.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(file_path)
    return df
df = load_data()

# --- AI ENGINE ---
@st.cache_resource 
def build_engine(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['productName'] + " " + data['details'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = build_engine(df)

# --- UI LAYOUT ---
st.title("🌿 Eco-Style Recommender")
st.markdown("Find sustainable alternatives to your favorite fashion pieces.")

# Search Bar
product_list = df['productName'].unique()
selected_product = st.selectbox("Type a product name to start:", product_list)
st.sidebar.header("Filter Settings")
min_score = st.sidebar.slider("Minimum Sustainability Score", 0, 100, 70)

if selected_product:
    # Find the index of the selected item
    idx = df[df['productName'] == selected_product].index[0]
    target = df.iloc[idx]
    
    st.divider()
    
    # Show Selected Item
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Selected Item")
        st.write(f"**{target['productName']}**")
        st.write(f"Category: {target['mainCatCode']}")
    with col2:
        st.metric("Sustainability Score", f"{target['sustainability_score']}/100")

    st.divider()
    
    # RECOMMENDATION LOGIC 
    st.subheader("🌿 Sustainable Alternatives")
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    recs = []
    seen_names = {selected_product}
    
    for i, score in sim_scores[1:]:
        row = df.iloc[i]
        if row['sustainability_score'] > 70 and row['productName'] not in seen_names:
            recs.append(row)
            seen_names.add(row['productName'])
        if len(recs) >= 3: break
            
    
    # Display Recommendations in Columns
    rec_cols = st.columns(3)
    for i, item in enumerate(recs):
        with rec_cols[i]:
            st.info(f"**{item['productName']}**")
            # This adds a visual progress bar!
            st.progress(int(item['sustainability_score']))
            st.write(f"Score: **{item['sustainability_score']}/100**")
            st.caption(f"Category: {item['mainCatCode']}")