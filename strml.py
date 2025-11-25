import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure page settings
st.set_page_config(
    page_title="Audible Insights: Intelligent Book Recommendations",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff6f00;   /* Deep Orange */
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #ff8a65;   /* Light Red-Orange */
        margin: 1.5rem 0;
    }
    .metric-card {
        background: linear-gradient(90deg, #ff7043 0%, #ffab91 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background: #fff5f0;     /* Very light orange/red */
        color: black;    
        border: 1px solid #ffab91;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(255, 87, 34, 0.2);
    }
    .feature-highlight {
        background: linear-gradient(135deg, #ff8a65 0%, #ff7043 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .personalized-card {
        background: linear-gradient(135deg, #ff9e80 0%, #ff6e6e 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .stats-card {
        background: #fff3e0;   /* Light peach background */
        border-left: 4px solid #ff6f00;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# HIDE / SHOW SIDEBAR TOGGLE
# -------------------------------------------------

# Initialize session state
if "show_sidebar" not in st.session_state:
    st.session_state.show_sidebar = False

# Toggle button in main page
if st.button("☰ Menu"):
    st.session_state.show_sidebar = not st.session_state.show_sidebar

# CSS to hide the sidebar completely
if not st.session_state.show_sidebar:
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                display: block;
            }
        </style>
    """, unsafe_allow_html=True)


# Data loading and preprocessing
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the book dataset"""
    try:
        # Try to load from uploaded file or default CSV
        df = pd.read_csv("clustered_books_improved.csv")
        
        # Data cleaning and preprocessing
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        df['Number of Reviews'] = pd.to_numeric(df['Number of Reviews'], errors='coerce').fillna(0)
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
        df['Description'] = df['Description'].fillna('No description available')
        
        # Create combined text for content-based filtering
        df['combined_text'] = (df['Book Name'].fillna('') + ' ' + 
                              df['Author'].fillna('') + ' ' + 
                              df['Genre'].fillna('') + ' ' + 
                              df['Description'].fillna(''))
        
        # Extract additional features
        df['Description_Length'] = df['Description'].str.len()
        df['Title_Length'] = df['Book Name'].str.len()
        df['Review_Category'] = pd.cut(df['Number of Reviews'], 
                                     bins=[0, 10, 100, 1000, float('inf')], 
                                     labels=['Low', 'Medium', 'High', 'Very High'])
        df['Price_Category'] = pd.cut(df['Price'], 
                                    bins=[0, 500, 1500, 3000, float('inf')], 
                                    labels=['Budget', 'Affordable', 'Premium', 'Luxury'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Create sample data for demonstration
        sample_data = {
            'Book Name': ['Sample Book 1', 'Sample Book 2', 'Sample Book 3'],
            'Author': ['Author A', 'Author B', 'Author C'],
            'Genre': ['Fiction', 'Non-Fiction', 'Science'],
            'Rating': [4.5, 4.2, 4.8],
            'Number of Reviews': [100, 250, 75],
            'Price': [500, 750, 1200],
            'Description': ['A great fiction book', 'An informative non-fiction', 'A fascinating science book']
        }
        return pd.DataFrame(sample_data)

# Updated Recommendation Engine Class
class BookRecommendationEngine:
    def __init__(self, df):
        self.df = df
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.clusters = None
        self.scaler = StandardScaler()
        self.genre_encoder = None
        
    def prepare_content_based(self):
        """Prepare TF-IDF matrix for content-based filtering"""
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = tfidf.fit_transform(self.df['combined_text'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
    def content_based_recommendations(self, book_title, n_recommendations=10):
        """Get content-based recommendations"""
        if self.cosine_sim is None:
            self.prepare_content_based()
            
        try:
            idx = self.df[self.df['Book Name'] == book_title].index[0]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:n_recommendations+1]
            
            book_indices = [i[0] for i in sim_scores]
            recommendations = self.df.iloc[book_indices].copy()
            recommendations['Similarity_Score'] = [i[1] for i in sim_scores]
            
            return recommendations
        except IndexError:
            return pd.DataFrame()
    
    def prepare_clustering(self):
        """Prepare clustering model with proper genre encoding"""
        from sklearn.preprocessing import LabelEncoder
        
        # Select numerical features for clustering
        numerical_features = ['Rating', 'Number of Reviews', 'Price', 'Description_Length', 'Title_Length']
        X_numerical = self.df[numerical_features].fillna(0)
        
        # Encode categorical Genre feature
        self.genre_encoder = LabelEncoder()
        genre_encoded = self.genre_encoder.fit_transform(self.df['Genre'].fillna('Unknown'))
        
        # Combine numerical and encoded categorical features
        X = np.column_stack([X_numerical, genre_encoded])
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform K-means clustering
        n_clusters = min(8, len(self.df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(X_scaled)
        self.df['Cluster'] = self.clusters
        
        return X_scaled
    
    def cluster_based_recommendations(self, book_title, n_recommendations=10):
        """Get cluster-based recommendations with genre consideration"""
        if self.clusters is None:
            self.prepare_clustering()
            
        try:
            book_row = self.df[self.df['Book Name'] == book_title]
            if book_row.empty:
                return pd.DataFrame()
                
            book_cluster = book_row['Cluster'].iloc[0]
            book_genre = book_row['Genre'].iloc[0]
            
            # Get books from the same cluster
            cluster_books = self.df[self.df['Cluster'] == book_cluster]
            cluster_books = cluster_books[cluster_books['Book Name'] != book_title]
            
            if cluster_books.empty:
                # Fallback: get books from all clusters
                all_books = self.df[self.df['Book Name'] != book_title]
                recommendations = all_books.sort_values(
                    ['Rating', 'Number of Reviews'], 
                    ascending=[False, False]
                ).head(n_recommendations)
            else:
                # Prioritize books with same genre within the cluster
                same_genre_books = cluster_books[cluster_books['Genre'] == book_genre]
                other_books = cluster_books[cluster_books['Genre'] != book_genre]
                
                # Combine same genre books first, then others
                if not same_genre_books.empty:
                    same_genre_sorted = same_genre_books.sort_values(
                        ['Rating', 'Number of Reviews'], 
                        ascending=[False, False]
                    )
                    other_sorted = other_books.sort_values(
                        ['Rating', 'Number of Reviews'], 
                        ascending=[False, False]
                    )
                    
                    # Take more from same genre if available
                    same_genre_count = min(len(same_genre_sorted), max(1, n_recommendations // 2))
                    other_count = n_recommendations - same_genre_count
                    
                    recommendations = pd.concat([
                        same_genre_sorted.head(same_genre_count),
                        other_sorted.head(other_count)
                    ]).head(n_recommendations)
                else:
                    recommendations = cluster_books.sort_values(
                        ['Rating', 'Number of Reviews'], 
                        ascending=[False, False]
                    ).head(n_recommendations)
            
            # Add similarity scores based on cluster and genre
            recommendations = recommendations.copy()
            recommendations['Genre_Match'] = recommendations['Genre'] == book_genre
            recommendations['Cluster_Score'] = 1.0  # Same cluster
            
            return recommendations
            
        except (IndexError, KeyError) as e:
            print(f"Error in cluster-based recommendations: {e}")
            return pd.DataFrame()
    
    def hybrid_recommendations(self, book_title, n_recommendations=10, content_weight=0.6):
        """Get hybrid recommendations combining content-based and clustering"""
        content_recs = self.content_based_recommendations(book_title, n_recommendations)
        cluster_recs = self.cluster_based_recommendations(book_title, n_recommendations)
        
        if content_recs.empty and cluster_recs.empty:
            return pd.DataFrame()
        
        # Combine recommendations with weights
        all_recs = []
        
        if not content_recs.empty:
            content_recs_copy = content_recs.copy()
            content_recs_copy['Content_Score'] = content_recs_copy.get('Similarity_Score', 0) * content_weight
            all_recs.append(content_recs_copy)
        
        if not cluster_recs.empty:
            cluster_recs_copy = cluster_recs.copy()
            cluster_recs_copy['Cluster_Score'] = (1 - content_weight)
            # Add bonus for genre match
            cluster_recs_copy['Genre_Bonus'] = cluster_recs_copy.get('Genre_Match', False).astype(float) * 0.1
            all_recs.append(cluster_recs_copy)
        
        if not all_recs:
            return pd.DataFrame()
        
        # Merge and calculate final scores
        combined_recs = pd.concat(all_recs).drop_duplicates(subset=['Book Name'])
        
        # Calculate final scores
        combined_recs['Final_Score'] = (
            combined_recs.get('Content_Score', 0) + 
            combined_recs.get('Cluster_Score', 0) + 
            combined_recs.get('Genre_Bonus', 0) +
            combined_recs['Rating'] * 0.1
        )
        
        return combined_recs.sort_values('Final_Score', ascending=False).head(n_recommendations)
    
    def get_cluster_analysis(self):
        """Get analysis of clusters formed"""
        if self.clusters is None:
            self.prepare_clustering()
        
        cluster_analysis = []
        for cluster_id in range(len(np.unique(self.clusters))):
            cluster_books = self.df[self.df['Cluster'] == cluster_id]
            
            analysis = {
                'Cluster_ID': cluster_id,
                'Number_of_Books': len(cluster_books),
                'Average_Rating': cluster_books['Rating'].mean(),
                'Average_Price': cluster_books['Price'].mean(),
                'Average_Reviews': cluster_books['Number of Reviews'].mean(),
                'Top_Genres': cluster_books['Genre'].value_counts().head(3).to_dict(),
                'Price_Range': f"₹{cluster_books['Price'].min():.0f} - ₹{cluster_books['Price'].max():.0f}"
            }
            cluster_analysis.append(analysis)
        
        return pd.DataFrame(cluster_analysis)

# Load data
df = load_and_preprocess_data()
if df is not None:
    recommendation_engine = BookRecommendationEngine(df)

# Sidebar navigation
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem; 
    background: linear-gradient(90deg, #ff9a76 0%, #ff6f3c 100%); 
    border-radius: 10px; margin-bottom: 1rem;'>
    <h2 style='color: white; margin: 0;'>📚 Book Recommender</h2>
    <p style='color: white; margin: 0;'>AI-Powered Recommendation System</p>
</div>
""", unsafe_allow_html=True)


page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview",
     "📊 Visual Insights",
     "📚 Book Recommendation System",
     "👤 Personalized Recommendations"]
)


# PAGE 1: Dataset Overview
if page == "🏠 Overview":
    st.markdown("<h1 class='main-header'>📚 Audible Insights: Intelligent Book Recommendations</h1>", unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <h2 style='color:#d35400;'>📘 Project Overview</h2>

    <p>
    This Book Recommendation System is designed to help readers discover books they are likely to enjoy.
    It analyzes book features such as genres, authors, ratings, and user preferences to generate meaningful recommendations.
    </p>

    <p>
    The system uses multiple machine learning techniques, including:
    </p>

    <ul>
    <li><b>Content-Based Filtering</b> – recommends books similar to what a user already likes.</li>
    <li><b>Clustering</b> – groups books based on similarities in genre, description, and metadata.</li>
    <li><b>Simple NLP Processing</b> – extracts keywords from book titles and descriptions to improve matching.</li>
    </ul>

    <p>
    Users can explore the dataset, analyze trends through visualizations, and receive personalized suggestions.
    The app is built using Streamlit and can be deployed on cloud platforms like AWS for easy access.
    </p>
    """, unsafe_allow_html=True)


    
    if df is not None:
        # Dataset statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{len(df):,}</h3>
                <p>Total Books</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{df['Author'].nunique():,}</h3>
                <p>Unique Authors</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{df['Genre'].nunique():,}</h3>
                <p>Genres</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{df['Rating'].mean():.2f}</h3>
                <p>Avg Rating</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Dataset information
        st.markdown("<h2 class='sub-header'>📋 Dataset Overview</h2>", unsafe_allow_html=True)
        feature_info = {
            'Feature': ['Book Name', 'Author', 'Genre', 'Rating', 'Number of Reviews', 'Price', 'Description'],
            'Description': [
                'Title of the book',
                'Book author(s)',
                'Genre/Category',
                'Average user rating (1-5)',
                'Total number of reviews',
                'Price in local currency',
                'Book description/summary'
            ],
            'Data Type': ['Text', 'Text', 'Categorical', 'Numerical', 'Numerical', 'Numerical', 'Text']
        }
        st.dataframe(pd.DataFrame(feature_info), use_container_width=True)

        # Sample data
        st.markdown("<h2 class='sub-header'>📖 Sample Books</h2>", unsafe_allow_html=True)
        sample_books = df.sample(n=min(5, len(df)))
    
        for idx, book in sample_books.iterrows():
            with st.expander(f"📚 {book['Book Name']} by {book['Author']}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**Genre:** {book['Genre']}")
                    st.write(f"**Description:** {book['Description'][:300]}...")
                with col2:
                    st.metric("Rating", f"{book['Rating']:.1f}/5.0")
                    st.metric("Reviews", f"{int(book['Number of Reviews']):,}")
                    st.metric("Price", f"₹{int(book['Price']):,}")
# PAGE 2: Visual Insights
elif page == "📊 Visual Insights":
    st.markdown("<h1 class='main-header'>📊 Visual Insights</h1>", unsafe_allow_html=True)
    
    if df is not None:
        # ------------------------------
        # Distribution Analysis (Dropdown)
        # ------------------------------

        st.markdown("<h2 class='sub-header'>📈 Distribution Analysis</h2>", unsafe_allow_html=True)

        option = st.selectbox(
            "Select a column to view its distribution:",
            ["Rating", "Price", "Number of Reviews", "Genre"]
        )

        # Plot based on selection
        if option == "Rating":
            fig = px.histogram(
                df, x='Rating', nbins=20,
                title='Distribution of Book Ratings'
        )
            st.plotly_chart(fig, use_container_width=True)

        elif option == "Price":
            fig = px.box(
                df, y='Price',
                title='Price Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        elif option == "Number of Reviews":
            fig_reviews = px.histogram(df, y='Number of Reviews', title='Distribution of Number of Reviews',
                                     color_discrete_sequence=['#2ca02c'])
            fig_reviews.update_xaxes(type="log")
            st.plotly_chart(fig_reviews, use_container_width=True)
        elif option == "Genre":
            genre_counts = df['Genre'].value_counts().head(10)
            fig = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation='h',
                title='Top 10 Genres'
            )
            st.plotly_chart(fig, use_container_width=True)

        
        # Correlation Analysis
        st.markdown("<h2 class='sub-header'>🔗 Correlation Analysis</h2>", unsafe_allow_html=True)
        
        # Correlation matrix
        numeric_cols = ['Rating', 'Number of Reviews', 'Price', 'Description_Length', 'Title_Length']
        corr_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(corr_matrix, 
                           title='Feature Correlation Matrix',
                           color_continuous_scale='RdBu',
                           aspect='auto')
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, width='stretch')
        
        # Advanced visualizations
        st.markdown("<h2 class='sub-header'>🎯 Advanced Analysis</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating vs Reviews
            # Define bins based on your requirement
            bins = [0, 100, 1000, 5000, df['Number of Reviews'].max()]
            labels = ['Low (0–100)', 'Medium (100–1k)', 'High (1k–5k)', 'Very High (5k+)']

            # Create the category column
            df['Review_Level'] = pd.cut(df['Number of Reviews'], bins=bins, labels=labels, include_lowest=True)

            # Box Plot: Rating distribution across review levels
            fig_box = px.box(
                               df,
                               x='Review_Level',
                               y='Rating',
                               title='Rating Distribution Across Number of Reviews',
                               #points='all'   # shows all data points
           )

            st.plotly_chart(fig_box, use_container_width=True)

        with col2:
            # Price vs Rating
            fig_price_rating = px.scatter(
                                            df,
                                            x='Price',
                                            y='Rating',
                                            title='Price vs Rating',
                                            hover_data=['Book Name'],   
                                            opacity=0.7
                )
            st.plotly_chart(fig_price_rating, use_container_width=True)
        # Statistical insights
        st.markdown("<h2 class='sub-header'>💡 Key Insights</h2>", unsafe_allow_html=True)
        
        insights_col1, insights_col2, insights_col3 = st.columns(3)
        
        with insights_col1:
            st.info(f"""
            **Rating Insights**
            - Average rating: {df['Rating'].mean():.2f}
            - Most books rated above 4.0
            - Rating std: {df['Rating'].std():.2f}
            """)
        
        with insights_col2:
            most_reviewed_genre = df.groupby('Genre')['Number of Reviews'].sum().idxmax()
            st.success(f"""
            **Popular Genres**
            - Most reviewed: {most_reviewed_genre}
            - Total genres: {df['Genre'].nunique()}
            - Avg books per genre: {df.groupby('Genre').size().mean():.0f}
            """)
        
        with insights_col3:
            high_rated_books = len(df[df['Rating'] >= 4.5])
            st.warning(f"""
            **Quality Metrics**
            - High-rated books (≥4.5): {high_rated_books}
            - Price range: ₹{df['Price'].min():.0f} - ₹{df['Price'].max():.0f}
            - Avg description length: {df['Description_Length'].mean():.0f} chars
            """)

# PAGE 3: Recommendation Engine
elif page == "📚 Book Recommendation System":
    st.markdown("<h1 class='main-header'>📚 Book Recommendation System</h1>", unsafe_allow_html=True)
    
    if df is not None:
        # Algorithm selection
        st.markdown("<h2 class='sub-header'>🔧 Choose Recommendation Algorithm</h2>", unsafe_allow_html=True)
        
        algorithm = st.selectbox(
            "Select recommendation approach:",
            ["Content-Based Filtering", "Clustering-Based Recommendations", "Hybrid Approach"]
        )
        
        # Book selection
        book_titles = sorted(df['Book Name'].tolist())
        selected_book = st.selectbox("Choose a book you like:", book_titles)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            n_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
        
        with col2:
            if algorithm == "Hybrid Approach":
                content_weight = st.slider("Content-based weight:", 0.0, 1.0, 0.6, 0.1)
        
        if st.button("🚀 Get Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                # Show selected book info
                selected_book_info = df[df['Book Name'] == selected_book].iloc[0]
                
                st.markdown("<h3>📖 Selected Book</h3>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='recommendation-card'>
                    <h4 style='color:#1a1a1a;'>📖 {selected_book_info['Book Name']}</h4>
                    <p><strong>Author:</strong> ✍️ {selected_book_info['Author']}</p>
                    <p><strong>Genre:</strong> 🏷️ {selected_book_info['Genre']}</p>
                    <p><strong>Rating:</strong> ⭐ {selected_book_info['Rating']}/5.0 ({int(selected_book_info['Number of Reviews'])} reviews)</p>
                    <p><strong>Price:</strong> 💰 {selected_book_info['Price']}</p>
                    <p><strong>Description:</strong> {selected_book_info['Description'][:300]}...</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate recommendations
                if algorithm == "Content-Based Filtering":
                    recommendations = recommendation_engine.content_based_recommendations(
                        selected_book, n_recommendations
                    )
                elif algorithm == "Clustering-Based Recommendations":
                    recommendations = recommendation_engine.cluster_based_recommendations(
                        selected_book, n_recommendations
                    )
                else:  # Hybrid Approach
                    recommendations = recommendation_engine.hybrid_recommendations(
                        selected_book, n_recommendations, content_weight
                    )
                
                if not recommendations.empty:
                    st.markdown(f"<h3>🎯 {algorithm} Recommendations</h3>", unsafe_allow_html=True)
                    
                    for idx, (_, book) in enumerate(recommendations.iterrows(), 1):
                        score_info = ""
                        if 'Similarity_Score' in book:
                            score_info = f" (Similarity: {book['Similarity_Score']:.3f})"
                        elif 'Final_Score' in book:
                            score_info = f" (Score: {book['Final_Score']:.3f})"
                        
                        st.markdown(f"""
                        <div class='recommendation-card'>
                            <h4 style='color:#1a1a1a;'>{idx}. 📖 {book['Book Name']}{score_info}</h4>
                            <p><strong>Author:</strong> ✍️ {book['Author']}</p>
                            <p><strong>Genre:</strong> 🏷️ {book['Genre']}</p>
                            <p><strong>Rating:</strong> ⭐ {book['Rating']}/5.0 | 
                               <strong>Reviews:</strong> {int(book['Number of Reviews']):,} | 
                               <strong>Price:</strong> 💰 {int(book['Price']):,}</p>
                            <p><strong>Description:</strong> {book['Description'][:250]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No recommendations found. Please try a different book or algorithm.")
        
        # Algorithm explanations
        st.markdown("<h2 class='sub-header'>📚 Algorithm Explanations</h2>", unsafe_allow_html=True)
        
        with st.expander("🔍 Content-Based Filtering"):
            st.markdown("""
            **How it works:**
            - Analyzes book features like title, author, genre, and description
            - Uses TF-IDF vectorization to convert text into numerical features
            - Calculates cosine similarity between books
            - Recommends books most similar to your selected book
            
            **Best for:** Finding books with similar themes, writing styles, or topics
            """)
        
        with st.expander("🎯 Clustering-Based Recommendations"):
            st.markdown("""
            **How it works:**
            - Groups books into clusters based on numerical features (rating, reviews, price, etc.)
            - Uses K-means clustering algorithm
            - Recommends books from the same cluster as your selected book
            
            **Best for:** Finding books with similar popularity and quality metrics
            """)
        
        with st.expander("🔄 Hybrid Approach"):
            st.markdown("""
            **How it works:**
            - Combines content-based and clustering approaches
            - Weights recommendations from both methods
            - Provides more diverse and accurate recommendations
            
            **Best for:** Getting the best of both worlds with balanced recommendations
            """)

# PAGE 4: Personalized Recommendations
elif page == "👤 Personalized Recommendations":
    st.markdown("<h1 class='main-header'>👤 Personalized Recommendations</h1>", unsafe_allow_html=True)
    
    if df is not None:
        st.markdown("<h2 class='sub-header'>🎯 Tell Us About Your Preferences</h2>", unsafe_allow_html=True)
        
        # User preference inputs
        col1, col2 = st.columns(2)
        
        with col1:
            # Genre preferences
            st.subheader("📚 Favorite Genres")
            all_genres = sorted(df['Genre'].dropna().unique())
            selected_genres = st.multiselect(
                "Select your favorite genres (you can choose multiple):",
                all_genres,
                default=all_genres[:3] if len(all_genres) >= 3 else all_genres
            )
            
            # Reading goals
            st.subheader("🎯 Reading Goals")
            reading_goal = st.selectbox(
                "What's your primary reading goal?",
                [
                    "Entertainment and relaxation",
                    "Learning and self-improvement",
                    "Professional development",
                    "Academic research",
                    "Exploring new topics"
                ]
            )
            
            # Budget preference
            st.subheader("💰 Budget Range")
            budget_range = st.select_slider(
                "Select your budget range:",
                options=["Free", "Budget (₹0-500)", "Standard (₹500-1500)", "Premium (₹1500+)"],
                value="Standard (₹500-1500)"
            )
        
        with col2:
            # Quality preferences
            st.subheader("⭐ Quality Preferences")
            min_rating = st.slider("Minimum rating:", 1.0, 5.0, 4.0, 0.1)
            
            popularity_pref = st.radio(
                "Book popularity preference:",
                ["Hidden gems (fewer reviews)", "Balanced", "Popular books (many reviews)"]
            )
            
            # Book characteristics
            st.subheader("📖 Book Characteristics")
            preferred_length = st.selectbox(
                "Preferred description length:",
                ["Any length", "Detailed descriptions", "Concise descriptions"]
            )
            
            # Number of recommendations
            num_recommendations = st.slider("Number of recommendations:", 5, 30, 15)
        
        if st.button("🎯 Get My Personalized Recommendations", type="primary"):
            with st.spinner("Analyzing your preferences and generating recommendations..."):
                # Filter books based on user preferences
                filtered_df = df.copy()
                
                # Apply genre filter
                if selected_genres:
                    filtered_df = filtered_df[filtered_df['Genre'].isin(selected_genres)]
                
                # Apply rating filter
                filtered_df = filtered_df[filtered_df['Rating'] >= min_rating]
                
                # Apply budget filter
                if budget_range == "Free":
                    filtered_df = filtered_df[filtered_df['Price'] == 0]
                elif budget_range == "Budget (₹0-500)":
                    filtered_df = filtered_df[filtered_df['Price'] <= 500]
                elif budget_range == "Standard (₹500-1500)":
                    filtered_df = filtered_df[(filtered_df['Price'] > 500) & (filtered_df['Price'] <= 1500)]
                else:  # Premium
                    filtered_df = filtered_df[filtered_df['Price'] > 1500]
                
                # Apply popularity filter
                if popularity_pref == "Hidden gems (fewer reviews)":
                    filtered_df = filtered_df[filtered_df['Number of Reviews'] <= filtered_df['Number of Reviews'].quantile(0.3)]
                elif popularity_pref == "Popular books (many reviews)":
                    filtered_df = filtered_df[filtered_df['Number of Reviews'] >= filtered_df['Number of Reviews'].quantile(0.7)]
                
                # Apply description length filter
                if preferred_length == "Detailed descriptions":
                    filtered_df = filtered_df[filtered_df['Description_Length'] >= filtered_df['Description_Length'].quantile(0.7)]
                elif preferred_length == "Concise descriptions":
                    filtered_df = filtered_df[filtered_df['Description_Length'] <= filtered_df['Description_Length'].quantile(0.3)]
                
                if not filtered_df.empty:
                    # Smart ranking based on user goals
                    if reading_goal == "Entertainment and relaxation":
                        # Prioritize fiction, high ratings
                        filtered_df['goal_score'] = np.where(
                            filtered_df['Genre'].str.contains('Fiction|Story|Novel', case=False, na=False),
                            filtered_df['Rating'] * 1.2, filtered_df['Rating']
                        )
                    elif reading_goal == "Learning and self-improvement":
                        # Prioritize self-help, educational content
                        filtered_df['goal_score'] = np.where(
                            filtered_df['Genre'].str.contains('Self|Improvement|Education|Learning', case=False, na=False),
                            filtered_df['Rating'] * 1.2, filtered_df['Rating']
                        )
                    elif reading_goal == "Professional development":
                        # Prioritize business, career-related books
                        filtered_df['goal_score'] = np.where(
                            filtered_df['Genre'].str.contains('Business|Career|Professional|Management', case=False, na=False),
                            filtered_df['Rating'] * 1.2, filtered_df['Rating']
                        )
                    else:
                        filtered_df['goal_score'] = filtered_df['Rating']
                    
                    # Final ranking
                    filtered_df['final_score'] = (
                        filtered_df['goal_score'] * 0.4 +
                        (filtered_df['Number of Reviews'] / filtered_df['Number of Reviews'].max()) * 0.3 +
                        (1 - filtered_df['Price'] / filtered_df['Price'].max()) * 0.3
                    )
                    
                    recommendations = filtered_df.nlargest(num_recommendations, 'final_score')
                    
                    # Display results
                    st.success(f"Found {len(recommendations)} personalized recommendations for you!")
                    
                    # Summary of filters applied
                    st.markdown("<h3>📋 Your Preference Summary</h3>", unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.info(f"**Genres:** {', '.join(selected_genres[:3])}")
                    with col2:
                        st.info(f"**Budget:** {budget_range}")
                    with col3:
                        st.info(f"**Min Rating:** {min_rating}⭐")
                    with col4:
                        st.info(f"**Goal:** {reading_goal}")
                    
                    st.markdown("<h3>🎯 Your Personalized Recommendations</h3>", unsafe_allow_html=True)
                    
                    # Display recommendations with enhanced information
                    for idx, (_, book) in enumerate(recommendations.iterrows(), 1):
                        match_reasons = []
                        
                        # Determine why this book matches user preferences
                        if book['Genre'] in selected_genres:
                            match_reasons.append(f"Matches your favorite genre: {book['Genre']}")
                        if book['Rating'] >= min_rating + 0.5:
                            match_reasons.append(f"High rating: {book['Rating']}/5.0")
                        if popularity_pref == "Hidden gems (fewer reviews)" and book['Number of Reviews'] <= 100:
                            match_reasons.append("Hidden gem with fewer reviews")
                        elif popularity_pref == "Popular books (many reviews)" and book['Number of Reviews'] >= 1000:
                            match_reasons.append("Popular book with many reviews")
                        
                        reasons_text = " | ".join(match_reasons) if match_reasons else "Good overall match"
                        
                        st.markdown(f"""
                        <div class='recommendation-card'>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <h4 style='color:#1a1a1a;'>{idx}.📖 {book['Book Name']}</h4>
                                <span style='background: #28a745; color: white; padding: 0.2rem 0.5rem; border-radius: 15px; font-size: 0.8rem;'>
                                    Match Score: {book['final_score']:.2f}
                                </span>
                            </div>
                            <p><strong>Author:</strong> ✍️ {book['Author']} | <strong>Genre:</strong> 🏷️ {book['Genre']}</p>
                            <p><strong>Rating:</strong> ⭐ {book['Rating']}/5.0 | 
                               <strong>Reviews:</strong> {int(book['Number of Reviews']):,} | 
                               <strong>Price:</strong> 💰 {int(book['Price']):,}</p>
                            <p style='color: #28a745; font-weight: bold;'>💡 Why recommended: {reasons_text}</p>
                            <p><strong>Description:</strong> {book['Description'][:300]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Additional insights
                    st.markdown("<h3>📊 Recommendation Insights</h3>", unsafe_allow_html=True)
                    
                    insights_col1, insights_col2, insights_col3 = st.columns(3)
                    
                    with insights_col1:
                        avg_rating = recommendations['Rating'].mean()
                        st.metric("Average Rating", f"{avg_rating:.2f}⭐", f"+{avg_rating-df['Rating'].mean():.2f}")
                    
                    with insights_col2:
                        avg_price = recommendations['Price'].mean()
                        st.metric("Average Price", f"₹{avg_price:.0f}", f"₹{avg_price-df['Price'].mean():.0f}")
                    
                    with insights_col3:
                        genre_diversity = recommendations['Genre'].nunique()
                        st.metric("Genre Diversity", f"{genre_diversity} genres", f"{genre_diversity}/{len(selected_genres)}")
                    
                    # Genre distribution of recommendations
                    if len(recommendations) > 0:
                        st.subheader("📚 Genre Distribution in Your Recommendations")
                        genre_dist = recommendations['Genre'].value_counts()
                        fig_genre_dist = px.pie(values=genre_dist.values, names=genre_dist.index, 
                                              title="Recommended Books by Genre")
                        st.plotly_chart(fig_genre_dist, width='stretch')
                    
                    # Export functionality
                    st.markdown("<h3>📥 Export Your Recommendations</h3>", unsafe_allow_html=True)
                    
                    export_df = recommendations[['Book Name', 'Author', 'Genre', 'Rating', 'Number of Reviews', 'Price', 'Description']].copy()
                    export_df['Match_Score'] = recommendations['final_score']
                    
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name="my_book_recommendations.csv",
                        mime="text/csv",
                        type="secondary"
                    )
                    
                else:
                    st.warning("😔 No books found matching your criteria. Try adjusting your preferences!")
                    
                    # Suggestions for broadening search
                    st.markdown("<h3>💡 Suggestions to find more books:</h3>", unsafe_allow_html=True)
                    st.write("• Try selecting more genres")
                    st.write("• Lower the minimum rating requirement")
                    st.write("• Increase your budget range")
                    st.write("• Choose 'Any length' for book descriptions")
        
        # Reading recommendations based on data insights
        st.markdown("<h2 class='sub-header'>📚 Reading Recommendations Based on Data</h2>", unsafe_allow_html=True)
        
        if df is not None:
            # Trending books (high rating, recent reviews)
            trending_books = df[
                (df['Rating'] >= 4.3) & 
                (df['Number of Reviews'] >= 50) & 
                (df['Number of Reviews'] <= 500)
            ].sort_values(['Rating', 'Number of Reviews'], ascending=[False, False]).head(5)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔥 Trending Books")
                for _, book in trending_books.iterrows():
                    with st.expander(f"📖 {book['Book Name']}"):
                        st.write(f"**Author:** {book['Author']}")
                        st.write(f"**Genre:** {book['Genre']}")
                        st.write(f"**Rating:** {book['Rating']}⭐ ({int(book['Number of Reviews'])} reviews)")
                        st.write(f"**Price:** ₹{int(book['Price'])}")
            
            with col2:
                # Budget-friendly high-quality books
                budget_books = df[
                    (df['Rating'] >= 4.2) & 
                    (df['Price'] <= 500) & 
                    (df['Price'] > 0)
                ].sort_values('Rating', ascending=False).head(5)
                
                st.subheader("💰 Budget-Friendly Gems")
                for _, book in budget_books.iterrows():
                    with st.expander(f"📖 {book['Book Name']}"):
                        st.write(f"**Author:** {book['Author']}")
                        st.write(f"**Genre:** {book['Genre']}")
                        st.write(f"**Rating:** {book['Rating']}⭐")
                        st.write(f"**Price:** ₹{int(book['Price'])}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>📚 Book Recommendation System | Built with Streamlit & Machine Learning</p>
    <p>🔬 Data Science Project | ML & DL Algorithms</p>
</div>
""", unsafe_allow_html=True)

# Session state for user preferences
if 'user_session' not in st.session_state:
    st.session_state.user_session = {
        'recommendations_generated': 0,
        'favorite_genres': [],
        'last_recommendations': []
    }

# Update session state
if page == "👤 Personalized Recommendations":
    st.session_state.user_session['recommendations_generated'] += 1