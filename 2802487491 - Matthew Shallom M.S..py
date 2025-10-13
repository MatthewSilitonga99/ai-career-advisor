import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="AI Career Advisor",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .info-box {
        padding: 0.5rem;
        border-radius: 0.3rem;
        background-color: #e8f4fd;
        border: 1px solid #b8daff;
    }
</style>
""", unsafe_allow_html=True)

# Load AI Model
@st.cache_resource
def load_model():
    try:
        with open('tfidf_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        st.error(f"Error loading AI model: {e}")
        return None

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main application
def main():
    # Header
    st.markdown('<div class="main-header"> AI Career Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Tanya apapun tentang karir, interview, CV, dan skill development!</div>', unsafe_allow_html=True)
    
    # Load model
    model_data = load_model()
    if model_data is None:
        st.stop()
    
    vectorizer = model_data['vectorizer']
    X = model_data['X']
    df = model_data['df']
    
    # Chat input
    user_question = st.text_input(
        "*Apa pertanyaan karir Anda?*",
        placeholder="Contoh: Bagaimana cara buat CV yang menarik?",
        key="user_input"
    )
    
    # Process question
    if user_question:
        with st.spinner("AI sedang mencari jawaban terbaik..."):
            # AI Processing
            user_vec = vectorizer.transform([user_question.lower()])
            similarities = cosine_similarity(user_vec, X)
            best_match_idx = np.argmax(similarities)
            confidence = similarities[0][best_match_idx]
            
            if confidence > 0.3:
                answer = df.iloc[best_match_idx]['answer']
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': user_question,
                    'answer': answer,
                    'confidence': confidence
                })
                
                # Display answer
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success(f"AI Career Advisor:\n\n{answer}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Confidence score
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.info(f"*Confidence Score:* {confidence:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Saya belum cukup paham pertanyaan ini. Coba tanya tentang: CV, interview, skill IT, karir transition, atau workplace skills!")
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("Riwayat Percakapan")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-3:]), 1):
            with st.expander(f"Pertanyaan {i}: {chat['question']}"):
                st.write(f"*Jawaban:* {chat['answer']}")
                st.caption(f"Confidence: {chat['confidence']:.2%}")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("### 🛠 Tech Stack")
        st.markdown("""
        - *Python* + *Streamlit*
        - *Scikit-learn* (TF-IDF + Cosine Similarity)
        - *Machine Learning* Powered
        - *Hugging Face* Deployment
        """)
        
        st.markdown("### AI Implementation")
        st.markdown("""
        - *Algorithm:* TF-IDF Vectorization
        - *Similarity:* Cosine Similarity
        - *Training:* Supervised Learning
        - *Dataset:* 15 Career Q&A Pairs
        """)

if _name_ == "_main_":
    main()
