from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np

def extract_tfidf(texts, max_features=5000):
    """
    Extract TF-IDF features using unigrams and bigrams.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
    X_tfidf = vectorizer.fit_transform(texts)
    return X_tfidf, vectorizer

def extract_bert(texts, model_name='all-MiniLM-L6-v2'):
    """
    Extract BERT embeddings using sentence-transformers.
    """
    model = SentenceTransformer(model_name)
    # Convert texts to a list if it's a pandas Series
    texts_list = texts.tolist() if hasattr(texts, 'tolist') else texts
    embeddings = model.encode(texts_list, show_progress_bar=True)
    return embeddings

if __name__ == "__main__":
    sample_texts = ["machine learning algorithm", "pharmaceutical drug compound"]
    
    tfidf_matrix, vectorizer = extract_tfidf(sample_texts)
    print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
    
    bert_embeddings = extract_bert(sample_texts)
    print("BERT Embeddings Shape:", bert_embeddings.shape)
