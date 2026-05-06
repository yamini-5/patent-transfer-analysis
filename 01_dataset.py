import os
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SELECTED_DOMAINS = {
    0: "Pharma",
    1: "Mechanical",
    2: "Biotech",
    6: "Software",
    7: "Electrical"
}

def load_and_prepare_data(n_samples_per_domain=1000, random_seed=42):
    """
    Loads patent dataset, filters for selected domains, and balances it.
    Uses KMeans clustering within each domain to generate realistic binary pseudo-labels
    so that transfer learning actually has meaningful linguistic signals to transfer.
    """
    logging.info("Loading dataset ccdv/patent-classification...")
    
    dataset = load_dataset("ccdv/patent-classification", split="train")
    df = dataset.to_pandas()
    
    df_filtered = df[df['label'].isin(SELECTED_DOMAINS.keys())].copy()
    df_filtered['domain'] = df_filtered['label'].map(SELECTED_DOMAINS)
    
    sampled_dfs = []
    
    for domain, group in df_filtered.groupby('domain'):
        if len(group) >= n_samples_per_domain:
            group_sampled = group.sample(n=n_samples_per_domain, random_state=random_seed).copy()
        else:
            group_sampled = group.copy()
            
        group_sampled['text'] = group_sampled['text'].str.slice(0, 500)
        
        # Create realistic sub-labels via KMeans clustering on TF-IDF
        vec = TfidfVectorizer(stop_words='english', max_features=500)
        X = vec.fit_transform(group_sampled['text'].fillna(''))
        kmeans = KMeans(n_clusters=2, random_state=random_seed, n_init=10)
        group_sampled['target_label'] = kmeans.fit_predict(X)
        
        sampled_dfs.append(group_sampled)
            
    df_balanced = pd.concat(sampled_dfs).reset_index(drop=True)
    
    return df_balanced

def load_newsgroups_data(n_samples_per_domain=300, random_seed=42):
    """
    Loads the 20 Newsgroups dataset, mapping subcategories to broad domains.
    Creates a binary target label using KMeans within each domain.
    """
    from sklearn.datasets import fetch_20newsgroups
    logging.info("Loading dataset 20 Newsgroups...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    
    df = pd.DataFrame({
        'text': newsgroups.data,
        'label': newsgroups.target
    })
    
    # Map 20 categories to broad domains
    target_names = newsgroups.target_names
    domain_map = {}
    for i, name in enumerate(target_names):
        broad_domain = name.split('.')[0] # e.g. comp, sci, rec, talk
        domain_map[i] = broad_domain
        
    df['domain'] = df['label'].map(domain_map)
    
    # Filter out empty texts
    df = df[df['text'].str.strip().str.len() > 20].copy()
    
    # Filter out domains with too few samples
    valid_domains = [d for d, c in df['domain'].value_counts().items() if c >= 500]
    df_filtered = df[df['domain'].isin(valid_domains)].copy()
    
    sampled_dfs = []
    
    for domain, group in df_filtered.groupby('domain'):
        if len(group) >= n_samples_per_domain:
            group_sampled = group.sample(n=n_samples_per_domain, random_state=random_seed).copy()
        else:
            group_sampled = group.copy()
            
        group_sampled['text'] = group_sampled['text'].str.slice(0, 1000)
        
        # Create realistic sub-labels via KMeans clustering on TF-IDF
        vec = TfidfVectorizer(stop_words='english', max_features=500)
        X = vec.fit_transform(group_sampled['text'].fillna(''))
        kmeans = KMeans(n_clusters=2, random_state=random_seed, n_init=10)
        group_sampled['target_label'] = kmeans.fit_predict(X)
        
        sampled_dfs.append(group_sampled)
            
    df_balanced = pd.concat(sampled_dfs).reset_index(drop=True)
    
    return df_balanced

if __name__ == "__main__":
    df = load_newsgroups_data(200)
    print(df['domain'].value_counts())
    print(df.groupby('domain')['target_label'].value_counts())
