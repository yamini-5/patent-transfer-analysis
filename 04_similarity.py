import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import permutations
import pandas as pd

def compute_domain_similarity(df, embeddings):
    """
    Computes average vectors per domain and calculates pairwise cosine similarity.
    """
    domain_vectors = {}
    domains = df['domain'].unique()
    
    # Calculate average vector for each domain
    for domain in domains:
        idx = df[df['domain'] == domain].index
        domain_emb = embeddings[idx]
        avg_vec = np.mean(domain_emb, axis=0)
        domain_vectors[domain] = avg_vec
        
    pairs = list(permutations(domains, 2))
    similarity_results = []
    
    for domain_a, domain_b in pairs:
        vec_a = domain_vectors[domain_a].reshape(1, -1)
        vec_b = domain_vectors[domain_b].reshape(1, -1)
        
        sim = cosine_similarity(vec_a, vec_b)[0][0]
        
        similarity_results.append({
            'source': domain_a,
            'target': domain_b,
            'similarity': sim
        })
        
    return pd.DataFrame(similarity_results)

if __name__ == "__main__":
    df = pd.DataFrame({'domain': ['Software', 'Software', 'Pharma', 'Pharma']})
    embs = np.random.rand(4, 384)
    sim_df = compute_domain_similarity(df, embs)
    print(sim_df)
