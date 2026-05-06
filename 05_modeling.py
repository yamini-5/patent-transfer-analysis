import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import jensenshannon
from itertools import permutations
import logging


# ─────────────────────────────────────────────────────────────────────────────
# Feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_vocab_overlap(df, source, target, text_col='clean_text'):
    """
    Jaccard index between the vocabulary sets of source and target domains.
    Higher value → more shared vocabulary → potentially safer transfer.
    """
    src_vocab = set(
        " ".join(df[df['domain'] == source][text_col].fillna('').tolist()).split()
    )
    tgt_vocab = set(
        " ".join(df[df['domain'] == target][text_col].fillna('').tolist()).split()
    )
    intersection = len(src_vocab & tgt_vocab)
    union = len(src_vocab | tgt_vocab)
    return round(intersection / union, 4) if union > 0 else 0.0


def compute_label_shift(y_source, y_target):
    """
    Jensen-Shannon Divergence between label distributions of source and target.
    Higher value → greater domain mismatch → higher negative-transfer risk.
    """
    classes = np.unique(np.concatenate([y_source, y_target]))

    def dist(y):
        counts = np.array([np.sum(y == c) for c in classes], dtype=float)
        counts += 1e-10  # Laplace smoothing
        return counts / counts.sum()

    jsd = jensenshannon(dist(y_source), dist(y_target))
    return round(float(jsd), 4)


def compute_prediction_stats(clf, X_target, y_true):
    """
    Applies the source model to target data and returns:
      - avg_confidence : mean of the highest class probability per sample
      - entropy        : mean prediction entropy (uncertainty)
      - error_rate     : fraction of incorrect predictions (1 - accuracy)
    """
    proba = clf.predict_proba(X_target)          # shape (n_samples, n_classes)
    y_pred = clf.predict(X_target)
    max_proba = proba.max(axis=1)

    avg_confidence = float(max_proba.mean())
    entropy = float(-np.sum(proba * np.log(proba + 1e-10), axis=1).mean())
    
    # Correct Error Rate: Incorrect / Total
    error_rate = float(1.0 - accuracy_score(y_true, y_pred))

    return round(avg_confidence, 4), round(entropy, 4), round(error_rate, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_models(df, vectors, test_size=0.2, random_state=42):
    """
    Evaluates Baseline vs Transfer Learning for all domain pairs and
    extracts the full 7-feature vector used by the meta-model.

    Returns a DataFrame with one row per (source, target) pair containing:
        f1_baseline, f1_transfer, delta_f1, negative_transfer,
        label_shift, vocab_overlap, avg_confidence, entropy, error_rate
    """
    domains = df['domain'].unique()
    pairs = list(permutations(domains, 2))

    # ── Pre-split every domain once so Baseline and Transfer use identical test sets ──
    target_data = {}
    for dom in domains:
        idx = df[df['domain'] == dom].index
        X_dom = vectors[idx]
        y_dom = df.loc[idx, 'target_label'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X_dom, y_dom, test_size=test_size, random_state=random_state
        )
        target_data[dom] = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'X_all': X_dom,     'y_all': y_dom,
        }

    # ── Pre-train one classifier per domain (reused as "source model") ──
    source_clfs = {}
    for dom in domains:
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        clf.fit(target_data[dom]['X_train'], target_data[dom]['y_train'])
        source_clfs[dom] = clf

    # ── Evaluate every (source → target) pair ──
    results = []
    for source, target in pairs:
        logging.info(f"  Evaluating: {source} → {target}")

        # 1. Baseline Model — trained & tested entirely on target domain
        clf_baseline = LogisticRegression(max_iter=1000, random_state=random_state)
        clf_baseline.fit(target_data[target]['X_train'], target_data[target]['y_train'])
        y_pred_base = clf_baseline.predict(target_data[target]['X_test'])
        f1_base  = f1_score(target_data[target]['y_test'], y_pred_base, average='weighted', zero_division=0)
        acc_base = accuracy_score(target_data[target]['y_test'], y_pred_base)

        # 2. Transfer Model — source model applied directly to target test set
        clf_transfer = source_clfs[source]
        y_pred_trans = clf_transfer.predict(target_data[target]['X_test'])
        f1_trans  = f1_score(target_data[target]['y_test'], y_pred_trans, average='weighted', zero_division=0)
        acc_trans = accuracy_score(target_data[target]['y_test'], y_pred_trans)

        delta_f1     = f1_trans - f1_base
        neg_transfer = delta_f1 < 0

        # 3. Extended features for meta-model
        # Use PREDICTED labels on target data to estimate label shift (no data leakage)
        y_pred_target_all = clf_transfer.predict(target_data[target]['X_all'])
        estimated_label_shift = compute_label_shift(
            target_data[source]['y_all'], y_pred_target_all
        )
        vocab_overlap = compute_vocab_overlap(df, source, target)
        avg_conf, entropy, error_rate = compute_prediction_stats(
            clf_transfer, target_data[target]['X_test'], target_data[target]['y_test']
        )

        results.append({
            'source':           source,
            'target':           target,
            # Core metrics
            'f1_baseline':      round(f1_base,  4),
            'acc_baseline':     round(acc_base,  4),
            'f1_transfer':      round(f1_trans,  4),
            'acc_transfer':     round(acc_trans,  4),
            'delta_f1':         round(delta_f1,  4),
            'negative_transfer': neg_transfer,
            'magnitude_drop':   round(abs(delta_f1), 4),
            # Meta-model features
            'estimated_label_shift': estimated_label_shift,
            'vocab_overlap':    vocab_overlap,
            'avg_confidence':   avg_conf,
            'entropy':          entropy,
            'error_rate':       error_rate,
        })

    return pd.DataFrame(results)
