import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import logging
import os

# Features fed into the meta-model (similarity comes from 04_similarity.py merge)
META_FEATURE_COLS = [
    'similarity', 'avg_confidence', 'entropy', 'estimated_label_shift'
]


# ─────────────────────────────────────────────────────────────────────────────
# Correlation
# ─────────────────────────────────────────────────────────────────────────────

def check_correlation(similarity_df, results_df):
    """Merges similarity scores with model results and computes Pearson correlation."""
    df_merged = pd.merge(similarity_df, results_df, on=['source', 'target'])
    
    corr, p_val = pearsonr(df_merged['similarity'], df_merged['delta_f1'])
    logging.info(
        f"Pearson Correlation (Similarity vs Delta F1): {corr:.4f}  (p={p_val:.4f})"
    )
    return df_merged, corr


# ─────────────────────────────────────────────────────────────────────────────
# Legacy threshold (kept for comparison)
# ─────────────────────────────────────────────────────────────────────────────

def find_best_threshold(df_merged):
    """
    Grid search over similarity thresholds to best predict safe/unsafe transfer.
    Returns the threshold with highest accuracy vs actual delta_f1 sign.
    """
    thresholds = np.round(np.arange(0.10, 0.95, 0.05), 2)
    best_threshold, best_accuracy = None, 0.0
    actual_good = df_merged['delta_f1'] >= 0.02

    for thresh in thresholds:
        predicted = df_merged['similarity'] >= thresh
        acc = np.mean(predicted == actual_good)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = thresh

    logging.info(
        f"Best Similarity Threshold (legacy): {best_threshold:.2f}  "
        f"Accuracy: {best_accuracy:.2f}"
    )
    return best_threshold


# ─────────────────────────────────────────────────────────────────────────────
# Meta-model (supervised — the BIG UPGRADE)
# ─────────────────────────────────────────────────────────────────────────────

def train_meta_model(df_merged):
    """
    Trains meta-models predicting Safe (1) / Unsafe (0) transfer using LODO CV.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler

    feature_cols = [f for f in META_FEATURE_COLS if f in df_merged.columns]
    X = df_merged[feature_cols].values
    
    # Binary Labeling: Safe (1) vs Unsafe (0)
    y = (df_merged['delta_f1'] >= 0.0).astype(int).values
    
    n_safe = int(y.sum())
    n_unsafe = len(y) - n_safe
    logging.info(f"\nMeta-Model Dataset: {len(df_merged)} pairs | Safe={n_safe} Unsafe={n_unsafe} | Features={feature_cols}")

    groups = df_merged['target'].values
    logo = LeaveOneGroupOut()
    
    scaler = StandardScaler()
    
    lodo_preds_rf = np.zeros(len(y))
    lodo_probas_rf = np.zeros(len(y))
    lodo_preds_lr = np.zeros(len(y))
    lodo_probas_lr = np.zeros(len(y))
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rf.fit(X_train_scaled, y_train)
        lodo_preds_rf[test_idx] = rf.predict(X_test_scaled)
        lodo_probas_rf[test_idx] = rf.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y_train)) > 1 else 0
        
        lr.fit(X_train_scaled, y_train)
        lodo_preds_lr[test_idx] = lr.predict(X_test_scaled)
        lodo_probas_lr[test_idx] = lr.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y_train)) > 1 else 0

    # Calculate Variance over 5 seeds
    rf_seeds = [42, 123, 456, 789, 999]
    precs, recs, f1s, aucs = [], [], [], []
    for s in rf_seeds:
        seed_preds = np.zeros(len(y))
        seed_probas = np.zeros(len(y))
        seed_rf = RandomForestClassifier(n_estimators=100, random_state=s, class_weight='balanced')
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            seed_rf.fit(X_train_scaled, y_train)
            seed_preds[test_idx] = seed_rf.predict(X_test_scaled)
            seed_probas[test_idx] = seed_rf.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y_train)) > 1 else 0
            
        precs.append(precision_score(y, seed_preds, zero_division=0))
        recs.append(recall_score(y, seed_preds, zero_division=0))
        f1s.append(f1_score(y, seed_preds, zero_division=0))
        aucs.append(roc_auc_score(y, seed_probas) if len(np.unique(y)) > 1 else 0)
        
    rf_variance = {
        'prec_mean': np.mean(precs), 'prec_std': np.std(precs),
        'rec_mean': np.mean(recs), 'rec_std': np.std(recs),
        'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s),
        'auc_mean': np.mean(aucs), 'auc_std': np.std(aucs)
    }

    def print_metrics(name, y_true, y_pred, y_proba):
        auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        logging.info(f"LODO CV {name} | AUC: {auc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")
    
    print_metrics("Random Forest", y, lodo_preds_rf, lodo_probas_rf)
    print_metrics("Logistic Regression", y, lodo_preds_lr, lodo_probas_lr)

    # Train final models on ALL data
    X_scaled = scaler.fit_transform(X)
    rf.fit(X_scaled, y)
    rf.scaler = scaler
    
    lr.fit(X_scaled, y)
    lr.scaler = scaler
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'rf_importance': rf.feature_importances_,
        'lr_coef': lr.coef_[0]
    }).sort_values('rf_importance', ascending=False).reset_index(drop=True)
    
    logging.info(f"\nFeature Importances & Coefficients:\n{importance_df.to_string(index=False)}")
    
    # Store lodo probas in df for plotting
    df_merged['lodo_proba_rf'] = lodo_probas_rf
    df_merged['lodo_probas_lr'] = lodo_probas_lr
    df_merged['lodo_preds_lr'] = lodo_preds_lr
    
    return rf, feature_cols, importance_df, rf_variance

def run_ablation_study(df_merged, feature_cols):
    """
    Evaluates meta-model performance while dropping one feature at a time.
    Returns a dictionary of results.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, precision_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
    
    y = (df_merged['delta_f1'] >= 0.0).astype(int).values
    groups = df_merged['target'].values
    
    def evaluate_features(feats):
        if not feats:
            return 0.0, 0.0
        X = df_merged[feats].values
        logo = LeaveOneGroupOut()
        scaler = StandardScaler()
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        
        preds = np.zeros(len(y))
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            if len(np.unique(y_train)) < 2:
                preds[test_idx] = 0
                continue
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            rf.fit(X_train_scaled, y_train)
            preds[test_idx] = rf.predict(X_test_scaled)
            
        f1 = f1_score(y, preds, zero_division=0)
        prec = precision_score(y, preds, zero_division=0)
        return f1, prec
        
    base_f1, base_prec = evaluate_features(feature_cols)
    logging.info(f"\n--- Ablation Study (Base F1: {base_f1:.4f}, Prec: {base_prec:.4f}) ---")
    
    ablation_results = {'Base': {'f1': base_f1, 'prec': base_prec}}
    
    for feat in feature_cols:
        reduced_feats = [f for f in feature_cols if f != feat]
        f1, prec = evaluate_features(reduced_feats)
        drop_f1 = base_f1 - f1
        logging.info(f"Dropped '{feat:<22}': F1 = {f1:.4f} (Drop: {drop_f1:+.4f}) | Prec = {prec:.4f}")
        ablation_results[f'Drop_{feat}'] = {'f1': f1, 'prec': prec}
        
    return ablation_results

# ─────────────────────────────────────────────────────────────────────────────
# Decision engine
# ─────────────────────────────────────────────────────────────────────────────

def predict_transfer(df_merged, meta_model, feature_cols):
    """
    Makes transfer decisions using binary meta-model probabilities.
    """
    available = [f for f in feature_cols if f in df_merged.columns]
    X = df_merged[available].values
    X_scaled = meta_model.scaler.transform(X)
    
    meta_preds  = meta_model.predict(X_scaled)
    meta_probas = meta_model.predict_proba(X_scaled)[:, 1] # Prob(Safe)

    records = []
    for i, (_, row) in enumerate(df_merged.iterrows()):
        prob_safe = round(float(meta_probas[i]), 4)
        is_safe = bool(meta_preds[i])

        if prob_safe > 0.6:
            decision = "SAFE TO TRANSFER [OK]"
            reason = f"Meta-model high confidence ({prob_safe:.2f})"
        elif prob_safe < 0.4:
            decision = "DO NOT TRANSFER [BLOCK]"
            reason = f"Meta-model block ({prob_safe:.2f})"
        else:
            decision = "UNCERTAIN [CAUTION]"
            reason = f"Meta-model uncertain ({prob_safe:.2f})"

        # Ensemble score: blend RF proba with cosine similarity
        sim_val = float(row.get('similarity', 0.5))
        ensemble = round((prob_safe + sim_val) / 2, 4)

        # Error level derived from ensemble score
        if ensemble >= 0.65:
            error_level = 'Low'
        elif ensemble >= 0.4:
            error_level = 'Medium'
        else:
            error_level = 'High'

        records.append({
            'source':           row['source'],
            'target':           row['target'],
            'meta_prediction':  'Safe' if is_safe else 'Unsafe',
            'meta_confidence':  prob_safe,
            'ensemble_score':   ensemble,
            'error_level':      error_level,
            'final_decision':   decision,
            'decision_reason':  reason,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Framework evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_framework(df_merged, threshold):
    """Always / Never / Similarity-Threshold comparison (legacy)."""
    avg_f1_never  = df_merged['f1_baseline'].mean()
    avg_f1_always = df_merged['f1_transfer'].mean()
    f1_framework  = np.where(
        df_merged['similarity'] >= threshold,
        df_merged['f1_transfer'],
        df_merged['f1_baseline']
    ).mean()

    logging.info("--- Decision Framework Results (similarity threshold) ---")
    logging.info(f"Always Transfer : {avg_f1_always:.4f}")
    logging.info(f"Never Transfer  : {avg_f1_never:.4f}")
    logging.info(f"Framework F1    : {f1_framework:.4f}")

    return {
        'Always Transfer': avg_f1_always,
        'Never Transfer':  avg_f1_never,
        'Framework':       f1_framework,
    }


def evaluate_meta_framework(df_merged, decisions_df, rf_variance):
    """Compares baseline rules vs Meta-model."""
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    from scipy.stats import ttest_rel
    
    y_true = (df_merged['delta_f1'] >= 0.0).astype(int).values
    
    def get_metrics(y_pred, y_proba=None):
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_proba) if y_proba is not None and len(np.unique(y_true)) > 1 else 0.5
        return prec, rec, f1, auc
        
    results = {}
    
    # 1. Similarity Threshold
    sim_scores = df_merged['similarity'].values
    sim_thresh = np.median(sim_scores)
    y_pred_sim = (sim_scores > sim_thresh).astype(int)
    results['Similarity Threshold'] = get_metrics(y_pred_sim, sim_scores)
    
    # 2. Entropy Rule
    ent_scores = -df_merged['entropy'].values # Negative because lower entropy is better
    ent_thresh = np.median(ent_scores)
    y_pred_ent = (ent_scores > ent_thresh).astype(int)
    results['Entropy Rule'] = get_metrics(y_pred_ent, ent_scores)
    
    # 3. Random Guessing
    np.random.seed(42)
    y_proba_rand = np.random.rand(len(y_true))
    y_pred_rand = (y_proba_rand > 0.5).astype(int)
    results['Random Guessing'] = get_metrics(y_pred_rand, y_proba_rand)
    
    # 4. Logistic Regression
    if 'lodo_preds_lr' in df_merged.columns:
        y_pred_lr = df_merged['lodo_preds_lr'].values
        y_probas_lr = df_merged['lodo_probas_lr'].values if 'lodo_probas_lr' in df_merged.columns else None
        results['Logistic Regression'] = get_metrics(y_pred_lr, y_probas_lr)
    
    # 5. Meta-Model (Random Forest with Variance)
    y_pred_meta = decisions_df['final_decision'].str.contains('SAFE').astype(int)
    results['Meta-Model'] = (
        rf_variance['prec_mean'], rf_variance['rec_mean'], rf_variance['f1_mean'],
        rf_variance['prec_std'], rf_variance['rec_std'], rf_variance['f1_std'],
        rf_variance['auc_mean'], rf_variance['auc_std']
    )
    
    # Statistical Significance (Paired T-Test)
    acc_meta = (y_pred_meta == y_true).astype(int)
    acc_sim = (y_pred_sim == y_true).astype(int)
    if np.any(acc_meta != acc_sim):
        t_stat, p_val = ttest_rel(acc_meta, acc_sim)
    else:
        p_val = 1.0
    results['p_value_vs_sim'] = p_val
    
    # Calculate % reduction in unsafe transfers
    fp_sim = np.sum((y_pred_sim == 1) & (y_true == 0))
    fp_meta = np.sum((y_pred_meta == 1) & (y_true == 0))
    reduction = ((fp_sim - fp_meta) / fp_sim * 100) if fp_sim > 0 else 0
    results['reduction_unsafe'] = reduction
    
    # Store True Positive / False Positive counts for Confusion Matrix
    cm_counts = {
        'True Safe': np.sum((y_pred_meta == 1) & (y_true == 1)),
        'False Safe': fp_meta,
        'True Unsafe': np.sum((y_pred_meta == 0) & (y_true == 0)),
        'False Unsafe': np.sum((y_pred_meta == 0) & (y_true == 1))
    }
    results['cm_counts'] = cm_counts
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Explainability
# ─────────────────────────────────────────────────────────────────────────────

def explain_mismatch(df, source, target, top_n=10):
    """Extracts top TF-IDF words per domain to surface vocabulary mismatch."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    def top_words(texts, n):
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        X   = vec.fit_transform(texts)
        mean_scores = np.array(X.mean(axis=0)).flatten()
        return list(np.array(vec.get_feature_names_out())[mean_scores.argsort()[-n:][::-1]])

    src_top = top_words(df[df['domain'] == source]['clean_text'].fillna('').tolist(), top_n)
    tgt_top = top_words(df[df['domain'] == target]['clean_text'].fillna('').tolist(), top_n)

    logging.info(f"\nExplainability Mismatch — {source} → {target}:")
    logging.info(f"  Top {source} words : {', '.join(src_top)}")
    logging.info(f"  Top {target} words : {', '.join(tgt_top)}")


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def generate_visualizations(df_merged, threshold, decisions_df=None, output_dir="plots"):
    """Generates all plots and saves to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Baseline vs Transfer F1 bar chart
    plt.figure(figsize=(14, 6))
    df_merged['pair_name'] = df_merged['source'] + " → " + df_merged['target']
    df_melt = df_merged.melt(
        id_vars='pair_name',
        value_vars=['f1_baseline', 'f1_transfer'],
        var_name='Model', value_name='F1 Score'
    )
    sns.barplot(data=df_melt, x='pair_name', y='F1 Score', hue='Model')
    plt.xticks(rotation=45, ha='right')
    plt.title("Baseline vs Transfer F1 Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bar_baseline_vs_transfer.png'))
    plt.close()

    # 2. Similarity vs ΔF1 scatter
    plt.figure(figsize=(8, 6))
    colors = {True: 'red', False: 'green'}
    sns.scatterplot(
        data=df_merged, x='similarity', y='delta_f1',
        hue='negative_transfer', palette=colors, s=110
    )
    plt.axhline(0, color='gray', linestyle='--', label='ΔF1 = 0')
    if threshold is not None:
        plt.axvline(threshold, color='blue', linestyle=':', label=f'Threshold {threshold:.2f}')
    plt.title("Domain Similarity vs ΔF1 (Transfer − Baseline)")
    plt.xlabel("Cosine Similarity (BERT)")
    plt.ylabel("ΔF1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_similarity_vs_deltaF1.png'))
    plt.close()

    # 3. Feature correlation heatmap (NEW)
    feat_cols = [f for f in META_FEATURE_COLS if f in df_merged.columns]
    if len(feat_cols) > 1:
        plt.figure(figsize=(9, 7))
        corr_m = df_merged[feat_cols].corr()
        sns.heatmap(corr_m, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0, square=True, linewidths=0.5)
        plt.title("Feature Correlation Heatmap (Pre-Transfer Features Only)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_correlation_heatmap.png'))
        plt.close()

    # 4. Meta-model decision summary bar
    if decisions_df is not None:
        plt.figure(figsize=(8, 5))
        counts = decisions_df['final_decision'].value_counts()
        bar_colors = ['#2ecc71' if 'SAFE' in d else '#f1c40f' if 'UNCERTAIN' in d else '#e74c3c' for d in counts.index]
        plt.bar(counts.index, counts.values, color=bar_colors, edgecolor='white')
        plt.title("Transfer Decision Summary - Meta-Model")
        plt.ylabel("Number of Domain Pairs")
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'meta_decision_summary.png'))
        plt.close()

    # 5. Transfer Risk Curve (The "Killer Graph")
    if 'lodo_proba_rf' in df_merged.columns:
        # Calculate Correlation and R²
        corr_rf, _ = pearsonr(df_merged['lodo_proba_rf'], df_merged['delta_f1'])
        r2_rf = corr_rf ** 2
        logging.info(f"Transfer Risk Curve - Pearson r: {corr_rf:.4f}, R²: {r2_rf:.4f}")
        
        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=df_merged, x='lodo_proba_rf', y='delta_f1',
            scatter_kws={'alpha': 0.6, 's': 50},
            line_kws={'color': 'red', 'linestyle': '--', 'label': f'Trend (R² = {r2_rf:.3f})'}
        )
        plt.axhline(0, color='gray', linestyle='--', label='ΔF1 = 0')
        plt.axvline(0.5, color='blue', linestyle=':', label='Decision Threshold')
        plt.title(f"Transfer Risk Curve (Pearson r: {corr_rf:.3f}, R²: {r2_rf:.3f})")
        plt.xlabel("Meta-Model Predicted Probability of Safe Transfer")
        plt.ylabel("Actual ΔF1 (Transfer - Baseline)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'transfer_risk_curve.png'))
        plt.close()

    # 6. ROC Curve
    from sklearn.metrics import roc_curve, auc
    y_true = (df_merged['delta_f1'] >= 0.0).astype(int).values
    if 'lodo_proba_rf' in df_merged.columns and len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, df_merged['lodo_proba_rf'])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (Unsafe passed as Safe)')
        plt.ylabel('True Positive Rate (Safe passed correctly)')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
        
    # 7. Calibration Curve (Reliability Diagram)
    from sklearn.calibration import calibration_curve
    if 'lodo_proba_rf' in df_merged.columns and len(np.unique(y_true)) > 1:
        prob_true, prob_pred = calibration_curve(y_true, df_merged['lodo_proba_rf'], n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Meta-Model (RF)')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        plt.xlabel('Predicted Probability of Safe Transfer')
        plt.ylabel('Fraction of True Safe Transfers')
        plt.title('Calibration Curve (Reliability Diagram)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'calibration_curve.png'))
        plt.close()
        
    # 8. Precision vs Threshold
    from sklearn.metrics import precision_recall_curve
    if 'lodo_proba_rf' in df_merged.columns and len(np.unique(y_true)) > 1:
        precisions, recalls, thresholds = precision_recall_curve(y_true, df_merged['lodo_proba_rf'])
        
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision (Safe Class)", linewidth=2)
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
        plt.xlabel("Decision Threshold")
        plt.ylabel("Score")
        plt.title("Precision and Recall vs. Decision Threshold")
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_vs_threshold.png'))
        plt.close()
        
    # 9. Visual Confusion Matrix
    from sklearn.metrics import confusion_matrix
    if 'lodo_proba_rf' in df_merged.columns:
        y_pred = (df_merged['lodo_proba_rf'] >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted Unsafe', 'Predicted Safe'],
                    yticklabels=['Actual Unsafe', 'Actual Safe'])
        plt.title("Meta-Model Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

    logging.info(f"Visualizations saved to '{output_dir}/'")
