import os
import logging
from importlib import import_module

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

dataset_module      = import_module('01_dataset')
preprocessing_module = import_module('02_preprocessing')
features_module     = import_module('03_features')
similarity_module   = import_module('04_similarity')
modeling_module     = import_module('05_modeling')
analysis_module     = import_module('06_analysis')
report_module       = import_module('07_report')


def run_pipeline(dataset_name="patents", no_split=False):
    logging.info("=" * 60)
    logging.info(f"Starting Negative Transfer Evaluation Pipeline ({dataset_name})")
    logging.info("=" * 60)

    # ── Step 1: Data Acquisition ──────────────────────────────────────────────
    logging.info("\n--- Step 1: Data Acquisition ---")
    if dataset_name == "patents":
        df = dataset_module.load_and_prepare_data(n_samples_per_domain=300)
    elif dataset_name == "newsgroups":
        df = dataset_module.load_newsgroups_data(n_samples_per_domain=300)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # ── Step 2: Preprocessing ─────────────────────────────────────────────────
    logging.info("\n--- Step 2: Preprocessing ---")
    df = preprocessing_module.preprocess_dataframe(df, text_column='text')

    # ── Step 2.5: Artificially increase domains by splitting ──────────────────
    import numpy as np
    logging.info("\n--- Step 2.5: Domain Subsetting ---")
    if not no_split:
        logging.info("Splitting domains to simulate intra-domain variability and realistic sub-domain shifts.")
        df_copy = df.copy()
        np.random.seed(42)
        for dom in df_copy['domain'].unique():
            idx = df_copy[df_copy['domain'] == dom].index
            subset_labels = [f"{dom}_{i+1}" for i in range(3)]
            df_copy.loc[idx, 'domain'] = np.random.choice(subset_labels, size=len(idx))
        df = df_copy
        logging.info(f"Split original domains into {len(df['domain'].unique())} subsets for robust cross-domain evaluation.")
    else:
        logging.info("Skipping domain splitting (--no-split enabled). Using macro-domains only.")

    # ── Step 3: Feature Extraction ────────────────────────────────────────────
    logging.info("\n--- Step 3: Feature Extraction ---")
    logging.info("Extracting TF-IDF features for domain classifiers...")
    X_tfidf, vectorizer = features_module.extract_tfidf(df['clean_text'])
    X_tfidf_array = X_tfidf.toarray()

    logging.info("Extracting BERT embeddings for domain similarity...")
    embeddings = features_module.extract_bert(df['clean_text'])

    # ── Step 4: Domain Similarity ─────────────────────────────────────────────
    logging.info("\n--- Step 4: Compute Domain Similarity (BERT cosine) ---")
    similarity_df = similarity_module.compute_domain_similarity(df, embeddings)
    logging.info(f"Generated similarity scores for {len(similarity_df)} domain pairs.")

    # ── Step 5: Modeling + Extended Feature Extraction (UPGRADED) ────────────
    logging.info("\n--- Step 5: Model Training + 7-Feature Extraction ---")
    logging.info("  Computing: ΔF1, label_shift, vocab_overlap, "
                 "avg_confidence, entropy, error_rate per pair...")
    results_df = modeling_module.evaluate_models(df, X_tfidf_array, test_size=0.2)

    # ── Step 6: Merge similarity into results ─────────────────────────────────
    logging.info("\n--- Step 6: Merging Similarity + Results ---")
    df_merged, corr = analysis_module.check_correlation(similarity_df, results_df)

    # ── Step 7: Legacy threshold (kept for comparison) ────────────────────────
    logging.info("\n--- Step 7: Legacy Similarity Threshold (grid search) ---")
    best_threshold = analysis_module.find_best_threshold(df_merged)
    framework_results = analysis_module.evaluate_framework(df_merged, best_threshold)

    # ── Step 8: Meta-Model Training (BIG UPGRADE) ────────────────────────────
    logging.info("\n--- Step 8: Supervised Meta-Model Training (Random Forest) ---")
    meta_model, feature_cols, importance_df, rf_variance = analysis_module.train_meta_model(df_merged)

    # ── Step 9: Transfer Decisions via Meta-Model ─────────────────────────────
    logging.info("\n--- Step 9: Generating Transfer Decisions ---")
    decisions_df = analysis_module.predict_transfer(df_merged, meta_model, feature_cols)
    logging.info(f"\n{decisions_df[['source','target','final_decision']].to_string(index=False)}")

    # ── Step 10: Meta-Model Framework Evaluation ──────────────────────────────
    logging.info("\n--- Step 10: Meta-Model Framework Evaluation ---")
    meta_framework_results = analysis_module.evaluate_meta_framework(df_merged, decisions_df, rf_variance)

    # ── Step 11: Explainability sample pair ───────────────────────────────────
    sample_source = df['domain'].iloc[0]
    sample_target = df[df['domain'] != sample_source]['domain'].iloc[0]
    analysis_module.explain_mismatch(df, sample_source, sample_target, top_n=5)

    # ── Step 11.5: Ablation Study ─────────────────────────────────────────────
    logging.info("\n--- Step 11.5: Ablation Study ---")
    ablation_results = analysis_module.run_ablation_study(df_merged, feature_cols)

    # ── Step 12: Visualizations ───────────────────────────────────────────────
    logging.info("\n--- Step 12: Generating Visualizations ---")
    analysis_module.generate_visualizations(
        df_merged, best_threshold,
        decisions_df=decisions_df,
        output_dir="plots"
    )

    # ── Step 13: Save results ─────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    df_merged.to_csv("results/final_results.csv", index=False)
    decisions_df.to_csv("results/transfer_decisions.csv", index=False)
    importance_df.to_csv("results/feature_importances.csv", index=False)
    
    # Save metrics to JSON
    import json
    metrics = {
        'correlation_r': float(corr),
        'best_similarity_threshold': float(best_threshold),
        'ablation_study': ablation_results
    }
    # Add meta_framework_results safely
    metrics['meta_framework_results'] = {}
    for k, v in meta_framework_results.items():
        if isinstance(v, tuple):
            metrics['meta_framework_results'][k] = {'precision': float(v[0]), 'recall': float(v[1]), 'f1': float(v[2])}
        elif isinstance(v, dict):
            metrics['meta_framework_results'][k] = {str(ki): int(vi) for ki, vi in v.items()}
        else:
            metrics['meta_framework_results'][k] = float(v)
            
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    logging.info("Results saved to results/ directory.")

    # ── Step 14: Full Report ──────────────────────────────────────────────────
    logging.info("\n--- Step 14: Generating Report ---")
    report_module.generate_report(
        df, df_merged, corr, best_threshold,
        framework_results, sample_source, sample_target,
        decisions_df=decisions_df,
        importance_df=importance_df,
        meta_framework_results=meta_framework_results,
        ablation_results=ablation_results
    )

    logging.info("\n✅ Pipeline Completed! Check 'results/' and 'plots/' directories.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Negative Transfer Evaluation Pipeline")
    parser.add_argument("--dataset", type=str, default="patents", choices=["patents", "newsgroups"],
                        help="Which dataset to run (patents or newsgroups)")
    parser.add_argument("--no-split", action="store_true",
                        help="Skip domain subsetting baseline")
    args = parser.parse_args()
    
    run_pipeline(dataset_name=args.dataset, no_split=args.no_split)
