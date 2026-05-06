import pandas as pd
import json
import os

def export_to_js():
    results_path   = "results/final_results.csv"
    decisions_path = "results/transfer_decisions.csv"
    importance_path= "results/feature_importances.csv"

    if not os.path.exists(results_path):
        print("ERROR: results/final_results.csv not found. Run main.py first.")
        return

    df_results    = pd.read_csv(results_path)
    df_decisions  = pd.read_csv(decisions_path)  if os.path.exists(decisions_path)  else pd.DataFrame()
    df_importance = pd.read_csv(importance_path) if os.path.exists(importance_path) else pd.DataFrame()

    # Merge all data into one flat array for the dashboard
    if not df_decisions.empty:
        df_merged = pd.merge(df_results, df_decisions, on=['source','target'], how='left')
    else:
        df_merged = df_results.copy()

    # Rename columns to match what the dashboard JavaScript expects
    rename_map = {}
    if 'estimated_label_shift' in df_merged.columns:
        rename_map['estimated_label_shift'] = 'label_shift'
    if rename_map:
        df_merged = df_merged.rename(columns=rename_map)

    # Compute summary KPIs
    best_sim_row   = df_merged.loc[df_merged['similarity'].idxmax()]
    worst_f1_row   = df_merged.loc[df_merged['delta_f1'].idxmin()]
    neg_count      = int(df_results['negative_transfer'].sum())
    total_pairs    = len(df_merged)
    avg_confidence = round(df_merged['avg_confidence'].mean(), 4) if 'avg_confidence' in df_merged.columns else None
    # label_shift is the renamed column; fall back to estimated_label_shift if rename didn't happen
    shift_col      = 'label_shift' if 'label_shift' in df_merged.columns else 'estimated_label_shift'
    avg_label_shift= round(df_merged[shift_col].mean(), 4) if shift_col in df_merged.columns else None

    # Strategy comparison (read from decisions)
    safe_pairs   = 0
    unsafe_pairs = 0
    if not df_decisions.empty and 'final_decision' in df_decisions.columns:
        safe_pairs   = int(df_decisions['final_decision'].str.contains('SAFE').sum())
        unsafe_pairs = int(df_decisions['final_decision'].str.contains('DO NOT').sum())

    kpis = {
        "highest_sim_pair":   f"{best_sim_row['source']} → {best_sim_row['target']}",
        "highest_sim_value":  round(float(best_sim_row['similarity']), 4),
        "worst_transfer_pair":f"{worst_f1_row['source']} → {worst_f1_row['target']}",
        "worst_delta_f1":     round(float(worst_f1_row['delta_f1']), 4),
        "neg_transfer_count": neg_count,
        "total_pairs":        total_pairs,
        "safe_pairs":         safe_pairs,
        "unsafe_pairs":       unsafe_pairs,
        "avg_confidence":     avg_confidence,
        "avg_label_shift":    avg_label_shift,
    }

    importance_list = df_importance.to_dict(orient='records') if not df_importance.empty else []

    os.makedirs("dashboard", exist_ok=True)
    js_content = (
        f"const transferData = {json.dumps(df_merged.to_dict(orient='records'), indent=2)};\n\n"
        f"const dashboardKPIs = {json.dumps(kpis, indent=2)};\n\n"
        f"const featureImportances = {json.dumps(importance_list, indent=2)};\n"
    )

    with open("dashboard/data.js", "w", encoding="utf-8") as f:
        f.write(js_content)

    print(f"[OK] Exported dashboard/data.js  ({total_pairs} pairs, {neg_count} negative transfers)")

if __name__ == "__main__":
    export_to_js()
