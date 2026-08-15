"""
Run the full propensity-model pipeline:
  1. Load + clean data
  2. Cross-validated evaluation (honest metrics on small/imbalanced data)
  3. Fit final model on all data
  4. Score every user and save results

Run with: python3 main.py
"""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import config
from data_prep import load_features
from model import evaluate_with_cv, fit_final_model, get_feature_importance


def main():
    df, X, y = load_features()
    print(f"Loaded {len(df)} rows, {y.sum()} converters ({y.mean():.1%})")

    #  Evaluation 
    metrics = evaluate_with_cv(X, y)
    print("\n=== 5-FOLD CROSS-VALIDATED METRICS ===")
    print(f"AUC-ROC: {metrics['auc']:.3f}  (0.5 = random ranking, 1.0 = perfect)")
    print(f"\nTop {config.TOP_K_FRACTION:.0%} lead list (n={metrics['k']}):")
    print(f"  Precision@K: {metrics['precision_at_k']:.3f}  "
          f"(base rate {metrics['base_rate']:.3f}, lift = {metrics['lift_at_k']:.2f}x)")
    print(f"  Recall@K:    {metrics['recall_at_k']:.3f}  "
          f"(captures {metrics['recall_at_k']:.0%} of all converters)")

    # Final model + feature importance 
    final_model = fit_final_model(X, y)
    importance_df = get_feature_importance(final_model, X, y)
    print("\n=== TOP FEATURES ===")
    print(importance_df.head(12).to_string(index=False))

    #  Score every user 
    df = df.copy()
    df["propensity_score"] = final_model.predict_proba(X)[:, 1].round(4)
    df_sorted = df.sort_values("propensity_score", ascending=False)

    df_sorted.to_csv(config.SCORED_USERS_CSV, index=False)
    importance_df.to_csv(config.FEATURE_IMPORTANCE_CSV, index=False)
    print(f"\nSaved: {config.SCORED_USERS_CSV}, {config.FEATURE_IMPORTANCE_CSV}")


if __name__ == "__main__":
    main()