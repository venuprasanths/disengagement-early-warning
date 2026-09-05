import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from src.features import prepare_feature_matrix
from src.baseline_model import LaggingAttendanceMarksBaseline
from src.main_model import TransparentMultiSignalModel, SIGNAL_FAMILY_MAPPING

df = pd.read_csv("data/synthetic_cohort.csv")
df_train = df[df["week"] <= 10].copy()
df_test = df[df["week"] >= 11].copy()

X_train, y_train, _ = prepare_feature_matrix(df_train)
X_test, y_test, _ = prepare_feature_matrix(df_test)

main_model = TransparentMultiSignalModel(random_state=42).fit(X_train, y_train)
baseline = LaggingAttendanceMarksBaseline()

print(f"Dataset: {len(df)} total rows across {df['student_id'].nunique()} students.")
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# 1. Permutation Feature Importance on Test Set
perm_res = permutation_importance(
    main_model.calibrated_model,
    X_test[main_model.feature_names],
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="roc_auc",
)

sorted_idx = perm_res.importances_mean.argsort()[::-1]
total_imp = np.sum(np.maximum(0, perm_res.importances_mean)) or 1.0

print("\n=== TOP CONTRIBUTING FEATURES (FULL MODEL PERMUTATION IMPORTANCE ON TEST SET) ===")
for i in sorted_idx[:10]:
    f = main_model.feature_names[i]
    mean_imp = perm_res.importances_mean[i]
    rel_wt = max(0.0, mean_imp) / total_imp * 100.0
    fam = SIGNAL_FAMILY_MAPPING.get(f, "Other")
    print(f"  {f:28s} [{fam:18s}]: AUC Drop = {mean_imp:.4f} (Rel Weight: {rel_wt:.1f}%)")

# Group by Signal Family
family_imp = {}
for i in range(len(main_model.feature_names)):
    f = main_model.feature_names[i]
    fam = SIGNAL_FAMILY_MAPPING.get(f, "Other")
    family_imp[fam] = family_imp.get(fam, 0.0) + max(0.0, perm_res.importances_mean[i])

print("\n=== SIGNAL FAMILY CONTRIBUTION TO AUC ===")
fam_total = sum(family_imp.values()) or 1.0
for fam, val in sorted(family_imp.items(), key=lambda x: x[1], reverse=True):
    print(f"  {fam:22s}: Total Drop = {val:.4f} ({val/fam_total*100:.1f}%)")

# 2. Ablation Study: Assessment-Only vs Full Model
assessment_features = [
    "quiz_score_recent",
    "cumulative_score_avg",
    "score_trend_slope",
    "score_variance",
]

# Fit assessment-only model
from sklearn.calibration import CalibratedClassifierCV

assess_base = HistGradientBoostingClassifier(
    max_iter=100, max_leaf_nodes=15, min_samples_leaf=20, learning_rate=0.08, random_state=42
)
model_assess = CalibratedClassifierCV(estimator=assess_base, method="sigmoid", cv=3)
model_assess.fit(X_train[assessment_features], y_train)

pred_assess_prob = model_assess.predict_proba(X_test[assessment_features])[:, 1]
pred_assess_bin = pred_assess_prob >= 0.50

auc_assess = roc_auc_score(y_test, pred_assess_prob)
f1_assess = f1_score(y_test, pred_assess_bin)
rec_assess = recall_score(y_test, pred_assess_bin)
prec_assess = precision_score(y_test, pred_assess_bin)

# Full model metrics on test set
full_prob = main_model.predict_risk_score(X_test)
full_bin = full_prob >= 0.50
auc_full = roc_auc_score(y_test, full_prob)
f1_full = f1_score(y_test, full_bin)
rec_full = recall_score(y_test, full_bin)
prec_full = precision_score(y_test, full_bin)

print("\n=== ABLATION STUDY RESULTS (HOLDOUT TEST SET: WEEKS 11-16) ===")
print(f"Full 5-Signal Model : ROC-AUC = {auc_full:.4f} | F1 = {f1_full:.4f} | Recall = {rec_full:.4f} | Precision = {prec_full:.4f}")
print(f"Assessment-Only     : ROC-AUC = {auc_assess:.4f} | F1 = {f1_assess:.4f} | Recall = {rec_assess:.4f} | Precision = {prec_assess:.4f}")

# Archetype Specific Ablation Comparison
test_arch = df_test["archetype"].values
print("\n=== RECALL COMPARISON ACROSS ARCHETYPES (FULL VS ASSESSMENT-ONLY) ===")
for arch in ["quietly_struggling", "checked_out", "signal_gamer", "genuinely_improving"]:
    mask = (test_arch == arch) & (y_test == 1)
    if np.sum(mask) > 0:
        r_full = recall_score(y_test[test_arch == arch], full_bin[test_arch == arch])
        r_assess = recall_score(y_test[test_arch == arch], pred_assess_bin[test_arch == arch])
        print(f"  [{arch:20s}] Full Model Recall = {r_full:.3f} vs Assess-Only = {r_assess:.3f}")

# False Positive Rate on Genuinely Improving
recov_mask = (test_arch == "genuinely_improving") & (y_test == 0)
fp_full = np.mean(full_bin[recov_mask])
fp_assess = np.mean(pred_assess_bin[recov_mask])
print(f"  [recovering FP rate  ] Full Model FP Rate = {fp_full:.3f} vs Assess-Only = {fp_assess:.3f}")

# 3. Lead Time Calculation Bug Fix
dis = df[df["is_disengaged"] == True]["student_id"].unique()
X_all, _, _ = prepare_feature_matrix(df)
df["main_flag"] = main_model.predict_risk_score(X_all) >= 0.50
df["base_flag"] = baseline.predict(df)

lead_weeks = []
for s_id in dis:
    s_data = df[df["student_id"] == s_id].sort_values("week")
    m_weeks = s_data[s_data["main_flag"] == True]["week"]
    b_weeks = s_data[s_data["base_flag"] == True]["week"]
    first_m = int(m_weeks.min()) if not m_weeks.empty else 17
    first_b = int(b_weeks.min()) if not b_weeks.empty else 17
    lead_weeks.append(first_b - first_m)

lead_arr_d = np.array(lead_weeks) * 7.0

boot_medians_d = [np.median(np.random.choice(lead_arr_d, size=len(lead_arr_d), replace=True)) for _ in range(2000)]
boot_means_d = [np.mean(np.random.choice(lead_arr_d, size=len(lead_arr_d), replace=True)) for _ in range(2000)]

print("\n=== DEBUGGED LEAD TIME METRICS ===")
med_days = float(np.median(lead_arr_d))
ci_med_low = float(np.percentile(boot_medians_d, 2.5))
ci_med_high = float(np.percentile(boot_medians_d, 97.5))

mean_days = float(np.mean(lead_arr_d))
ci_mean_low = float(np.percentile(boot_means_d, 2.5))
ci_mean_high = float(np.percentile(boot_means_d, 97.5))

print(f"Median Lead Time: {med_days:.1f} days | 95% CI of MEDIAN: [{ci_med_low:.1f}, {ci_med_high:.1f}] days")
print(f"Mean Lead Time  : {mean_days:.1f} days | 95% CI of MEAN  : [{ci_mean_low:.1f}, {ci_mean_high:.1f}] days")
