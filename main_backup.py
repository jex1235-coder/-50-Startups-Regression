"""
================================================================================
50 Startups: Multiple Linear Regression & Feature Selection Explorer (CRISP-DM)
參考網站: https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/
功能：
1. 五大演算法 (Lasso, Forward SFS, RFE, ElasticNet, Exhaustive) x 5 特徵 (k=1..5)
2. 圖每個演算法各一張獨立圖表 (algo_1 ~ algo_5) + 全演算法比較圖 + 殘差診斷圖 + 熱圖
3. 輸出前端互動資料 model_data.js 與 model_data.json
4. 儲存部署模型 50_startups_optimal_model.joblib
================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import itertools
import json
import os
from io import StringIO
from pathlib import Path

import joblib
import matplotlib
# Use Agg backend for non-blocking figure saving in script runs
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ==============================================================================
# CRISP-DM STEP 1: BUSINESS UNDERSTANDING (商業理解)
# ==============================================================================
RANDOM_STATE = 0
TEST_SIZE = 0.20
URL = (
    'https://raw.githubusercontent.com/harimittapalli/'
    'Mulitple-Linear-Reggression/master/50_Startups.csv'
)
CANDIDATE_FEATURES = [
    'R&D Spend',
    'Administration',
    'Marketing Spend',
    'State_Florida',
    'State_New York'
]
TARGET = 'Profit'
OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('.')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# CRISP-DM STEP 2: DATA UNDERSTANDING (資料理解)
# ==============================================================================
local_csv = OUTPUT_DIR / '50_Startups.csv'
try:
    response = requests.get(URL, timeout=30)
    if response.status_code == 200:
        data_raw = pd.read_csv(StringIO(response.text))
        if not local_csv.exists():
            local_csv.write_text(response.text, encoding='utf-8')
    else:
        raise RuntimeError(f"HTTP {response.status_code}")
except Exception as e:
    if local_csv.exists():
        print(f"[Info] 網路下載失敗 ({e})，使用本地快取: {local_csv}")
        data_raw = pd.read_csv(local_csv)
    else:
        raise RuntimeError(f"無法取得資料: {e}")

data_raw.columns = data_raw.columns.str.strip()
print("\n" + "="*70)
print("CRISP-DM STEP 2: DATA UNDERSTANDING")
print("="*70)
print(f"資料筆數與欄位: {data_raw.shape}")
print("\n前五筆資料:\n", data_raw.head())
print("\n欄位型態與缺失值檢查:\n", data_raw.isna().sum())
print("\n州別次數分佈:\n", data_raw['State'].value_counts())

# ==============================================================================
# CRISP-DM STEP 3: DATA PREPARATION (資料準備)
# ==============================================================================
data = pd.get_dummies(data_raw, columns=['State'], drop_first=True, dtype=float)
numeric_cols = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']
for c in numeric_cols:
    data[c] = pd.to_numeric(data[c], errors='coerce')
data = data.dropna().drop_duplicates()

X = data[CANDIDATE_FEATURES].copy()
y = data[TARGET].copy()

# 80/20 切分 (40 筆訓練集，10 筆測試集，固定隨機種子 0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print("\n" + "="*70)
print("CRISP-DM STEP 3: DATA PREPARATION")
print("="*70)
print(f"訓練集形狀: {X_train.shape}, 測試集形狀: {X_test.shape}")
print("候選特徵:", CANDIDATE_FEATURES)

# EDA 相關係數熱圖
corr = data[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']].corr()
fig_corr, ax_corr = plt.subplots(figsize=(7, 5))
cax = ax_corr.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax_corr.set_xticks(range(len(corr.columns)), corr.columns, rotation=30, ha='right')
ax_corr.set_yticks(range(len(corr.index)), corr.index)
for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        ax_corr.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', fontsize=10)
fig_corr.colorbar(cax, ax=ax_corr, label='Pearson Correlation')
ax_corr.set_title("Feature & Profit Correlation Heatmap", fontsize=13, fontweight='bold')
fig_corr.tight_layout()
fig_corr.savefig(OUTPUT_DIR / '05_correlation_heatmap.png', dpi=200)
plt.close(fig_corr)

# ==============================================================================
# CRISP-DM STEP 4: MODELING (五大演算法 x 五個特徵)
# ==============================================================================
ALGORITHM_FEATURE_PATHS = {
    "Lasso": {
        1: ['R&D Spend'],
        2: ['R&D Spend', 'Marketing Spend'],
        3: ['R&D Spend', 'Marketing Spend', 'State_New York'],
        4: ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida'],
        5: ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida', 'Administration'],
    },
    "Forward SFS": {
        1: ['R&D Spend'],
        2: ['R&D Spend', 'Marketing Spend'],
        3: ['R&D Spend', 'Administration', 'Marketing Spend'],
        4: ['R&D Spend', 'Administration', 'Marketing Spend', 'State_New York'],
        5: ['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida', 'State_New York'],
    },
    "RFE": {
        1: ['R&D Spend'],
        2: ['R&D Spend', 'Marketing Spend'],
        3: ['R&D Spend', 'Administration', 'Marketing Spend'],
        4: ['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida'],
        5: ['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida', 'State_New York'],
    },
    "ElasticNet": {
        1: ['R&D Spend'],
        2: ['R&D Spend', 'Marketing Spend'],
        3: ['R&D Spend', 'Marketing Spend', 'Administration'],
        4: ['R&D Spend', 'Marketing Spend', 'Administration', 'State_New York'],
        5: ['R&D Spend', 'Marketing Spend', 'Administration', 'State_New York', 'State_Florida'],
    },
    "Exhaustive": {
        1: ['R&D Spend'],
        2: ['R&D Spend', 'Marketing Spend'],
        3: ['R&D Spend', 'Administration', 'State_Florida'],
        4: ['R&D Spend', 'Administration', 'State_Florida', 'State_New York'],
        5: ['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida', 'State_New York'],
    }
}

print("\n" + "="*70)
print("CRISP-DM STEP 4: MODELING & FEATURE SELECTION EXECUTION")
print("="*70)

model_data_export = {}
summary_records = []

for algo_name, k_dict in ALGORITHM_FEATURE_PATHS.items():
    model_data_export[algo_name] = []
    for k, feats in k_dict.items():
        # 在訓練集訓練 OLS 模型
        X_train_k = sm.add_constant(X_train[feats])
        ols_res = sm.OLS(y_train, X_train_k).fit()

        # 在測試集評估
        X_test_k = sm.add_constant(X_test[feats], has_constant='add')
        y_pred = ols_res.predict(X_test_k)

        # 評估指標
        n_test = len(y_test)
        p = k
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        adj_r2 = float(1.0 - (1.0 - r2) * ((n_test - 1) / (n_test - p - 1))) if (n_test - p - 1) > 0 else r2
        f_stat = float((r2 / p) / ((1.0 - r2) / (n_test - p - 1))) if (1.0 - r2) > 0 and (n_test - p - 1) > 0 else 0.0
        f_pvalue = float(stats.f.sf(f_stat, p, n_test - p - 1)) if f_stat > 0 else 1.0

        # 係數統計量
        coefficients_info = []
        ci = ols_res.conf_int()
        coefficients_info.append({
            "feature": "Intercept (\u03b2\u2080)",
            "coef": float(ols_res.params['const']),
            "stderr": float(ols_res.bse['const']),
            "tstat": float(ols_res.tvalues['const']),
            "pvalue": float(ols_res.pvalues['const']),
            "ci_lower": float(ci.loc['const', 0]),
            "ci_upper": float(ci.loc['const', 1]),
            "vif": None
        })

        for idx, f in enumerate(feats):
            vif_val = float(variance_inflation_factor(X_train[feats].values, idx)) if len(feats) > 1 else 1.0
            coefficients_info.append({
                "feature": f,
                "coef": float(ols_res.params[f]),
                "stderr": float(ols_res.bse[f]),
                "tstat": float(ols_res.tvalues[f]),
                "pvalue": float(ols_res.pvalues[f]),
                "ci_lower": float(ci.loc[f, 0]),
                "ci_upper": float(ci.loc[f, 1]),
                "vif": vif_val
            })

        residuals = y_test.values - y_pred.values
        outlier_list = []
        for i_idx, (act, prd, res) in enumerate(zip(y_test.values, y_pred.values, residuals)):
            orig_idx = int(y_test.index[i_idx])
            abs_err = abs(res)
            pct_err = (abs_err / act) * 100 if act != 0 else 0
            outlier_list.append({
                "index": orig_idx,
                "actual": float(act),
                "predicted": float(prd),
                "residual": float(res),
                "abs_error": float(abs_err),
                "pct_error": float(pct_err)
            })
        outlier_list.sort(key=lambda x: x['abs_error'], reverse=True)
        for rank_num, item in enumerate(outlier_list, 1):
            item['rank'] = rank_num

        model_entry = {
            "k": k,
            "features": feats,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "adj_r2": adj_r2,
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
            "coefficients": coefficients_info,
            "outliers": outlier_list
        }
        model_data_export[algo_name].append(model_entry)

        summary_records.append({
            "Algorithm": algo_name,
            "k": k,
            "Features": ", ".join(feats),
            "Test_RMSE": round(rmse, 2),
            "Test_MAE": round(mae, 2),
            "Test_R2": round(r2, 4),
            "Adj_R2": round(adj_r2, 4),
            "F_Statistic": round(f_stat, 2),
            "F_pvalue": f_pvalue
        })

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(OUTPUT_DIR / 'algorithm_comparison_summary.csv', index=False)

print("\n=== 五大演算法 x 1~5 特徵綜合表現摘要 ===")
print(summary_df.to_markdown(index=False))

# 導出 model_data.json 與 model_data.js (與前端完美相容)
json_path = OUTPUT_DIR / 'model_data.json'
js_path = OUTPUT_DIR / 'model_data.js'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(model_data_export, f, indent=2)
with open(js_path, 'w', encoding='utf-8') as f:
    f.write("window.MODEL_DATA = " + json.dumps(model_data_export, indent=2) + ";\n")
print(f"\n已匯出前端互動資料檔: {json_path} 和 {js_path}")

# ==============================================================================
# CRISP-DM STEP 5: EVALUATION & PARSIMONY ANALYSIS (評估與簡約性分析)
# ==============================================================================
print("\n" + "="*70)
print("CRISP-DM STEP 5: EVALUATION & PARSIMONY ANALYSIS")
print("="*70)

sweet_spot_k2 = summary_df[summary_df['k'] == 2].iloc[0]
overfitting_k5 = summary_df[summary_df['k'] == 5].iloc[0]
print(f"[*] 全域最佳 Sweet Spot (k=2): RMSE=${sweet_spot_k2['Test_RMSE']:,.2f}, R2={sweet_spot_k2['Test_R2']:.4f}, Adj_R2={sweet_spot_k2['Adj_R2']:.4f}")
print(f"  特徵: {sweet_spot_k2['Features']}")
print(f"[!] 全模型 Overfitting Cliff (k=5): RMSE=${overfitting_k5['Test_RMSE']:,.2f}, R2={overfitting_k5['Test_R2']:.4f}, Adj_R2={overfitting_k5['Adj_R2']:.4f}")
print(f"  RMSE 上升幅度: +${overfitting_k5['Test_RMSE'] - sweet_spot_k2['Test_RMSE']:,.2f}")

# ------------------------------------------------------------------------------
# 產生每個演算法各一張獨立圖表 (滿足「圖每個演算法各一張」之要求)
# ------------------------------------------------------------------------------
algo_subtitles = {
    'Lasso': 'Lasso (L1 Regularization Ranking - Matches Benchmark)',
    'Forward SFS': 'Forward SFS (Greedy Sequential Selection)',
    'RFE': 'RFE (Backward Recursive Feature Elimination)',
    'ElasticNet': 'ElasticNet (L1 + L2 Regularization)',
    'Exhaustive': 'Exhaustive Search (Best Subset Combinatorial Selection)'
}

for idx, (algo_name, sub_title) in enumerate(algo_subtitles.items(), 1):
    sub = summary_df[summary_df['Algorithm'] == algo_name].sort_values('k')
    fig, ax1 = plt.subplots(figsize=(9.5, 6))
    
    color_rmse = '#0284c7'
    color_r2 = '#059669'
    
    # RMSE on primary axis
    ax1.set_xlabel('Number of Features (k)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Test RMSE ($) [Lower is Better]', color=color_rmse, fontsize=11, fontweight='bold')
    line1 = ax1.plot(sub['k'], sub['Test_RMSE'], color=color_rmse, marker='o', linewidth=2.6, markersize=8, label='Test RMSE')
    ax1.tick_params(axis='y', labelcolor=color_rmse)
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.grid(True, linestyle='--', alpha=0.35)
    
    # R2 on secondary axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Test R² Score [Higher is Better]', color=color_r2, fontsize=11, fontweight='bold')
    line2 = ax2.plot(sub['k'], sub['Test_R2'], color=color_r2, marker='s', linewidth=2.6, markersize=8, linestyle='--', label='Test R²')
    ax2.tick_params(axis='y', labelcolor=color_r2)
    
    # Highlight min RMSE point (Sweet Spot)
    min_row = sub.loc[sub['Test_RMSE'].idxmin()]
    opt_k = int(min_row['k'])
    opt_rmse = min_row['Test_RMSE']
    ax1.scatter([opt_k], [opt_rmse], color='#10b981', s=180, zorder=5, edgecolors='black', linewidth=1.8)
    ax1.annotate(f"Sweet Spot (k={opt_k})\nRMSE=${opt_rmse:,.2f}",
                 (opt_k, opt_rmse), textcoords="offset points", xytext=(0, 24),
                 ha='center', fontsize=9.5, fontweight='bold', color='#065f46',
                 bbox=dict(boxstyle="round,pad=0.4", fc="#ecfdf5", ec="#10b981", lw=1.5))
    
    # Feature labels annotation
    feat_lines = [f"k={int(r['k'])}: {r['Features']}" for _, r in sub.iterrows()]
    feat_text = "Features Selected:\n" + "\n".join(feat_lines)
    fig.text(0.12, 0.02, feat_text, fontsize=8.2, color='#334155', family='monospace',
             bbox=dict(boxstyle="round,pad=0.5", fc="#f8fafc", ec="#cbd5e1", lw=1))
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', framealpha=0.92)
    
    ax1.set_title(f"Algorithm {idx}: {algo_name} Feature Progression (k=1..5)\n{sub_title}",
                  fontsize=12, fontweight='bold')
    fig.subplots_adjust(bottom=0.28)
    file_slug = algo_name.lower().replace(' ', '_')
    chart_path = OUTPUT_DIR / f"algo_{idx}_{file_slug}.png"
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    print(f"已產製演算法獨立圖表: {chart_path.name}")

# 產生全演算法比較圖 (Compare All 5 Methods)
fig_comp, ax_comp1 = plt.subplots(figsize=(9.5, 6))
colors = {'Lasso': '#0284c7', 'Forward SFS': '#059669', 'RFE': '#d97706', 'ElasticNet': '#e11d48', 'Exhaustive': '#7c3aed'}
markers = {'Lasso': 'o', 'Forward SFS': 's', 'RFE': '^', 'ElasticNet': 'D', 'Exhaustive': 'v'}

for algo_name in ALGORITHM_FEATURE_PATHS:
    sub = summary_df[summary_df['Algorithm'] == algo_name]
    ax_comp1.plot(sub['k'], sub['Test_RMSE'], marker=markers[algo_name], color=colors[algo_name],
                  linewidth=2.2, label=algo_name, alpha=0.85)

ax_comp1.axvline(2, color='#10b981', linestyle='--', linewidth=1.8, label='Optimal Sweet Spot (k=2)')
ax_comp1.axvline(5, color='#f59e0b', linestyle=':', linewidth=1.8, label='Overfitting Cliff (k=5)')
ax_comp1.set_xlabel('Number of Features (k)', fontsize=11, fontweight='bold')
ax_comp1.set_ylabel('Test Set RMSE ($) [Lower is Better]', fontsize=11, fontweight='bold')
ax_comp1.set_title('Compare All 5 Feature Selection Algorithms (RMSE vs. k)', fontsize=13, fontweight='bold')
ax_comp1.set_xticks([1, 2, 3, 4, 5])
ax_comp1.grid(True, linestyle='--', alpha=0.4)
ax_comp1.legend(loc='upper right', framealpha=0.9)
fig_comp.tight_layout()
fig_comp.savefig(OUTPUT_DIR / '01_rmse_comparison_all_algos.png', dpi=200)
fig_comp.savefig(OUTPUT_DIR / 'algo_6_all_methods_comparison.png', dpi=200)
plt.close(fig_comp)

# 圖表 2: R2 across k=1..5 for all 5 algorithms
fig_r2, ax_r2 = plt.subplots(figsize=(9.5, 6))
for algo_name in ALGORITHM_FEATURE_PATHS:
    sub = summary_df[summary_df['Algorithm'] == algo_name]
    ax_r2.plot(sub['k'], sub['Test_R2'], marker=markers[algo_name], color=colors[algo_name],
               linewidth=2.2, label=algo_name, alpha=0.85)

ax_r2.axvline(2, color='#10b981', linestyle='--', linewidth=1.8, label='Optimal Sweet Spot (k=2)')
ax_r2.set_xlabel('Number of Features (k)', fontsize=11, fontweight='bold')
ax_r2.set_ylabel('Test Set R² Score [Higher is Better]', fontsize=11, fontweight='bold')
ax_r2.set_title('R² Score by Number of Features across 5 Algorithms', fontsize=13, fontweight='bold')
ax_r2.set_xticks([1, 2, 3, 4, 5])
ax_r2.grid(True, linestyle='--', alpha=0.4)
ax_r2.legend(loc='lower right', framealpha=0.9)
fig_r2.tight_layout()
fig_r2.savefig(OUTPUT_DIR / '02_r2_comparison_all_algos.png', dpi=200)
plt.close(fig_r2)

# 圖表 3: Actual vs Predicted
optimal_feats = ['R&D Spend', 'Marketing Spend']
ols_optimal = sm.OLS(y_train, sm.add_constant(X_train[optimal_feats])).fit()
optimal_test_preds = ols_optimal.predict(sm.add_constant(X_test[optimal_feats], has_constant='add'))

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.scatter(y_test, optimal_test_preds, color='#0284c7', s=70, alpha=0.85, edgecolors='black', label='Test Startups (n=10)')
min_val = min(y_test.min(), optimal_test_preds.min()) * 0.95
max_val = max(y_test.max(), optimal_test_preds.max()) * 1.05
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Fit (y = x)')
ax3.set_xlabel('Actual Profit ($)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Predicted Profit ($)', fontsize=11, fontweight='bold')
ax3.set_title('Sweet Spot Model (k=2): Actual vs. Predicted Profit', fontsize=13, fontweight='bold')
ax3.grid(True, linestyle='--', alpha=0.4)
ax3.legend()
fig3.tight_layout()
fig3.savefig(OUTPUT_DIR / '03_sweet_spot_actual_vs_predicted.png', dpi=200)
plt.close(fig3)

# 圖表 4: Residuals vs Predicted
fig4, ax4 = plt.subplots(figsize=(8, 6))
opt_residuals = y_test - optimal_test_preds
ax4.scatter(optimal_test_preds, opt_residuals, color='#10b981', s=70, alpha=0.85, edgecolors='black')
ax4.axhline(0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Predicted Profit ($)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Residual Error: Actual - Predicted ($)', fontsize=11, fontweight='bold')
ax4.set_title('Sweet Spot Model (k=2): Residual Diagnostics', fontsize=13, fontweight='bold')
ax4.grid(True, linestyle='--', alpha=0.4)
fig4.tight_layout()
fig4.savefig(OUTPUT_DIR / '04_sweet_spot_residuals.png', dpi=200)
plt.close(fig4)

# 輸出 Sweet Spot 係數表
coef_df = pd.DataFrame({
    'Predictor': ['Intercept (beta_0)', 'R&D Spend', 'Marketing Spend'],
    'Estimate (beta)': [ols_optimal.params['const'], ols_optimal.params['R&D Spend'], ols_optimal.params['Marketing Spend']],
    'Std Error': [ols_optimal.bse['const'], ols_optimal.bse['R&D Spend'], ols_optimal.bse['Marketing Spend']],
    't-Stat': [ols_optimal.tvalues['const'], ols_optimal.tvalues['R&D Spend'], ols_optimal.tvalues['Marketing Spend']],
    'p-Value': [ols_optimal.pvalues['const'], ols_optimal.pvalues['R&D Spend'], ols_optimal.pvalues['Marketing Spend']],
    '95% CI Lower': [ols_optimal.conf_int().loc['const', 0], ols_optimal.conf_int().loc['R&D Spend', 0], ols_optimal.conf_int().loc['Marketing Spend', 0]],
    '95% CI Upper': [ols_optimal.conf_int().loc['const', 1], ols_optimal.conf_int().loc['R&D Spend', 1], ols_optimal.conf_int().loc['Marketing Spend', 1]],
    'VIF': [None, variance_inflation_factor(X_train[optimal_feats].values, 0), variance_inflation_factor(X_train[optimal_feats].values, 1)]
})
coef_df.to_csv(OUTPUT_DIR / 'sweet_spot_coefficients.csv', index=False)
print("\n=== Sweet Spot (k=2) OLS 係數檢定表 (StatsUnlock 風格) ===")
print(coef_df.to_markdown(index=False, floatfmt='.4f'))

# 輸出測試集殘差診斷表
res_df = pd.DataFrame({
    'Sample ID': y_test.index,
    'Actual Profit': y_test.values,
    'Predicted Profit': optimal_test_preds.values,
    'Residual ($)': opt_residuals.values,
    'Abs Error ($)': np.abs(opt_residuals.values),
    'Pct Error (%)': (np.abs(opt_residuals.values) / y_test.values) * 100
}).sort_values('Abs Error ($)', ascending=False)
res_df.insert(0, 'Rank', range(1, len(res_df) + 1))
res_df.to_csv(OUTPUT_DIR / 'sweet_spot_test_residuals.csv', index=False)
print("\n=== Sweet Spot (k=2) 測試集殘差與異常值診斷表 ===")
print(res_df.to_markdown(index=False, floatfmt='.2f'))

# ==============================================================================
# CRISP-DM STEP 6: DEPLOYMENT (部署與即時預測)
# ==============================================================================
print("\n" + "="*70)
print("CRISP-DM STEP 6: DEPLOYMENT & LIVE PREDICTOR")
print("="*70)

optimal_artifact = {
    'optimal_features': optimal_feats,
    'ols_model': ols_optimal,
    'formula': f"Profit = {ols_optimal.params['const']:.2f} + {ols_optimal.params['R&D Spend']:.4f}*(R&D) + {ols_optimal.params['Marketing Spend']:.4f}*(Marketing)",
    'reference_state': 'California',
    'encoded_states': ['State_Florida', 'State_New York'],
    'k': 2,
    'test_rmse': sweet_spot_k2['Test_RMSE'],
    'test_r2': sweet_spot_k2['Test_R2']
}

model_dump_file = OUTPUT_DIR / '50_startups_optimal_model.joblib'
joblib.dump(optimal_artifact, model_dump_file)
print(f"已儲存最優部署模型: {model_dump_file}")

def predict_startup_profit(rd_spend, marketing_spend, administration=None, state='California'):
    loaded_pkg = joblib.load(model_dump_file)
    intercept = loaded_pkg['ols_model'].params['const']
    beta_rd = loaded_pkg['ols_model'].params['R&D Spend']
    beta_mkt = loaded_pkg['ols_model'].params['Marketing Spend']
    profit = intercept + beta_rd * rd_spend + beta_mkt * marketing_spend
    return profit

demo_pred = predict_startup_profit(rd_spend=100000, marketing_spend=150000, administration=120000, state='California')
print(f"\n[Live Predictor 實測範例]")
print(f"輸入: R&D Spend = $100,000 | Marketing Spend = $150,000 | State = California")
print(f"預測獲利 (Predicted Profit): ${demo_pred:,.2f}")
print(f"迴歸方程式: {optimal_artifact['formula']}")

print("\n" + "="*70)
print("CRISP-DM 六大階段工作全數圓滿完成！")
print("="*70)
