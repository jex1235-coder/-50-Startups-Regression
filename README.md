# -50-Startups-Regression

✨ **50 Startups 多元線性迴歸與五大特徵選擇決策工作台 (CRISP-DM AIoT)**

🌐 **線上互動展示網址 (GitHub Pages)**:  
👉 **[https://jex1235-coder.github.io/-50-Startups-Regression/](https://jex1235-coder.github.io/-50-Startups-Regression/)**

---

## 📌 專案簡介與商業背景 (CRISP-DM Step 1)

本專案基於 **CRISP-DM 六大標準流程**，深入研究 50 家新創公司之研發支出（R&D Spend）、行政管理費（Administration）、行銷支出（Marketing Spend）以及註冊州別（State），探討其對公司年度淨利潤（Profit）之影響。

核心目標依循**統計簡約法則（Occam's Razor 奧坎剃刀原則）**：
1. **比較 5 種特徵選擇演算法**：Lasso (L1)、Forward SFS (貪婪向前)、RFE (遞迴剔除)、ElasticNet (彈性網)、Exhaustive Search (窮舉最佳子集)。
2. **評估特徵數量 $k=1 \sim 5$ 的邊際效益**：
   - **$k=2$ 全域最佳甜蜜點 (Optimal Sweet Spot ★)**：保留 `R&D Spend` + `Marketing Spend`，獲得最低測試集 RMSE (**$8,198.80**) 與最高 $R^2$ (**0.9474**)。
   - **$k=5$ 過度擬合懸崖 (Overfitting Cliff ⚠)**：加入全部變數導致測試集 RMSE 激增至 **$9,137.99**，Adjusted $R^2$ 崩跌至 **0.8531**，顯現小樣本過度擬合危害。

---

## 🚀 核心功能與互動工作台 (index.html)

- **Vue 3 + Bootstrap 5 主從儀表板 (Master-Detail Dashboard)**
- **五大演算法獨立特徵路徑軌跡圖 (Chart.js 4)**
- **動態焦點高亮**：點擊特徵數量 $k$ 時，圖表焦點紅點放大標記
- **5 步驟專家引導教學橫幅 (Step-by-Step Guided Assistant)**
- **即時獲利預測模擬器 (Real-time Profit Predictor)**
- **OLS 模型統計推論檢定表**（係數 $\beta$、標準誤、t 統計量、p-value、95% 信賴區間、VIF 多重共線性診斷）
- **獨立測試集殘差與異常值診斷分析**

---

## 📊 五大演算法跨特徵評估摘要

| 特徵數 $k$ | 選取特徵組合 (Lasso Benchmark) | Test RMSE ($) | Test $R^2$ | 模型診斷狀態 |
|---|---|---|---|---|
| **$k=1$** | `R&D Spend` | 8,274.87 | 0.9465 | 欠擬合 (Underfitting) |
| **$k=2$ ★** | `R&D Spend`, `Marketing Spend` | **8,198.80** | **0.9474** | **最佳甜蜜點 (Sweet Spot)** |
| **$k=3$** | `+ State_New York` | 8,309.06 | 0.9460 | 邊際效益反轉 ($p>0.79$) |
| **$k=4$** | `+ Administration` | 8,409.92 | 0.9447 | 自由度稀釋 ($p>0.60$) |
| **$k=5$ ⚠** | `+ State_Florida` (Full Model) | 9,137.99 | 0.9347 | **過度擬合懸崖 (Cliff)** |

---

## 🛠️ 本地執行方式

1. 建立並啟用 Python 虛擬環境：
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt  # 或安裝 pandas numpy scikit-learn statsmodels matplotlib joblib
```

2. 執行完整訓練與產圖程式：
```bash
python main.py
```

3. 開啟前端互動視覺化工作台：
```bash
# 直接用瀏覽器開啟 index.html
start index.html
```

---

## 📄 專案目錄結構

```text
├── index.html                   # 互動式儀表板 (支援 GitHub Pages 直接部署)
├── main.py                      # CRISP-DM 建模、特徵選擇、殘差診斷與數據匯出
├── 50_Startups_CRISP_DM.md      # 完整學術與商業分析報告 (含 APA 格式表格)
├── model_data.js                # 前端模型數據
├── model_data.json              # 結構化模型評估 JSON
├── 50_Startups.csv              # 原始新創資料集
├── 50_startups_optimal_model.joblib # 最優模型權重持久化檔
├── algo_1_lasso.png             # 各演算法獨立雙 Y 軸分析圖 (1~5)
├── algo_2_forward_sfs.png
├── algo_3_rfe.png
├── algo_4_elasticnet.png
├── algo_5_exhaustive.png
├── algo_6_all_methods_comparison.png # 5 種演算法橫向對比圖
└── README.md                    # 專案說明文件
```
