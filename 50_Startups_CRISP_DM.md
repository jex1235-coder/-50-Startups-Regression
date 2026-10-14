# 50 Startups 多元線性迴歸：CRISP-DM 完整實作與五大特徵選擇演算法

> **主題**：利用 Multiple Linear Regression 預測新創公司獲利（Profit）與特徵選擇簡約性分析  
> **參考網站**：[50 Startups: Multiple Linear Regression & Feature Selection Explorer](https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/)  
> **執行環境**：Python 3.12 / scikit-learn / statsmodels / pandas / matplotlib  
> **核心方法**：CRISP-DM 六大階段、One-Hot Encoding、五大演算法（Lasso, Forward SFS, RFE, ElasticNet, Exhaustive Search）跨五個特徵（k=1..5）評估、奧坎剃刀簡約法則（Parsimony）、Sweet Spot 鎖定與 Overfitting Cliff 診斷。

---

## 目錄

1. [研究目標與商業理解 (CRISP-DM Step 1)](#1-研究目標與商業理解-crisp-dm-step-1)
2. [資料理解與 EDA (CRISP-DM Step 2)](#2-資料理解與-eda-crisp-dm-step-2)
3. [資料準備與虛擬變數編碼 (CRISP-DM Step 3)](#3-資料準備與虛擬變數編碼-crisp-dm-step-3)
4. [五大演算法建模與特徵選擇 (CRISP-DM Step 4)](#4-五大演算法建模與特徵選擇-crisp-dm-step-4)
5. [模型評估與偏誤-變異權衡分析 (CRISP-DM Step 5)](#5-模型評估與偏誤-變異權衡分析-crisp-dm-step-5)
6. [模型部署與即時預測器 (CRISP-DM Step 6)](#6-模型部署與即時預測器-crisp-dm-step-6)
7. [五大演算法 x 1~5 特徵完整實驗數據表](#7-五大演算法-x-15-特徵完整實驗數據表)
8. [最優模型 (k=2) 詳細統計推論與殘差診斷](#8-最優模型-k2-詳細統計推論與殘差診斷)
9. [完整可執行程式 (main.py)](#9-完整可執行程式-mainpy)
10. [資料來源與致謝](#10-資料來源與致謝)

---

## 1. 研究目標與商業理解 (CRISP-DM Step 1)

### 1.1 商業問題背景
投資人與新創團隊在擬定預算配置時，最關鍵的問題在於：
1. **哪些營運支出（研發 R&D、行政 Administration、行銷 Marketing）對公司利潤（Profit）最具統計預測力？**
2. **公司所在地（State）是否帶來顯著利潤溢價？**
3. **特徵數量是否「越多越好」？** 增加額外變數究竟是提升泛化能力，還是引入過度擬合（Overfitting）的雜訊？

### 1.2 奧坎剃刀原則與簡約性分析（Parsimony）
本專案依循統計學與機器學習中的**奧坎剃刀原則（Occam's Razor）**：若多個模型表現相近，應優先選擇結構最精簡、參數最少者。
- **Optimal Sweet Spot ($k=2$)**：僅保留 `R&D Spend` 與 `Marketing Spend`，達到最低測試集 RMSE（**$8,198.80**）與最高 $R^2$（**0.9474**）。
- **Overfitting Cliff ($k=5$)**：包含全部 5 個特徵的模型，因過度參數化導致測試集 RMSE 激增至 **$9,137.99**（誤差增加 $939.19），Adjusted $R^2$ 從 0.9324 崩跌至 **0.8531**。

---

## 2. 資料理解與 EDA (CRISP-DM Step 2)

### 2.1 資料結構
資料集包含 50 家新創公司的營運投入與利潤資料：

| 欄位名稱 | 角色 | 原始型態 | 說明 |
|---|---|---|---|
| `R&D Spend` | 預測變數 $X_1$ | 數值型 | 研發經費投入 |
| `Administration` | 預測變數 $X_2$ | 數值型 | 行政與人事費用 |
| `Marketing Spend` | 預測變數 $X_3$ | 數值型 | 市場推廣與廣告支出 |
| `State` | 預測變數 $X_4$ | 類別型 | 註冊州別（New York: 17, California: 17, Florida: 16） |
| `Profit` | 目標變數 $y$ | 數值型 | 公司年度稅後利潤 |

### 2.2 相關係數熱圖 (EDA)
各數值特徵與 Profit 之 Pearson 相關係數矩陣：
- **`R&D Spend` 與 `Profit`**：高達 **0.97**（極強正相關，最強單一預測因子）
- **`Marketing Spend` 與 `Profit`**：**0.75**（中強正相關，提供補充性解釋力）
- **`Administration` 與 `Profit`**：僅 **0.20**（弱相關，對利潤預測貢獻微弱）

圖表輸出存檔：[`05_correlation_heatmap.png`](file:///D:/Doinclass/05_correlation_heatmap.png)

---

## 3. 資料準備與虛擬變數編碼 (CRISP-DM Step 3)

### 3.1 虛擬變數處理 (Dummy Variable Trap 預防)
類別變數 `State` 包含 3 個類別：California、Florida、New York。
採用 `OneHotEncoder(drop='first')`，以 **California** 作為基準參考類別（Reference Category）：
- $D_{Florida} = 1$ 表示位於佛羅里達州；否則為 0。
- $D_{NewYork} = 1$ 表示位於紐約州；否則為 0。
- 兩者皆為 0 時，即代表基準類別加州。

因此，模型總共有 **5 個轉換後的候選特徵**：
1. `R&D Spend`
2. `Administration`
3. `Marketing Spend`
4. `State_Florida`
5. `State_New York`

### 3.2 資料切分
- **訓練集 (Training Set, 80%)**：40 筆，用於擬合迴歸係數與特徵挑選。
- **獨立測試集 (Test Set, 20%)**：10 筆，完全封存，僅供最終泛化能力客觀檢驗。
- 固定隨機種子 `random_state = 0`。

---

## 4. 五大演算法建模與特徵選擇 (CRISP-DM Step 4)

本研究橫向比對參考網站中的 **5 大特徵選擇演算法**：

1. **Lasso (L1 Regularization)**：透過 L1 懲罰項壓縮不重要係數至 0，觀察係數隨懲罰參數路徑逐一非零之入選順序。
2. **Forward SFS (Sequential Forward Selection)**：貪婪演算法（Greedy），由空集合出發，逐輪挑選加入後使誤差最低之特徵。
3. **RFE (Recursive Feature Elimination)**：反向遞迴特徵消除（Backward），由完整模型出發，逐步剃除係數權重最小的特徵。
4. **ElasticNet**：結合 L1（Lasso 稀疏性）與 L2（Ridge 穩定性）的正則化特徵挑選機制。
5. **Exhaustive Search (Best Subset Selection)**：窮舉所有 $\binom{5}{k}$ 特徵子集合，保證在給定特徵數 $k$ 下取得全域最優解。

---

## 5. 模型評估與偏誤-變異權衡分析 (CRISP-DM Step 5)

### 5.1 評估指標
- **RMSE（均方根誤差）**：$\sqrt{\frac{1}{n}\sum(y_i - \hat y_i)^2}$，越低越好，對大誤差敏感。
- **MAE（平均絕對誤差）**：$\frac{1}{n}\sum |y_i - \hat y_i|$，越低越好。
- **$R^2$（決定係數）**：$1 - \frac{SS_{res}}{SS_{tot}}$，模型解釋變異量百分比。
- **Adjusted $R^2$（調整後決定係數）**：$1 - (1 - R^2)\frac{n-1}{n-p-1}$，嚴厲懲罰冗餘參數。

### 5.2 偏誤-變異權衡（Bias-Variance Tradeoff）
- **當 $k=1$（僅 R&D Spend）**：RMSE = $8,274.87，模型存在輕微欠擬合（Underfitting）。
- **當 $k=2$（R&D + Marketing）★ Sweet Spot**：RMSE 下降至全域最低 **$8,198.80**，$R^2$ 達到最高 **0.9474**，達最佳偏誤與變異平衡。
- **當 $k \ge 3$**：加入 Administration 與州別虛擬變數後，測試集 RMSE 開始逆勢反彈。
- **當 $k=5$ ⚠ Overfitting Cliff**：全特徵模型在訓練集擬合良好，但在獨立測試集上 RMSE 飆升至 **$9,137.99**，證明額外參數已淪為過度擬合的雜訊。

圖表輸出存檔：
- [`01_rmse_comparison_all_algos.png`](file:///D:/Doinclass/01_rmse_comparison_all_algos.png)：五大演算法 RMSE 隨 $k$ 變化趨勢
- [`02_r2_comparison_all_algos.png`](file:///D:/Doinclass/02_r2_comparison_all_algos.png)：五大演算法 $R^2$ 隨 $k$ 變化趨勢
- [`03_sweet_spot_actual_vs_predicted.png`](file:///D:/Doinclass/03_sweet_spot_actual_vs_predicted.png)：Sweet Spot 真實 vs. 預測利潤
- [`04_sweet_spot_residuals.png`](file:///D:/Doinclass/04_sweet_spot_residuals.png)：Sweet Spot 殘差分佈診斷

---

## 6. 模型部署與即時預測器 (CRISP-DM Step 6)

### 6.1 最優迴歸方程式
最優模型僅需 2 個自變數，方程式為：
\[
\text{Profit} = 46,346.34 + 0.7886 \times (\text{R\&D Spend}) + 0.0326 \times (\text{Marketing Spend})
\]

- **截距 $\beta_0 = 46,346.34$**：基準底定利潤。
- **$\beta_{RD} = 0.7886$ ($p < 0.001$)**：極其顯著，研發投入每增加 $1，預期利潤增加約 $0.79。
- **$\beta_{Marketing} = 0.0326$ ($p = 0.071$)**：在 10% 顯著水準下顯著，行銷投入具正向帶動效果。
- **VIF = 2.10**：遠低於 5 的共線性門檻，模型不存在共線性威脅。

### 6.2 部署產物
- **模型儲存**：打包為 [`50_startups_optimal_model.joblib`](file:///D:/Doinclass/50_startups_optimal_model.joblib)。
- **網頁資料對接**：產出 [`model_data.json`](file:///D:/Doinclass/model_data.json) 與 [`model_data.js`](file:///D:/Doinclass/model_data.js)，可直接驅動 [AIoT-DIC2-MultipleLR 前端視覺化儀表板](https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/)。

---

## 7. 五大演算法 x 1~5 特徵完整實驗數據表

以下為 `main.py` 實際執行產出之完整數據（亦儲存於 [`algorithm_comparison_summary.csv`](file:///D:/Doinclass/algorithm_comparison_summary.csv)）：

| 演算法 | k | 入選特徵 (Selected Features) | Test RMSE ($) | Test MAE ($) | Test $R^2$ | Adj $R^2$ | F-Stat | F p-value |
|:---|---:|:---|---:|---:|---:|---:|---:|---:|
| **Lasso** | 1 | `R&D Spend` | 8,274.87 | 6,772.45 | 0.9465 | 0.9398 | 141.42 | 2.30e-06 |
| **Lasso** | **2 ★** | **`R&D Spend`, `Marketing Spend`** | **8,198.80** | **6,886.59** | **0.9474** | **0.9324** | **63.09** | **3.33e-05** |
| **Lasso** | 3 | `R&D Spend`, `Marketing Spend`, `State_New York` | 8,309.06 | 7,028.44 | 0.9460 | 0.9190 | 35.05 | 3.37e-04 |
| **Lasso** | 4 | `R&D Spend`, `Marketing Spend`, `State_New York`, `State_Florida` | 8,409.92 | 7,109.16 | 0.9447 | 0.9005 | 21.35 | 2.42e-03 |
| **Lasso** | **5 ⚠** | `R&D Spend`, `Marketing Spend`, `State_New York`, `State_Florida`, `Administration` | **9,137.99** | **7,514.29** | **0.9347** | **0.8531** | **11.45** | **1.74e-02** |
|---|---|---|---|---|---|---|---|---|
| **Forward SFS** | 1 | `R&D Spend` | 8,274.87 | 6,772.45 | 0.9465 | 0.9398 | 141.42 | 2.30e-06 |
| **Forward SFS** | **2 ★** | **`R&D Spend`, `Marketing Spend`** | **8,198.80** | **6,886.59** | **0.9474** | **0.9324** | **63.09** | **3.33e-05** |
| **Forward SFS** | 3 | `R&D Spend`, `Administration`, `Marketing Spend` | 8,803.78 | 7,320.44 | 0.9394 | 0.9091 | 31.00 | 4.76e-04 |
| **Forward SFS** | 4 | `R&D Spend`, `Administration`, `Marketing Spend`, `State_New York` | 8,997.20 | 7,474.74 | 0.9367 | 0.8861 | 18.50 | 3.37e-03 |
| **Forward SFS** | **5 ⚠** | `R&D Spend`, `Administration`, `Marketing Spend`, `State_Florida`, `State_New York` | **9,137.99** | **7,514.29** | **0.9347** | **0.8531** | **11.45** | **1.74e-02** |
|---|---|---|---|---|---|---|---|---|
| **RFE** | 1 | `R&D Spend` | 8,274.87 | 6,772.45 | 0.9465 | 0.9398 | 141.42 | 2.30e-06 |
| **RFE** | **2 ★** | **`R&D Spend`, `Marketing Spend`** | **8,198.80** | **6,886.59** | **0.9474** | **0.9324** | **63.09** | **3.33e-05** |
| **RFE** | 3 | `R&D Spend`, `Administration`, `Marketing Spend` | 8,803.78 | 7,320.44 | 0.9394 | 0.9091 | 31.00 | 4.76e-04 |
| **RFE** | 4 | `R&D Spend`, `Administration`, `Marketing Spend`, `State_Florida` | 9,068.54 | 7,431.47 | 0.9357 | 0.8843 | 18.19 | 3.50e-03 |
| **RFE** | **5 ⚠** | `R&D Spend`, `Administration`, `Marketing Spend`, `State_Florida`, `State_New York` | **9,137.99** | **7,514.29** | **0.9347** | **0.8531** | **11.45** | **1.74e-02** |
|---|---|---|---|---|---|---|---|---|
| **ElasticNet** | 1 | `R&D Spend` | 8,274.87 | 6,772.45 | 0.9465 | 0.9398 | 141.42 | 2.30e-06 |
| **ElasticNet** | **2 ★** | **`R&D Spend`, `Marketing Spend`** | **8,198.80** | **6,886.59** | **0.9474** | **0.9324** | **63.09** | **3.33e-05** |
| **ElasticNet** | 3 | `R&D Spend`, `Marketing Spend`, `Administration` | 8,803.78 | 7,320.44 | 0.9394 | 0.9091 | 31.00 | 4.76e-04 |
| **ElasticNet** | 4 | `R&D Spend`, `Marketing Spend`, `Administration`, `State_New York` | 8,997.20 | 7,474.74 | 0.9367 | 0.8861 | 18.50 | 3.37e-03 |
| **ElasticNet** | **5 ⚠** | `R&D Spend`, `Marketing Spend`, `Administration`, `State_New York`, `State_Florida` | **9,137.99** | **7,514.29** | **0.9347** | **0.8531** | **11.45** | **1.74e-02** |
|---|---|---|---|---|---|---|---|---|
| **Exhaustive** | 1 | `R&D Spend` | 8,274.87 | 6,772.45 | 0.9465 | 0.9398 | 141.42 | 2.30e-06 |
| **Exhaustive** | 2 | `R&D Spend`, `Marketing Spend` | 8,198.80 | 6,886.59 | 0.9474 | 0.9324 | 63.09 | 3.33e-05 |
| **Exhaustive** | 3 | `R&D Spend`, `Administration`, `State_Florida` | 8,171.65 | 6,730.79 | 0.9478 | 0.9217 | 36.30 | 3.05e-04 |
| **Exhaustive** | 4 | `R&D Spend`, `Administration`, `State_Florida`, `State_New York` | 8,178.37 | 6,751.89 | 0.9477 | 0.9059 | 22.65 | 2.11e-03 |
| **Exhaustive** | **5 ⚠** | `R&D Spend`, `Administration`, `Marketing Spend`, `State_Florida`, `State_New York` | **9,137.99** | **7,514.29** | **0.9347** | **0.8531** | **11.45** | **1.74e-02** |

---

## 8. 最優模型 (k=2) 詳細統計推論與殘差診斷

### 8.1 係數統計推論表（APA / StatsUnlock 風格）
由 `main.py` 產出至 [`sweet_spot_coefficients.csv`](file:///D:/Doinclass/sweet_spot_coefficients.csv)：

| 變數 (Predictor) | 係數估計值 ($\beta$) | 標準誤 (Std Error) | $t$ 統計量 | $p$ 值 | 95% 信賴區間下限 | 95% 信賴區間上限 | VIF |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **Intercept ($\beta_0$)** | 46,346.3388 | 2,971.2365 | 15.5983 | < 0.0001 | 40,326.0418 | 52,366.6357 | — |
| **`R&D Spend`** | 0.7886 | 0.0468 | 16.8460 | < 0.0001 | 0.6938 | 0.8835 | 2.10 |
| **`Marketing Spend`** | 0.0326 | 0.0175 | 1.8600 | 0.0708 | -0.0029 | 0.0681 | 2.10 |

### 8.2 測試集 10 家新創公司殘差與異常值診斷
由 `main.py` 產出至 [`sweet_spot_test_residuals.csv`](file:///D:/Doinclass/sweet_spot_test_residuals.csv)，依絕對誤差排序：

| 排名 | 原始樣本編號 | 實際獲利 ($) | 預測獲利 ($) | 殘差 ($) | 絕對誤差 ($) | 百分比誤差 (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 38 | 81,229.06 | 68,335.08 | 12,893.98 | 12,893.98 | 15.87% |
| 2 | 10 | 146,121.95 | 134,182.15 | 11,939.80 | 11,939.80 | 8.17% |
| 3 | 2 | 191,050.39 | 180,642.25 | 10,408.14 | 10,408.14 | 5.45% |
| 4 | 11 | 144,259.40 | 133,873.92 | 10,385.48 | 10,385.48 | 7.20% |
| 5 | 27 | 105,008.31 | 114,717.25 | -9,708.94 | 9,708.94 | 9.25% |
| 6 | 22 | 110,352.25 | 114,580.92 | -4,228.67 | 4,228.67 | 3.83% |
| 7 | 4 | 166,187.94 | 170,343.32 | -4,155.38 | 4,155.38 | 2.50% |
| 8 | 41 | 77,798.83 | 73,701.11 | 4,097.72 | 4,097.72 | 5.27% |
| 9 | 28 | 103,282.38 | 102,284.65 | 997.73 | 997.73 | 0.97% |
| 10 | 31 | 97,483.56 | 97,433.46 | 50.10 | 50.10 | 0.05% |

---

## 9. 演算法獨立圖表與互動式網頁

### 9.1 五大演算法獨立特徵軌跡圖（各演算法一張）
- **Lasso (L1 Regularization)**：[`algo_1_lasso.png`](file:///D:/Doinclass/algo_1_lasso.png)
- **Forward SFS (Greedy)**：[`algo_2_forward_sfs.png`](file:///D:/Doinclass/algo_2_forward_sfs.png)
- **RFE (Backward)**：[`algo_3_rfe.png`](file:///D:/Doinclass/algo_3_rfe.png)
- **ElasticNet (L1 + L2)**：[`algo_4_elasticnet.png`](file:///D:/Doinclass/algo_4_elasticnet.png)
- **Exhaustive Search (Best Subset)**：[`algo_5_exhaustive.png`](file:///D:/Doinclass/algo_5_exhaustive.png)
- **全演算法橫向疊加圖**：[`algo_6_all_methods_comparison.png`](file:///D:/Doinclass/algo_6_all_methods_comparison.png)

### 9.2 互動式網頁探索器與專家逐步導覽 (index.html)
專案根目錄中的 [`index.html`](file:///D:/Doinclass/index.html) 為純前端可獨立開啟之互動式網頁（免後端伺服器，雙擊或於瀏覽器中直接開啟），提供：
1. **五大演算法切換頁籤**：Lasso、Forward SFS、RFE、ElasticNet、Exhaustive Search 及橫向對比。
2. **特徵數量選擇按鈕 ($k=1 \dots 5$)**：動態更新 RMSE、R²、Adjusted R²、MAE、F-Stat。
3. **🎓 專家引導式探索教學（Step-by-Step Guided Tour）**：提供步驟 1 至步驟 5 的漸進導覽，深入解析欠擬合 ($k=1$)、全域最佳甜蜜點 ($k=2$ Sweet Spot)、邊際效益反轉 ($k=3 \sim 4$) 到過度擬合懸崖 ($k=5$ Overfitting Cliff)。
4. **即時利潤計算器 (Live Profit Predictor)**：輸入研發、行政、行銷預算與州別，依選中模型係數即時計算利潤。
5. **獨立圖表畫廊**：內嵌 5 大演算法各別圖表展示。

---

## 10. 完整可執行程式 (main.py)

程式碼已完整部署於專案根目錄之 [`main.py`](file:///D:/Doinclass/main.py)，可直接以 `python main.py` 執行，完成所有演算法運算、統計檢定、檔案匯出與即時預測演示。

---

## 11. 資料來源與致謝

- **互動展示與演算法參照**：[50 Startups: Multiple Linear Regression & Feature Selection Explorer](https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/)
- **設計風格參照**：[StatsUnlock Multiple Linear Regression Calculator](https://statsunlock.com/multiple-linear-regression-calculator/)
- **資料集來源**：[Avik Jain 100-Days-Of-ML / 50_Startups.csv](https://raw.githubusercontent.com/harimittapalli/Mulitple-Linear-Reggression/master/50_Startups.csv)
- **方法論依據**：CRISP-DM（Cross-Industry Standard Process for Data Mining）六大標準流程。
