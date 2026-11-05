# EE451 Project TODO

Memory-Bounded ANN Search on a Single GPU with Pipelined CPU Reranking

分工：**LL** = Lin Lin（GPU kernel）｜**XJ** = Xinyi（CPU / pipeline）｜**HS** = He Sui（資料 / baseline）

時間沿用分工表的 Day 1–14，開工日再填：Day 1 = ____

## Day 1–3：環境與 baseline

- [x] **全員** 確認 CARC 帳號和共用目錄（資料不要放 home）
- [x] **全員** 建立共用 Git repo

### XJ

- [x] 在登入節點確認可用的 GCC、CUDA module，寫好環境設定腳本
- [x] 寫 `.fvecs` / `.ivecs` 讀取
- [x] 寫串行 brute force，用 CPU 節點測試：前 100 個查詢在 SIFT1M 上 Recall@10 應為 1.0（距離相同的情況除外）
- [x] 用 CPU 節點跑完 SIFT1M 10,000 個查詢，確認程式正確
- [x] 提交一個短 sbatch 到 A40 節點，一次記錄 CPU 型號、實體核心數、NUMA、AVX2/AVX-512、GPU 拓撲，並跑 STREAM
- [x] 在 A40 節點上重跑串行 baseline，作為正式的加速比參考

### LL

- [x] 寫 GPU brute force（cuBLAS GEMM + top-k），結果要和串行版一致
- [ ] 產生 DEEP10M ground truth
- [ ] 實測 GPU 記憶體頻寬和 PCIe 頻寬

### HS

- [ ] 下載 SIFT1M、DEEP10M，轉成統一格式
- [ ] 安裝 Faiss、cuVS（用 conda 或 Apptainer）
  - cuVS 裝不上時先用 Faiss 完成腳本，之後再補
- [ ] Faiss 可行性測試：SQ8、SQ4、binary index 在 SIFT1M 上能否接近 Recall@10 0.95

### Day 3 全員檢查點

- [ ] 定好資料、候選 ID、結果三個格式，寫進 `docs/interfaces.md`
  - 候選 ID 是原始向量 ID，不是重排後的位置
  - 候選不足 C 個時的填充值
  - 結果按距離由小到大排列
- [ ] 確認資料目錄和 CSV 結果欄位

## Day 4–7：端到端跑通

### LL

- [x] 寫 k-means，建立 IVF（SIFT1M L = 1,024；DEEP10M L = 4,096）
- [x] 寫 8-bit 掃描 kernel
- [x] 寫 1-bit 掃描 kernel（隨機旋轉 + residual sign + XOR/popcount）
- [x] 用 CUB 完成第一版 top-C

### XJ

- [x] 寫 OpenMP + SIMD CPU 精排，先用假候選測試，結果要和 brute force 一致
- [x] 加 prefetch，可用參數開關
- [x] 寫 GPU zero-copy 精排（pinned host memory）
- [x] 寫預算 allocator：所有 device 配置都經過它，超過 B 就報錯
- [x] 用 `cudaMemGetInfo` 記錄基準值和搜尋峰值

### HS

- [ ] 寫正確性檢查腳本
  - 距離相對誤差 ≤ 10⁻⁴
  - GPU brute force 要和串行版、Faiss `IndexFlatL2`、SIFT1M 官方答案一致
  - 掃描距離要和 CPU 解碼結果一致
  - top-C 要和 CPU partial sort 一致
- [ ] 寫 Faiss、cuVS 掃參數腳本，關閉記憶體 pool

### Day 7 全員檢查點（最重要）

- [ ] SIFT1M 端到端跑通：GPU 掃描 → 候選 ID → CPU 精排 → Recall
- [ ] 正確性檢查全部通過
- [ ] CPU 精排和 GPU 精排的結果完全相同

未通過就先處理這裡，不要開始 4-bit、DEEP10M 或 pipeline。

## Day 8–10：完整功能與主要實驗

### LL

- [ ] 寫 4-bit 掃描 kernel
- [ ] 用 Nsight Compute 量掃描頻寬，計算占 696 GB/s 的比例

### XJ

- [ ] 寫 sequential 版和 pipeline 版（CUDA streams + 深度 1 的佇列）
- [ ] 確認 pipeline 和 sequential 回傳的結果完全相同
- [ ] 完成精排時間拆解（結果 3）：C 從 10 到 1,000
  - 候選 ID 傳輸
  - 讀取向量或 PCIe 傳輸
  - 距離計算
  - 結果回傳

### HS

- [ ] 在 SIFT1M、DEEP10M 上跑 1/16、1/8、1/4、1/2 預算 sweep
- [ ] 跑 Faiss、cuVS baseline，在 Recall@10 0.90、0.95、0.99 下比較
- [ ] 比較頻寬模型的預測和實測，包含 crossover

## Day 11–12：分析與重現

### LL

- [ ] 用 Nsight 分析和 cuVS 的差距
- [ ] 優化 top-C

### XJ

- [ ] 量 batch 1、16 的 P50、P99 latency
- [ ] 量 OpenMP scaling：1 個執行緒到全部核心
- [ ] 比較 pipeline 和 sequential 的 QPS，對照加速公式（結果 4）
- [ ] 計算相對串行 baseline 的端到端加速（結果 5）

### HS

- [ ] 完成一鍵重現腳本
- [ ] 畫結果 1、2 的所有圖

## Day 13–14：報告與提交

- [ ] **LL** 寫 kernel 和分析章節
- [ ] **XJ** 寫 pipeline 和精排章節
- [ ] **HS** 整合報告、README、limitations
- [ ] **全員** 按 README 從頭跑一次，確認其他人也能重現
- [ ] **全員** 核對每張圖都能由腳本產生，並記錄負面結果（例如沒有 crossover、輸給 cuVS）
- [ ] **全員** 準備 presentation

## 必交結果

- [ ] 結果 1：1/4/8-bit 的 Recall@10–QPS 曲線（有無精排、各預算），加上 Faiss、cuVS 和模型對照
- [ ] 結果 2：每種配置的預算 B 和實測峰值 device memory
- [ ] 結果 3：CPU 與 GPU 精排的時間拆解
- [ ] 結果 4：pipeline 與 sequential 的 QPS，以及 batch 1、16 的 P50、P99 latency
- [ ] 結果 5：掃描頻寬比例、OpenMP scaling、相對串行 baseline 的加速

## Optional（必交完成後才做）

- [ ] GIST1M
- [ ] Dynamic batching
- [ ] KV-cache retrieval
- [ ] Binary tensor-core 掃描

## 實驗前必查

- [ ] 在計算節點上編譯
- [ ] 記錄 commit、資料集、參數、節點名稱
- [ ] 停止 GPU 計時前先呼叫 `cudaDeviceSynchronize()`
- [ ] 設定 `OMP_PLACES=cores`、`OMP_PROC_BIND=close`，並確認 NUMA 綁定
- [ ] warm-up 後跑五次，取中位數
- [ ] 正式實驗固定同一種 GPU（A40），盡量使用 `--exclusive`
