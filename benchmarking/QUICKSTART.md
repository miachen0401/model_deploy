# Benchmarking Quick Start Guide

## 🎯 Goal

Evaluate your deployed news categorization model by comparing its predictions against ground truth labels in Supabase.

## ⚡ 3-Step Setup

### Step 1: Configure Supabase

Add to `.env` file:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-api-key-here
```

### Step 2: Start vLLM Server

```bash
bash start_vllm_server.sh
```

Wait for "Uvicorn running on..." message.

### Step 3: Run Benchmarking

```bash
# Run complete evaluation on 100 samples
uv run python -m benchmarking.evaluate_model --limit 100
```

## 📊 What You Get

After completion, check `benchmarking/results/` for:

1. **`predictions_YYYYMMDD_HHMMSS.csv`** - All predictions with ground truth
2. **`metrics_YYYYMMDD_HHMMSS.json`** - Performance metrics

### Generate Visualizations

```bash
# Replace with your actual file timestamps
uv run python -m benchmarking.visualize_results \
    --metrics-file benchmarking/results/metrics_20260113_231036.json \
    --predictions-file benchmarking/results/predictions_20260113_231036.csv
```

Check `benchmarking/visualizations/` for:
- Confusion matrix
- Per-class metrics
- ROC AUC comparison
- Confidence distributions

## 📈 Sample Output

```
MODEL EVALUATION SUMMARY
================================================================================

Overall Performance:
  Accuracy:           0.8523
  Precision (macro):  0.8312
  Recall (macro):     0.8145
  F1 Score (macro):   0.8228
  F1 Score (weighted):0.8467
  ROC AUC (macro):    0.9012

Per-Class Performance (Top 5 by F1):
Category                  Precision     Recall         F1    Support
---------------------------------------------------------------------------
CORPORATE_EARNINGS           0.9123     0.8945     0.9033         45
CENTRAL_BANK_POLICY          0.8756     0.9012     0.8882         32
ANALYST_OPINION              0.8523     0.8234     0.8376         28
CORPORATE_ACTIONS            0.8234     0.7912     0.8070         22
BUSINESS_OPERATIONS          0.7890     0.8012     0.7950         18
================================================================================
```

## 🔧 Common Commands

```bash
# Evaluate 200 samples with larger batches
uv run python -m benchmarking.evaluate_model --limit 200 --batch-size 5

# Use remote server
uv run python -m benchmarking.evaluate_model \
    --api-url http://remote-server:8000/v1 \
    --limit 100

# Re-evaluate existing predictions (skip inference)
uv run python -m benchmarking.evaluate_model \
    --skip-inference \
    --predictions-file benchmarking/results/predictions_20260113_231036.csv

# Just fetch data from Supabase
uv run python -m benchmarking.data_fetcher.supabase_client
```

## 🐛 Quick Troubleshooting

### ❌ "No labeled data found"
→ Check Supabase connection and ensure `category` field is populated

### ❌ "vLLM server not available"
→ Start server: `bash start_vllm_server.sh`

### ❌ "JSON parsing errors"
→ Model output may be malformed, check logs in evaluation output

## 📚 Full Documentation

See [benchmarking/README.md](README.md) for:
- Detailed API reference
- Metric interpretations
- Advanced usage
- Troubleshooting guide
