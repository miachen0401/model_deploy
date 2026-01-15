# Model Benchmarking System

Comprehensive benchmarking system for evaluating the deployed news categorization model against ground truth data from Supabase.

## 📁 Directory Structure

```
benchmarking/
├── __init__.py
├── claasification_rules.txt          # Classification categories and rules
├── data_fetcher/                      # Data fetching from Supabase
│   ├── __init__.py
│   └── supabase_client.py            # Supabase client implementation
├── model_inference.py                 # Model inference logic
├── evaluate_model.py                  # Main benchmarking script
├── visualize_results.py               # Visualization generation
├── results/                           # Evaluation results (CSV, JSON)
└── visualizations/                    # Generated plots (PNG)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install benchmarking dependencies
uv sync
```

### 2. Configure Environment

Create or update `.env` file with Supabase credentials:

```bash
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-api-key-here
```

### 3. Start vLLM Server

The benchmarking system requires your vLLM server to be running:

```bash
# Start the server
bash start_vllm_server.sh
```

### 4. Run Benchmarking

```bash
# Run full benchmarking pipeline (fetch data, inference, metrics)
uv run python -m benchmarking.evaluate_model --limit 100

# Or use Python module syntax
python -m benchmarking.evaluate_model --limit 100 --batch-size 3
```

### 5. Generate Visualizations

```bash
# Generate plots from metrics file
uv run python -m benchmarking.visualize_results \
    --metrics-file benchmarking/results/metrics_20260113_120000.json \
    --predictions-file benchmarking/results/predictions_20260113_120000.csv
```

## 📊 What Gets Measured

### Overall Metrics

- **Accuracy**: Overall classification accuracy
- **Precision** (Macro/Micro/Weighted): Precision across all classes
- **Recall** (Macro/Micro/Weighted): Recall across all classes
- **F1 Score** (Macro/Micro/Weighted): Harmonic mean of precision and recall
- **ROC AUC** (Macro): Area under ROC curve (one-vs-rest)
- **Confidence Statistics**: Mean and standard deviation of prediction confidences

### Per-Class Metrics

For each of the 13 categories:
- Precision
- Recall
- F1 Score
- Support (number of samples)
- ROC AUC Score

### Confusion Matrix

- Full confusion matrix showing predicted vs actual categories
- Both normalized (by row) and raw count versions

## 📈 Generated Visualizations

All visualizations are saved in `benchmarking/visualizations/`:

1. **metrics_summary.png** - Bar chart of overall performance metrics
2. **confusion_matrix.png** - Normalized confusion matrix heatmap
3. **confusion_matrix_raw.png** - Raw count confusion matrix
4. **per_class_metrics.png** - Per-class precision, recall, F1 scores and support
5. **roc_auc_comparison.png** - ROC AUC scores by category
6. **confidence_distribution.png** - Distribution of prediction confidences

## 🔧 Usage Examples

### Basic Benchmarking

```bash
# Evaluate on 100 samples with batch size 3
python -m benchmarking.evaluate_model --limit 100 --batch-size 3
```

### Custom API URL

```bash
# Use different vLLM server
python -m benchmarking.evaluate_model \
    --api-url http://remote-server:8000/v1 \
    --limit 200
```

### Skip Inference (Use Existing Predictions)

```bash
# Re-evaluate existing predictions
python -m benchmarking.evaluate_model \
    --skip-inference \
    --predictions-file benchmarking/results/predictions_20260113_120000.csv
```

### Fetch Data Only

```bash
# Just fetch and save data from Supabase
python -m benchmarking.data_fetcher.supabase_client
```

## 📝 Output Files

### Results Directory (`benchmarking/results/`)

- **`predictions_YYYYMMDD_HHMMSS.csv`** - Full predictions with ground truth
  - Columns: id, title, summary, category (true), symbol (true), predicted_category, predicted_symbol, prediction_confidence

- **`metrics_YYYYMMDD_HHMMSS.json`** - Complete metrics in JSON format
  - Overall metrics
  - Per-class metrics
  - Confusion matrix

### Visualizations Directory (`benchmarking/visualizations/`)

- PNG files for all generated plots (300 DPI)

## 🎯 Classification Categories

The model classifies news into 13 categories:

1. **MACRO_ECONOMY** - Economic indicators (CPI, GDP, PMI, etc.)
2. **CENTRAL_BANK_POLICY** - Central bank decisions and rate changes
3. **GEOPOLITICAL_EVENT** - Geopolitical news with specific actors
4. **INDUSTRY_REGULATION** - Regulatory/policy news
5. **CORPORATE_EARNINGS** - Company earnings and financials
6. **CORPORATE_ACTIONS** - M&A, splits, buybacks, etc.
7. **MANAGEMENT_CHANGE** - Executive changes
8. **INCIDENT_LEGAL** - Lawsuits, fines, investigations
9. **PRODUCT_TECH_UPDATE** - New products, R&D, technology
10. **BUSINESS_OPERATIONS** - Supply chain, contracts, partnerships
11. **ANALYST_OPINION** - Analyst ratings and commentary
12. **MARKET_SENTIMENT** - Investor sentiment and flows
13. **NON_FINANCIAL** - General commentary or unrelated news

## 🔍 Data Fetching from Supabase

### Supabase Table Schema

The system expects a `stock_news` table with:

**Required columns:**
- `title` (TEXT) - News article title
- `summary` (TEXT) - News article summary
- `category` (TEXT) - Ground truth category label
- `symbol` (TEXT) - Stock ticker symbols (optional)

**Optional columns:**
- `id` - Unique identifier
- `published_at` - Publication timestamp
- `source` - News source
- Any other metadata

### Data Quality Requirements

For accurate benchmarking:
- ✅ **Labeled data**: Must have `category` field populated
- ✅ **Valid categories**: Categories must match one of the 13 defined categories
- ✅ **Non-empty content**: Both `title` and `summary` should be present
- ⚠️  **Balanced distribution**: Ideally have samples from all categories

### Checking Data Quality

```python
from benchmarking.data_fetcher.supabase_client import SupabaseNewsClient

client = SupabaseNewsClient()

# Get category distribution
distribution = client.get_category_distribution()
print(distribution)

# Fetch labeled data for inspection
df = client.fetch_labeled_data(limit=10)
print(df.head())
```

## 📊 Interpreting Results

### Overall Performance

| Metric | Interpretation |
|--------|----------------|
| **Accuracy > 0.80** | Excellent - Model performs well across all categories |
| **Accuracy 0.60-0.80** | Good - Acceptable performance, room for improvement |
| **Accuracy < 0.60** | Poor - Needs investigation or retraining |

### Per-Class Performance

- **High Precision, Low Recall**: Model is conservative, misses some cases
- **Low Precision, High Recall**: Model is aggressive, many false positives
- **Low Precision & Recall**: Model struggles with this category

### ROC AUC Interpretation

| ROC AUC | Quality |
|---------|---------|
| **0.90-1.00** | Excellent |
| **0.80-0.90** | Good |
| **0.70-0.80** | Fair |
| **0.60-0.70** | Poor |
| **<0.60** | Very Poor (worse than random for <0.50) |

### Confusion Matrix Analysis

- **Diagonal values**: Correct predictions (should be high)
- **Off-diagonal patterns**: Common misclassifications
  - Example: If CORPORATE_EARNINGS often misclassified as ANALYST_OPINION, categories may be too similar

## 🐛 Troubleshooting

### Issue: "No labeled data found in Supabase"

**Solution**:
1. Check Supabase credentials in `.env`
2. Verify `stock_news` table exists
3. Ensure records have non-null `category` field:
   ```sql
   SELECT COUNT(*) FROM stock_news WHERE category IS NOT NULL;
   ```

### Issue: "vLLM server not available"

**Solution**:
1. Start the vLLM server: `bash start_vllm_server.sh`
2. Check server is running: `curl http://localhost:8000/v1/models`
3. Verify correct `--api-url` parameter

### Issue: Low accuracy on specific categories

**Solutions**:
1. Check category distribution - may need more training data
2. Review misclassifications in confusion matrix
3. Examine specific examples:
   ```python
   df = pd.read_csv("benchmarking/results/predictions_XXX.csv")
   errors = df[df['category'] != df['predicted_category']]
   print(errors[['title', 'category', 'predicted_category']])
   ```

### Issue: JSON parsing errors during inference

**Solution**:
- Model may not be outputting valid JSON
- Increase `max_tokens` parameter
- Check model prompt formatting
- Review raw responses in logs

## 🔄 Continuous Benchmarking

### Automated Benchmarking

Create a cron job or scheduled task:

```bash
#!/bin/bash
# benchmark_daily.sh

DATE=$(date +%Y%m%d)
LOG_FILE="benchmarking/logs/benchmark_${DATE}.log"

# Run benchmarking
python -m benchmarking.evaluate_model \
    --limit 500 \
    --batch-size 5 \
    >> "$LOG_FILE" 2>&1

# Generate visualizations
METRICS_FILE=$(ls -t benchmarking/results/metrics_*.json | head -1)
PRED_FILE=$(ls -t benchmarking/results/predictions_*.csv | head -1)

python -m benchmarking.visualize_results \
    --metrics-file "$METRICS_FILE" \
    --predictions-file "$PRED_FILE" \
    >> "$LOG_FILE" 2>&1

echo "Benchmarking complete: $(date)" >> "$LOG_FILE"
```

### Tracking Performance Over Time

```python
import json
import glob
import pandas as pd

# Collect all metrics files
metrics_files = sorted(glob.glob("benchmarking/results/metrics_*.json"))

history = []
for file in metrics_files:
    with open(file) as f:
        metrics = json.load(f)
        history.append({
            "date": file.split("_")[-2],
            "accuracy": metrics["overall"]["accuracy"],
            "f1_macro": metrics["overall"]["f1_macro"],
            "f1_weighted": metrics["overall"]["f1_weighted"],
        })

df = pd.DataFrame(history)
print(df)
```

## 📚 API Reference

### SupabaseNewsClient

```python
from benchmarking.data_fetcher.supabase_client import SupabaseNewsClient

client = SupabaseNewsClient()

# Fetch labeled data
df = client.fetch_labeled_data(
    limit=100,
    require_category=True,
    require_symbol=False
)

# Get category distribution
dist = client.get_category_distribution()

# Save to CSV
client.save_to_csv(df, "my_data.csv")
```

### NewsClassifier

```python
from benchmarking.model_inference import NewsClassifier

classifier = NewsClassifier(api_base_url="http://localhost:8000/v1")

# Classify single batch
news_items = [
    {"title": "...", "summary": "..."},
    {"title": "...", "summary": "..."},
]
results = classifier.classify_batch(news_items)

# Classify DataFrame
df_with_predictions = classifier.classify_dataframe(df, batch_size=3)
```

### ModelEvaluator

```python
from benchmarking.evaluate_model import ModelEvaluator

evaluator = ModelEvaluator()

# Calculate metrics
metrics = evaluator.calculate_metrics(y_true, y_pred, confidences)

# Save results
evaluator.save_metrics(metrics, "metrics.json")
evaluator.save_predictions(df, "predictions.csv")

# Print summary
evaluator.print_summary(metrics)
```

### BenchmarkVisualizer

```python
from benchmarking.visualize_results import BenchmarkVisualizer

visualizer = BenchmarkVisualizer(output_dir="my_plots/")

# Generate all plots
visualizer.generate_all_plots(metrics, predictions_df)

# Generate individual plots
visualizer.plot_confusion_matrix(cm, categories)
visualizer.plot_per_class_metrics(per_class_metrics)
visualizer.plot_roc_auc_comparison(per_class_metrics)
```

## 🤝 Contributing

When adding new metrics or visualizations:

1. Update `evaluate_model.py` for new metrics
2. Update `visualize_results.py` for new plots
3. Update this README with documentation
4. Test with sample data

## 📄 License

Same as parent project.
