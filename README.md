# Kaggle Santander Customer Transaction Prediction

This project is a GPU LightGBM solution for the Kaggle competition
`santander-customer-transaction-prediction`.

## Best Result

| File | Public AUC | Private AUC |
| --- | ---: | ---: |
| `outputs/submission_lgbm_magic_real_gpu.csv` | 0.90438 | 0.90202 |

The best submission is approximately top 4% on the downloaded public leaderboard snapshot, which is above the target top 20% threshold.

## Method

The final model focuses on competition-specific, target-free feature engineering:

- Detect real vs synthetic test rows using per-variable unique values in the test set.
- Build frequency features from `train + detected real test` rather than all test rows.
- Add per-variable uniqueness flags.
- Add rank features for each anonymous variable.
- Add row-level statistics including mean, std, min, max, sum, skew, and kurtosis.
- Train a 5-fold StratifiedKFold LightGBM model on GPU.

No target encoding or test labels are used.

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python src/train_lgbm_magic_real_gpu.py
```

The script requires a GPU-enabled LightGBM installation and explicitly uses the NVIDIA OpenCL device:

- `gpu_platform_id=1`
- `gpu_device_id=0`

If GPU LightGBM is unavailable, the script raises an error instead of silently falling back to CPU.

## Outputs

Important files:

- `outputs/submission_lgbm_magic_real_gpu.csv`
- `outputs/oof_lgbm_magic_real_gpu.csv`
- `outputs/pred_lgbm_magic_real_gpu.csv`
- `outputs/lgbm_magic_real_gpu_summary.json`
- `outputs/experiment_log.csv`
- `outputs/best_result_summary.csv`

Raw Kaggle data is not committed.
