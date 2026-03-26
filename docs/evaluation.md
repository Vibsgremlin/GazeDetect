# Evaluation Logs and Outputs

## What was evaluated
- Image preprocessing and dataset conversion
- CNN training pipeline
- Saved-model loading for inference
- Visual prediction display in `test.py`

## Observable outputs
```text
Processed N images. Training: X, Testing: Y
Model training complete. Model saved as gaze_model.h5
Prediction: BottomCenter
```

## Notes
- The repo does not include accuracy tables or confusion matrices.
- The label definitions in inference scripts and the model output dimension are currently inconsistent, so evaluation should be treated as prototype-level.
