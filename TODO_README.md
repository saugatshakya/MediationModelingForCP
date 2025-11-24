# Project TODO Checklist & README

This `TODO_README.md` is the single-source-of-truth for next steps, experiments, and reproducibility for the CP2025 Project. Based on `main.ipynb`, much of the data exploration and regression modeling is complete. This is now a checklist of what's done and what needs to be done.

> Note: Removed classification aspects (OVR/OVO, classifiers) as the problem is regression-based for predicting exam scores.

## Quick Contract

- **Inputs**: CSV datasets in `data/` (e.g., `caffeine_intake.csv`, `sleep_efficiency.csv`, etc.).
- **Output**: Trained regression models, evaluation metrics, experiment logs, plots, and an interactive inference frontend.
- **Success Criteria**: Reproducible pipeline for exam score prediction, ablation studies, and a user-friendly frontend for personalized recommendations.
- **Error Modes**: Missing data, invalid inputs in frontend, model overfitting to synthetic data.

## What's Already Done (from `main.ipynb`)

- [x] Load and explore datasets (caffeine, coffee health, sleep efficiency, student habits).
- [x] Generate synthetic data: caffeine intake, sleep quality, stress proxy, focus proxy, expand to 10k rows.
- [x] Visualizations: scatter plots, heatmaps, histograms, pairplots, boxplots, barplots.
- [x] Correlations and statistical tests (Pearson r with exam_score).
- [x] Regression models: RandomForest for stress, focus, and exam_score prediction (R² ~0.98 for exam scores).
- [x] Feature importance analysis (RandomForest).
- [x] Model comparisons: Linear, Ridge, Lasso, RandomForest, GradientBoosting, SVR, KNN (RandomForest best).
- [x] Basic predictions (e.g., study hours from exam scores).

## What Needs to Be Done (Checklist)

- [ ] **Build Configurable Regression Pipeline**: Implement scikit-learn Pipeline with preprocessing (scalers: Standard/MinMax/Robust; encoders: OneHot/Ordinal; imputation) + regressor (RandomForest, GradientBoosting, etc.). Test on small data subset.
- [ ] **Ablation Study Design**: Define experiments to measure impact of scalers, encoders, models, and feature selections. Baseline: StandardScaler + OneHot + RandomForest. Ablations: no scaler, different encoders, model swaps.
- [ ] **Experiment Automation and Logging**: Create `run_experiment.py` script to run ablation experiments, log metrics (MSE, R², training time), save models/artifacts to `artifacts/` and `experiments/` folders.
- [ ] **Reproducibility Setup**: Add `requirements.txt` with pinned versions (pandas, scikit-learn, matplotlib, etc.), fix random seeds (numpy, sklearn), update `main.ipynb` and `test.ipynb` for reproducible runs.
- [ ] **Inference Frontend Development**: Build an interactive web UI (e.g., using Streamlit or Flask + HTML/JS) with the following features:
  - **Drink Selector**: User chooses number and types of drinks (use the detailed list from `test.ipynb`: Espresso (63mg/30ml), Americano (95mg/240ml), Cappuccino (75mg/240ml), Latte (75mg/240ml), Drip Coffee (130mg/240ml), Instant Coffee (100mg/240ml), Thai Iced Coffee (70mg/240ml), Black Tea (55mg/240ml), Green Tea (32mg/240ml), Oolong Tea (40mg/240ml), White Tea (22mg/240ml), Matcha (70mg/240ml), Thai Iced Tea (45mg/240ml), Red Bull (80mg/250ml), Monster (160mg/500ml), M-150 (80mg/250ml), Carabao (106mg/330ml), 5-hour Energy (200mg/57ml), etc.) throughout the day to calculate total caffeine intake. Allow custom servings and sum caffeine.
  - **Feature Inputs**: Sliders/inputs for relevant features from `main.ipynb` (sleep hours, sleep quality, study hours, social media hours, netflix hours, exercise frequency, mental health rating, attendance percentage, diet quality, internet quality, part-time job, age, gender, parental education).
  - **Interactive Timetable UI**: A drag-and-drop timetable interface (like a calendar or hourly grid) where users can add/edit activities (study, sleep, social media, exercise, etc.) and scale their durations to exactly match 24 hours in a day. Activities map to features (e.g., dragging "study" block increases study_hours_per_day).
  - **Exam Score Prediction**: Based on all inputs (caffeine, timetable features), use the trained model to predict exam score. Display prediction with confidence interval if possible.
  - **Optimization for Target Score**: If user inputs desired exam score, use optimization (e.g., via scipy.optimize or simple grid search) to suggest adjustments to the timetable (e.g., "increase study by 2 hours, reduce social media by 1 hour") to achieve the target. Show before/after predictions.
  - **UI Details**: Responsive design, real-time updates, save/load profiles, tooltips explaining features, and export recommendations as PDF or image.

## Pipeline Components (Recommended)

1. **Data Ingestion**: Robust CSV loader with schema checks.
2. **Train/Validation Split**: Use train_test_split or cross-validation.
3. **Preprocessing**:
   - Numerical: Scalers (`StandardScaler`, `MinMaxScaler`, `RobustScaler`).
   - Categorical: Encoders (`OneHotEncoder`, `OrdinalEncoder`).
   - Imputation: `SimpleImputer` (median for numerics, most_frequent for categoricals).
4. **Regressor Choices**: RandomForest, GradientBoosting, LinearRegression, etc.
5. **Feature Selection**: Optional (e.g., SelectKBest).

## Ablation Study Design

- **Baseline**: StandardScaler + OneHot + RandomForestRegressor.
- **Ablations**:
  - Scalers: None vs Standard vs MinMax vs Robust.
  - Encoders: OneHot vs Ordinal.
  - Models: RandomForest vs GradientBoosting vs LinearRegression.
  - Feature Selection: With/without top-k features.
- Evaluate on same CV folds, log MSE, R², time.

## Evaluation Metrics

- Primary: MSE, R².
- Secondary: MAE, training/inference time.
- Plots: Actual vs Predicted, residual plots.

## Logging & Artifacts

- Save models (joblib) to `artifacts/`.
- Metrics to `experiments/` as JSON/CSV.
- Frontend: Deploy locally or via Streamlit Cloud.

## Reproducibility

- `requirements.txt`: pandas==2.0.0, scikit-learn==1.3.0, matplotlib, seaborn, streamlit (for frontend).
- Seeds: np.random.seed(42), etc.
- Scripts: `run_experiment.py`, `evaluate.py`.

## Notebooks & Scripts

- `main.ipynb`: Exploration and pipeline demo.
- `test.ipynb`: Unit tests for pipeline.
- `run_experiment.py`: Run ablations.
- `evaluate.py`: Load and report.
- Frontend: `app.py` (Streamlit) or `frontend/` folder.

## Quick "Try It" (Developer Notes)

- Test pipeline: Run minimal pipeline in `main.ipynb` on small data.
- Frontend prototype: Start with Streamlit app for drink selector and basic prediction.

## Next Steps (Prioritized)

1. Implement regression pipeline in `main.ipynb`.
2. Build ablation runner script.
3. Add reproducibility files.
4. Develop inference frontend incrementally (start with inputs, then timetable, then optimization).

---

Updated as checklist based on `main.ipynb` progress. Frontend is the key new addition for user interaction.
