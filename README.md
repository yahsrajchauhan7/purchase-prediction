Predicting E-Commerce Customer Purchase Behaviour

A machine learning pipeline that predicts whether an online shopping session will end in a purchase, using only behavioural data captured during the session itself — no personal or payment information involved.

Run in: Google Colab (paths reference /content/, see Setup below)

Why this project

Online retailers only know if a session converted after it ends. If a model can flag a likely non-converting session early — based on how a visitor is browsing, not who they are — a business could intervene in real time: a discount pop-up, a chat prompt, a simplified checkout. This project builds and compares several models for that early-warning signal, then wraps the best one in a small interactive tool that outputs a purchase probability from live inputs.

Dataset

Online Shoppers Purchasing Intention Dataset (UCI Machine Learning Repository) — 12,330 real e-commerce sessions, 18 columns. Each row is one browsing session; the target column, Revenue, is True if the session ended in a purchase.

Key columns:

Administrative, Informational, ProductRelated — page counts and time spent per page category during the session
BounceRates, ExitRates — standard web analytics metrics for the pages visited
PageValues — Google Analytics' page value metric, historically the single strongest predictor in this dataset
Month, SpecialDay, Weekend — timing context
VisitorType, OperatingSystems, Browser, Region, TrafficType — visitor/session metadata

Class imbalance: only 15.47% of sessions end in a purchase (1,908 of 12,330). This is the central challenge of the project — a model that always predicts "no purchase" would already be 84.5% "accurate" while being useless, which is why accuracy alone is not used to judge the models (see Evaluation Metrics below).

Pipeline
Load & inspect — check shape, column types, missing values (there are none in this dataset) and confirm the target distribution.
Train/test split — 80/20, stratified on the target so both splits keep the same 15.47% purchase rate. Stratification matters here specifically because the minority class is small enough that a plain random split could easily over- or under-represent it.
Preprocessing pipeline (scikit-learn ColumnTransformer + Pipeline, fit only on the training set to avoid leakage):
Numeric columns → median imputation, then standard scaling
Categorical columns (Month, VisitorType, Weekend) → most frequent imputation, then one-hot encoding
Model training — five classifiers trained on identical train/test splits for a fair comparison:
Logistic Regression, Decision Tree, Random Forest — all with class_weight="balanced", which reweights the loss so the rare "purchase" class isn't ignored
Gradient Boosting, XGBoost — trained without class balancing, to see how boosted trees handle the imbalance on their own
Evaluation — accuracy, precision, recall, F1, and ROC-AUC on the held-out test set, plus confusion matrices and full classification reports for every model.
Threshold tuning — the default 0.5 decision threshold is rarely optimal for imbalanced problems, so each model's threshold is swept from 0.05 to 0.95, and the value that maximises F1 on the training set is selected, then evaluated once on the test set (kept separate throughout, so the test set is never used to choose the threshold).
Interactive interface — an ipywidgets-based form (see Interface below) that takes the same 17 input fields and returns a live purchase probability from the trained model.
Why these evaluation metrics (not just accuracy)

With only 15.47% of sessions converting, accuracy is misleading — a model can score ~85% by never predicting a purchase at all. Instead:

Precision — of the sessions the model flags as "will purchase," how many actually did? Matters if acting on a prediction costs something (e.g. a discount).
Recall — of the sessions that did purchase, how many did the model catch? Matters if missing a converting session is costly.
F1 — the balance between the two.
ROC-AUC — how well the model ranks purchasers above non-purchasers across all possible thresholds, independent of which threshold is eventually chosen. This is the primary metric used to compare models, since it isn't affected by the imbalance the way accuracy is.
Results (test set, 2,466 sessions)
Model	Accuracy	Precision	Recall	F1	ROC-AUC
XGBoost	0.901	0.716	0.599	0.652	0.929
Gradient Boosting	0.901	0.714	0.602	0.653	0.928
Random Forest	0.865	0.543	0.788	0.643	0.925
Decision Tree	0.835	0.481	0.840	0.612	0.916
Logistic Regression	0.850	0.511	0.749	0.607	0.896

XGBoost narrowly leads on ROC-AUC and is used as the final model.

A clear pattern emerges from the table: the three models trained with class_weight="balanced" (Logistic Regression, Decision Tree, Random Forest) trade precision for recall — they catch more true purchasers (74–84% recall) but flag more false positives along the way (48–54% precision). Gradient Boosting and XGBoost, trained without class balancing, land on the opposite side of that trade-off: higher precision (~71–72%) but lower recall (~60%). Neither behaviour is "wrong" — which one you'd deploy depends on whether false positives (unnecessary discounts) or false negatives (missed converting customers) cost the business more.

Threshold tuning results
Model	Best threshold	Test F1 (tuned)	Test F1 (default 0.5)
XGBoost	0.40	0.676	0.652
Gradient Boosting	0.40	0.669	0.653
Random Forest	0.70	0.659	0.643
Decision Tree	0.75	0.662	0.612
Logistic Regression	0.60	0.625	0.607

Every model improves with a tuned threshold instead of the default 0.5 — a reminder that the 0.5 cutoff scikit-learn uses by default is just a starting point, not a rule.

Interface

A small ipywidgets form (numeric fields for the behavioural metrics, dropdowns for categorical fields, e.g. Month, VisitorType) lets you enter session values and click Predict to get a purchase probability from the trained pipeline — useful for sanity-checking the model on hand-crafted examples (e.g. "what if PageValues is high but BounceRates is also high?").

Setup / running the notebook

This notebook was built and run in Google Colab; a few cells assume that environment (google.colab.files, and paths under /content/). To run it elsewhere:

Place online_shoppers_intention.csv in the same folder as the notebook.
Replace any /content/... path with a local path (e.g. "online_shoppers_intention.csv").
Remove or replace the google.colab.files.upload() call in the model-loading cell — it's only needed for re-uploading a saved model file between separate Colab sessions.
Install dependencies: pip install pandas numpy scikit-learn xgboost matplotlib seaborn ipywidgets joblib
What I'd add next
scale_pos_weight for XGBoost / Gradient Boosting, to see whether explicit class balancing changes their precision-recall trade-off to match the other three models
Feature importance analysis (SHAP or built-in importances) to confirm PageValues is the dominant signal, as prior work on this dataset suggests
Cross-validation instead of a single train/test split, for a more robust performance estimate
