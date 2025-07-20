# 💳 XGBoost Financial Fraud Detection Model

## 🧠 Model Summary

This XGBoost classifier is trained to detect **fraudulent financial transactions** using engineered features derived from transactional metadata. It was trained on a highly imbalanced dataset of over 6.3 million transactions, filtered down to `TRANSFER` and `CASH_OUT` types—since fraud only occurs in those categories.

The model uses domain-specific insights and advanced feature engineering to learn behavioral patterns associated with fraudulent activities. It is optimized for high **recall and precision**, making it suitable for real-time fraud prevention systems.

---

## 📊 Model Performance (on Validation Set)

| Metric                  | Score     |
|-------------------------|-----------|
| Accuracy                | 99.9%     |
| Precision (Fraud)       | 0.98      |
| Recall (Fraud)          | 1.00      |
| F1-Score (Fraud)        | 0.99      |
| ROC AUC                 | 0.9990    |
| **PR AUC (AUPRC)**      | **0.9979** |

> ⚠️ AUPRC is the most reliable metric for this imbalanced dataset.

---

## 🛠 Features Used

The following features were used to train the model:

- `amount`: Transaction amount
- `oldbalanceOrg`: Original balance of the sender
- `newbalanceOrig`: New balance of the sender
- `oldbalanceDest`: Original balance of the receiver
- `newbalanceDest`: New balance of the receiver
- `errorBalanceOrig`: Engineered feature capturing balance discrepancy for sender
- `errorBalanceDest`: Engineered feature capturing balance discrepancy for receiver
- `type_TRANSFER`: One-hot encoding for transaction type
- `type_CASH_OUT`: One-hot encoding for transaction type

---

## 🔍 Use Cases

- ✅ Real-time fraud detection in banking applications
- ✅ Alert prioritization systems for fraud analysts
- ✅ Risk scoring for financial transactions

---

## 🧪 Intended Use

This model is designed for **educational, research, and prototyping** purposes. It can be integrated into a fraud monitoring dashboard or used as a backend scoring engine. The model is best used **on transactions of type `TRANSFER` and `CASH_OUT` only**.

---

## 🚫 Limitations

- The model only works well on transactions similar in structure to the training data.
- It is not trained on `DEBIT`, `PAYMENT`, or `CASH_IN` transactions (fraud does not occur in them).
- Not suitable for production use **without further validation and retraining on live data.**

---

## 🔐 Ethics & Bias

This model makes **no assumptions about user identity**, geography, or behavior outside of financial transactions. However, like all ML systems, **model drift** and **adversarial behavior** may impact its long-term effectiveness.

---

## 📦 Deployment

This model is deployed via Hugging Face Spaces and can be tested through the interactive Gradio UI or invoked as an API using:

```python
from inference import predict
predict({...})

👤 Author
Sagar Rajak
Department of Artificial Intelligence and Data Science
[INSAID / Vivekanand Education Society’s Institute of Technology]