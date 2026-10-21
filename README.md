# DevSecOps Auditor
## Agent 1 (Member 1)
- Dataset: NASA PROMISE (PC1, KC1) from https://github.com/ApoorvaKrisna/NASA-promise-dataset-repository
- Put pc1.csv in data/ then run from the project root:
  python agents/agent1_metrics.py train --data data/pc1.csv
- Usage: from agents.agent1_metrics import predict_struct_risk
  Returns struct_risk_score, high_risk, threshold, status. status == "error" -> REVIEW.