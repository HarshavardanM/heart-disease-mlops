# Heart-Disease-Mlops
MlOps assignment 1

| Field | Details |
| :--- | :--- |
| **Name:** | Harshavardan M |
| **BITS ID:** | 2025ct05070@wilp.bits-pilani.ac.in |
| **Date:** | 06/20/2026 |

## Submission Resources

* 📄 **[Final Project Report](PASTE_YOUR_GOOGLE_DRIVE_REPORT_LINK_HERE)**
* 🎥 **[Project Demonstration Video](PASTE_YOUR_GOOGLE_DRIVE_VIDEO_LINK_HERE)**
* 🔗 **[GitHub Repository](https://github.com/HarshavardanM/heart-disease-mlops)**
* ⚙️ **[GitHub Actions — CI Test Results](https://github.com/HarshavardanM/heart-disease-mlops/actions)**

## Project Overview

This project demonstrates an end-to-end MLOps workflow for heart disease classification using the UCI Heart Disease dataset, with the Cleveland subset used for model development and evaluation. It covers data auditing and exploratory data analysis (EDA), pipeline-based preprocessing, Logistic Regression and Random Forest comparison, hyperparameter tuning, and MLflow experiment tracking.

The project deploys the trained Scikit-learn pipeline as a FastAPI prediction service, packages it using Docker, and validates the API through automated tests and GitHub Actions continuous integration. It also demonstrates local Kubernetes deployment and operational monitoring using Prometheus and Grafana, including API health, request metrics, and HTTP server errors.

**Best demonstrated test results:** 86.89% accuracy, 81.25% precision, 92.86% recall, 86.67% F1-score, and 95.67% ROC-AUC on a held-out test set of 61 Cleveland records.

**Technology stack:** Python, Pandas, Scikit-learn, MLflow, FastAPI, Docker, GitHub Actions, Kubernetes, Prometheus, and Grafana.

**Deployment status:** Demonstrated locally using Docker and Kubernetes. The current CI workflow runs API tests and builds the Docker image; automated deployment to a remote environment is future work.
