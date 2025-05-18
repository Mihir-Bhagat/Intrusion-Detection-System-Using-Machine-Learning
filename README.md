# Cybersecurity Attack Detection 

### Comparing Preprocessing Strategies with Supervised Machine Learning Algorithms

This project explores various preprocessing techniques and supervised machine learning algorithms to improve cybersecurity attack detection accuracy using the UNSW-NB15 dataset.

---

##  Table of Contents

* [Project Overview](#project-overview)
* [Objectives](#objectives)
* [Dataset](#dataset)
* [Methodology](#methodology)
* [Machine Learning Models](#machine-learning-models)
* [Results Summary](#results-summary)
* [Conclusion](#conclusion)
* [Limitations & Future Work](#limitations--future-work)
* [Team Members](#team-members)
* [References](#references)

---

##  Project Overview

Cyber threats are rising exponentially, making intrusion detection a critical area of research. This project aims to improve detection accuracy using a combination of:

* Missing value imputation strategies
* Class balancing via oversampling (SMOTE)
* Feature selection techniques
* Eight supervised ML algorithms

---

##  Objectives

* Impute missing values from highly relevant and correlated columns in the dataset.
* Handle class imbalance using oversampling (SMOTE).
* Use feature selection (Gini Index and Information Gain) to reduce dimensionality.
* Evaluate multiple ML algorithms on performance metrics.

---

##  Dataset

**Dataset Used:** [UNSW-NB15 (Kaggle)](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15/data)

* **Records:** 175,341 (Train), 82,332 (Test)
* **Features:** 49 (incl. IPs, ports, durations, bytes, etc.)
* **Label:** `0` (Normal), `1` (Attack)
* **Attack Categories:** DoS, Exploits, Reconnaissance, Fuzzers, Worms, etc.

---

##  Methodology

### 1. **Preprocessing:**

* **Frequency Encoding** for categorical columns
* **RobustScaler** for scaling while being resistant to outliers
* **Missing Value Imputation** in the `service` column using:

  * Random Imputation
  * Most Frequent Imputation
  * Correlation-based Imputation (using `proto`, `state`, `dwin`, `swin`, etc.)

### 2. **Data Balancing:**

* **SMOTE** used to balance the dataset where class 0 (normal) was underrepresented

### 3. **Feature Selection:**

* **Gini Index** (Threshold < 0.01)
* **Information Gain** (Threshold > 0.1)

---

##  Machine Learning Models

We evaluated **eight supervised learning models** on multiple versions of preprocessed datasets:

* **Decision Tree Classifier**
* **Random Forest Classifier**
* **XGBoost Classifier**
* **AdaBoost Classifier**
* **K-Nearest Neighbors (KNN)**
* **Gaussian Naive Bayes**
* **Multi-Layer Perceptron (MLP)**
* **Logistic Regression**

Metrics used:

* Accuracy
* Precision, Recall, F1-Score
* ROC AUC Score

---

##  Results Summary

* **Best Imputation:** Most Frequent Strategy based on relevant column combinations
* **Best Feature Selection:** Information Gain
* **Best Model:** **XGBoost Classifier**
* **Performance:** Highest classification accuracy and ROC AUC in most scenarios

---

##  Conclusion

This study highlights the importance of strategic preprocessing to enhance ML-based cybersecurity systems. Key takeaways include:

* Imputing missing categorical values using relevant and correlated features improves data quality.
* Balancing the dataset is crucial to avoid biased classifiers.
* Feature selection aids model interpretability and generalization.
* XGBoost consistently outperforms other models in detection accuracy.

---

##  Limitations & Future Work

### Limitations:

* Potential latency in real-time detection scenarios.
* High computational cost for preprocessing and training.
* Limited scalability for large-scale live network environments.

### Future Work:

* Integrate real-time IDS/IPS systems with improved speed.
* Develop adaptive learning models for emerging threat patterns.
* Explore deep learning and reinforcement learning for threat prediction.

---

##  Team Members

* **Arslan Anzar (A001)**
* **Mihir Bhagat (A003)**
* **Jash Moze (A017)**
* **Suman Yadav (A036)**
* **Gauri Sarangdhar (A037)**

**Supervisor:** *Prof. Shraddha Sarode*
**Institution:** NMIMS NSOMASA, Mumbai
**Date:** April 18, 2024

---

##  References

*  [Project Code](https://drive.google.com/drive/folders/1ZClqpm8PkKkESS5hGHGKDwpsvGI9K4La?usp=sharing)
*  [UNSW-NB15 Dataset (Kaggle)](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15/data)
*  Research Papers:

  * [Ensemble Intrusion Detection](https://infonomics-society.org/wp-content/uploads/A-Stacked-Ensemble-Intrusion-Detection-Approach-for-the-Protection-of-Information-System.pdf)
  * [Efficient DoS Preprocessing](https://link.springer.com/article/10.1007/s10489-021-02968-1)
  * [Feature Selection IDS](https://arxiv.org/abs/2205.06177)
  * [UNSW-NB15 NIDS Review](https://journal.uob.edu.bh/bitstream/handle/123456789/3580/paper%205.pdf)


