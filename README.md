# KNN from Scratch for Image Classification

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project features a from-scratch implementation of the K-Nearest Neighbors (KNN) algorithm to classify images from the **Fashion MNIST dataset**. It covers the complete machine learning pipeline, from data exploration and preprocessing to hyperparameter tuning and final model evaluation.

---

## 📋 Project Pipeline

The project is structured as a comprehensive machine learning workflow:

1.  **Data Exploration:**
    * Loaded the Fashion MNIST dataset, which consists of 70,000 grayscale images of clothing items across 10 classes.
    * Visualized sample images from each class to understand the data.
    * Used **t-SNE** to reduce the data's dimensionality to 3D and visualize the class separability in a scatter plot.

2.  **Data Preparation:**
    * Split the dataset into training (70%), validation (15%), and testing (15%) sets.
    * Implemented a preprocessing pipeline that includes:
        * **Min-Max Scaling:** Normalized pixel values to a [0, 1] range based on the training set.
        * **PCA (Principal Component Analysis):** Reduced the dimensionality of the data while preserving a specified ratio of variance (e.g., 90%).

3.  **KNN Algorithm Implementation:**
    * Developed the KNN classifier **from scratch** using only NumPy and standard Python libraries.
    * Calculated distances between data points using the **Euclidean distance** metric.
    * Implemented a **distance-weighted voting mechanism** to predict labels, where closer neighbors have more influence on the final prediction.

4.  **Hyperparameter Tuning:**
    * Systematically experimented with different values for `k` (the number of neighbors) and the PCA `variance_ratio`.
    * Identified the optimal hyperparameter configuration based on the model's accuracy on the validation set.

5.  **Final Model Evaluation:**
    * Evaluated the final, tuned model on the unseen testing set.
    * Calculated and reported comprehensive metrics, including per-class **Precision, Recall, F1-Score, and AUC**.
    * Visualized model performance using **ROC Curves** and a **Confusion Matrix**.

---

## 🛠️ Technologies Used
* Python
* NumPy
* scikit-learn (for data loading, PCA, t-SNE, and evaluation metrics)
* Matplotlib / Seaborn
* Jupyter Notebook

---

## 🚀 Usage

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the analysis:**
    Open and run the cells in the `code.ipynb` Jupyter Notebook to execute the entire pipeline from data exploration to final evaluation.

---

## 🙏 Acknowledgments
* This project uses the Fashion MNIST dataset, created by **Zalando Research**.

---

## 📄 License
This project is licensed under the MIT License.
