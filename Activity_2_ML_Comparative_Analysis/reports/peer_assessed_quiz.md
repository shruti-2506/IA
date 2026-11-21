# Peer-Assessed Classroom Quiz (5 Marks)

**Course Code:** 2304322T | **Course:** Artificial Intelligence and Machine Learning  
**Activity No.:** 2 | **Academic Year:** 2026–2027 | **Semester:** V  
**Department:** Computer Engineering | **Institute:** MIT Academy of Engineering, Alandi (D), Pune  

**Student Name:** `[Student Name]`  
**Seat Number:** `[Seat Number]`  
**PRN Number:** `[PRN Number]`  
**Evaluator Name (Peer Reviewer):** `[Peer Reviewer Name]`  
**Total Marks:** 5 Marks (0.5 Mark per Question) | **Time Allowed:** 15 Minutes  

---

### Question 1: Regression Evaluation Metrics
**Which of the following evaluation metrics is most sensitive to large outlier errors in a regression task?**  
- [ ] A) Mean Absolute Error (MAE)  
- [ ] B) Mean Squared Error (MSE)  
- [ ] C) R-squared ($R^2$) Score  
- [ ] D) Mean Absolute Percentage Error (MAPE)  

**Correct Answer:** **B) Mean Squared Error (MSE)**  
*Explanation:* MSE squares the residual errors $(y - \hat{y})^2$ before averaging, which exponentially penalizes large deviations compared to MAE, making it highly sensitive to outliers.

---

### Question 2: Data Preprocessing & Leakage
**To prevent data leakage when applying feature scaling (such as StandardScaler), what is the correct procedure?**  
- [ ] A) Fit the scaler on the entire dataset before splitting into train and test sets.  
- [ ] B) Fit the scaler separately on the test set and use those parameters on the train set.  
- [ ] C) Fit the scaler strictly on the training set, then transform both train and test sets.  
- [ ] D) Feature scaling does not cause data leakage, so fitting order does not matter.  

**Correct Answer:** **C) Fit the scaler strictly on the training set, then transform both train and test sets.**  
*Explanation:* Fitting the scaler on test data exposes the model to test set distribution statistics (mean and variance), leading to optimistic performance estimates and target leakage.

---

### Question 3: Cross-Validation Strategies
**Why is Stratified K-Fold Cross-Validation preferred over standard K-Fold Cross-Validation for classification tasks?**  
- [ ] A) It runs $K$ times faster than standard K-Fold.  
- [ ] B) It preserves the original target class percentage in each split fold.  
- [ ] C) It eliminates the need for a test dataset completely.  
- [ ] D) It converts multi-class problems into binary regression problems.  

**Correct Answer:** **B) It preserves the original target class percentage in each split fold.**  
*Explanation:* Stratified K-Fold ensures that each fold contains roughly the same proportion of classes as the complete dataset, preventing folds from missing minority classes.

---

### Question 4: Multi-Class Metrics
**In an imbalanced multi-class classification problem, which metric accounts for class support by weighting each class's score by its frequency?**  
- [ ] A) Macro-averaged F1 Score  
- [ ] B) Micro-averaged Precision  
- [ ] C) Weighted-averaged F1 Score  
- [ ] D) Binary ROC-AUC  

**Correct Answer:** **C) Weighted-averaged F1 Score**  
*Explanation:* Weighted F1 calculates the F1-score for each class independently and averages them weighted by the number of true instances (support) in each class.

---

### Question 5: Bias-Variance Tradeoff
**An unpruned Decision Tree with infinite depth that achieves 100% accuracy on training data but 55% accuracy on test data suffers from:**  
- [ ] A) High Bias and Underfitting  
- [ ] B) High Variance and Overfitting  
- [ ] C) High Bias and High Variance  
- [ ] D) Low Variance and Underfitting  

**Correct Answer:** **B) High Variance and Overfitting**  
*Explanation:* Deep unconstrained decision trees memorize specific noise in the training set (near-zero training error), causing high variance and poor generalization on unseen test data.

---

### Question 6: Tree Ensemble Algorithms
**How does a Random Forest Regressor reduce prediction variance compared to an individual Decision Tree?**  
- [ ] A) By training trees sequentially on the residual errors of prior trees.  
- [ ] B) By averaging the predictions of decorrelated trees trained on bootstrap samples with random feature subsets.  
- [ ] C) By converting non-linear numerical features into linear hyperplanes.  
- [ ] D) By discarding all features that have zero correlation with the target.  

**Correct Answer:** **B) By averaging the predictions of decorrelated trees trained on bootstrap samples with random feature subsets.**  
*Explanation:* Random Forest employs Bootstrap Aggregating (Bagging) alongside random feature subspacing, which substantially dampens variance without inflating individual tree bias.

---

### Question 7: Support Vector Machines (SVM)
**What role does the kernel trick (e.g., Radial Basis Function - RBF kernel) play in Support Vector Machines?**  
- [ ] A) It reduces the training time of SVM to $O(1)$.  
- [ ] B) It implicitly projects input features into a higher-dimensional space where non-linear patterns become linearly separable.  
- [ ] C) It automatically performs forward feature selection.  
- [ ] D) It normalizes the target variable to follow a Gaussian distribution.  

**Correct Answer:** **B) It implicitly projects input features into a higher-dimensional space where non-linear patterns become linearly separable.**  
*Explanation:* The kernel trick computes inner products in high-dimensional Hilbert spaces without explicitly calculating coordinates, enabling linear separation of complex non-linear manifolds.

---

### Question 8: Weather Regression Target Selection
**In our Open-Meteo weather forecasting project, why is 2-meter air temperature (`temperature_2m (°C)`) suitable as a regression target?**  
- [ ] A) It is a categorical variable with three discrete weather states.  
- [ ] B) It is a continuous numerical variable exhibiting strong physical dependencies on solar radiation, humidity, and pressure.  
- [ ] C) It can only be evaluated using confusion matrices.  
- [ ] D) It contains no seasonal or diurnal periodicity.  

**Correct Answer:** **B) It is a continuous numerical variable exhibiting strong physical dependencies on solar radiation, humidity, and pressure.**  
*Explanation:* Air temperature is a continuous physical quantity governed by thermodynamic equations, making it a classic regression benchmark for continuous loss functions.

---

### Question 9: Neural Network Architecture
**In a Multi-Layer Perceptron (MLP) Regressor, what is the primary role of the non-linear activation function (such as ReLU)?**  
- [ ] A) To scale inputs between 0 and 1.  
- [ ] B) To enable the network to learn non-linear input-output mappings rather than collapsing into a simple linear regression model.  
- [ ] C) To split the dataset into training and validation sets.  
- [ ] D) To prevent missing values from occurring in hidden layers.  

**Correct Answer:** **B) To enable the network to learn non-linear input-output mappings rather than collapsing into a simple linear regression model.**  
*Explanation:* Without non-linear activation functions, any composition of linear matrix transformations remains purely linear, incapable of approximating complex functions.

---

### Question 10: Hyperparameter Tuning
**What is the key advantage of GridSearchCV over manual trial-and-error parameter tuning?**  
- [ ] A) GridSearchCV guarantees 100% test accuracy on any dataset.  
- [ ] B) GridSearchCV systematically tests all specified parameter combinations using cross-validation to prevent subjective bias.  
- [ ] C) GridSearchCV never requires cross-validation.  
- [ ] D) GridSearchCV only works on linear algorithms.  

**Correct Answer:** **B) GridSearchCV systematically tests all specified parameter combinations using cross-validation to prevent subjective bias.**  
*Explanation:* GridSearchCV performs an exhaustive search over an explicit grid of hyperparameters, evaluating each candidate via cross-validation to select the mathematically optimal configuration.

---

### Peer Assessment Scoring Matrix

| Question No. | Correct (0.5 M) | Incorrect (0 M) | Evaluator Remarks |
| :---: | :---: | :---: | :--- |
| Q1 | [ ] | [ ] | |
| Q2 | [ ] | [ ] | |
| Q3 | [ ] | [ ] | |
| Q4 | [ ] | [ ] | |
| Q5 | [ ] | [ ] | |
| Q6 | [ ] | [ ] | |
| Q7 | [ ] | [ ] | |
| Q8 | [ ] | [ ] | |
| Q9 | [ ] | [ ] | |
| Q10 | [ ] | [ ] | |
| **Total Marks** | **/ 5.0 Marks** | | **Signature:** ________________ |
