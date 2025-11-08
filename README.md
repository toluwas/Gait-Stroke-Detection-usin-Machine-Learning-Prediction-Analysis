# Gait-Stroke-Detection-usin-Machine-Learning-Prediction-Analysis
Machine learning-based stroke detection system using gait pattern analysis.
This project focuses on developing a machine learning model for early stroke detection based on human gait analysis.
By analyzing walking patterns captured through force plates and infrared motion sensors, the model identifies gait abnormalities that may indicate neurological damage or risk of stroke.

🎯 Objectives

Extract gait features from force plate and motion capture data.

Train machine learning models to classify between stroke-affected and healthy individuals.

Evaluate performance using accuracy, ROC-AUC, and confusion matrices.

Investigate gait asymmetry and stride pattern deviations as predictive biomarkers.

🧩 Dataset Description

The dataset includes:

Force plate readings (ground reaction forces, gait phase data)

Infrared marker data (joint coordinates, step length, speed, cadence)

Demographic information (age, gender, health status)

Data collected under laboratory conditions for gait analysis and rehabilitation research by 22 persons

⚙️ Preprocessing and Feature Extraction

Data cleaning and normalization

Temporal alignment of gait cycles

Feature extraction: stride time, stance time, swing time, step symmetry, step_times_left[:min_len],
                'step_time_right': step_times_right[:min_len],
                'force_asymmetry': force_asymmetry[:min_len],
                'stance_time_left': stance_left[:min_len],
                'swing_time_left': swing_left[:min_len],
                'impulse_left': impulse_left[:min_len],
                'stride_length_left': stride_L[:min_len],
                'vertical_disp_left': disp_L[:min_len],
                'foot_velocity_left': vel_L[:min_len],
                'peak_force_left': peak_force_left[:min_len],
                'peak_force_right': peak_force_right[:min_len],
                'step_time_diff': np.abs(step_times_left[:min_len] - step_times_right[:min_len]),
                'stance_time_right': stance_right[:min_len],
                'swing_time_right': swing_right[:min_len],
                'impulse_right': impulse_right[:min_len],
                'vertical_disp_right': disp_R[:min_len],
                'foot_velocity_right': vel_R[:min_len],
                'stride_length_right': stride_R[:min_len],
                'step_width': width_R[:min_len],
                'subject': subject,
                'speed': speed,

Feature Engineering Techniques used include: Correlation Matrix, RFE, LDA, Dimensionality reduction using PCA, 

🧠 Machine Learning Models

Random Forest Classifier

Support Vector Machine (SVM)

Artificial Neural Network (ANN)

Evaluation metrics:

Accuracy, Confusion Matrix

Precision, F1- Score, and Recall

ROC Curve and AUC

🧰 Tools and Libraries

Python 3.x

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

TensorFlow / Keras (for ANN)
