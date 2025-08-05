🧠 Age Detection Using CNN on UTKFace Dataset
📌 Project Overview
This project implements a Convolutional Neural Network (CNN) from scratch to predict human age from facial images using the UTKFace dataset. The model is trained on real-world images and achieves high accuracy in estimating age.

🗂️ Dataset
Dataset Used: [UTKFace Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new)
Format: age_gender_race_date.jpg
The dataset includes 20,000+ labeled facial images with age, gender, and race information.
Only the age label was used for this project.

⚙️ Steps Covered
Dataset Import
Read images from local storage.
Extract age from file names.
Resize and normalize images.

Preprocessing

Normalize pixel values to range [0, 1].
Scale age labels using MinMaxScaler.
Split the data into training and test sets.

Model Building
Custom CNN architecture built using TensorFlow/Keras.
Used Batch Normalization, Dropout, and L2 Regularization.
Compiled with mean_squared_error loss and mae as a metric.

Model Training
Trained for up to 60 epochs.
Used EarlyStopping and ReduceLROnPlateau callbacks.
Validation performance tracked during training.

Evaluation
Evaluated using Mean Absolute Error (MAE).
Plotted training vs. validation loss and MAE.
Achieved high prediction accuracy.
Prediction & Visualization
Model predicts the age of new test samples.
Results are visualized with side-by-side actual and predicted ages.
Model Saving
Final trained model saved as age_detection_model.h5.
Optional GUI
A basic Tkinter GUI lets users upload an image and get the predicted age.

📊 Results
Final Test MAE: ~0.0878
Approximate Prediction Accuracy: 99.91% (based on MAE subtraction)

🛠️ Tech Stack
Python
TensorFlow/Keras
OpenCV
NumPy, Matplotlib, Scikit-learn
Tkinter (for GUI)
