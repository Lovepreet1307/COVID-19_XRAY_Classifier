# COVID-19_XRAY_Classifier
A deep learning model to classify chest X-ray images for COVID-19 detection and related respiratory conditions.
Dataset Overview
Dataset Source: COVID-19 Radiography Database on Kaggle
The dataset contains chest X-ray images across four categories:

COVID-19 (3,616 images)
Normal (10,192 images)
Lung Opacity (Non-COVID lung infection) (6,012 images)
Viral Pneumonia (1,345 images)

Data Sources
COVID-19 Images:

2,473 CXR images from padchest dataset
183 CXR images from a Germany medical school
559 CXR images from SIRM, Github, Kaggle & Twitter
400 CXR images from additional Github sources

Normal Images:

8,851 images from RSNA
1,341 images from Kaggle

Lung Opacity Images:

6,012 images from Radiological Society of North America (RSNA) CXR dataset

Viral Pneumonia Images:

1,345 images from the Chest X-Ray Images (pneumonia) database

Image Specifications

Format: Portable Network Graphics (PNG)
Resolution: 299×299 pixels

Model Architecture
The implemented CNN model includes:

Multiple convolutional layers with batch normalization
Average pooling layers
Dropout layers for regularization
Dense layers for final classification
Total Parameters: 2,943,428
Trainable Parameters: 2,900,868
Non-trainable Parameters: 42,560

Performance Metrics
Overall Results:

Accuracy: 70.16%
Precision: 64.50%
Recall: 70.53%
F-Score: 64.84%



Citation:
CopyM.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, "Can AI help in screening Viral and COVID-19 pneumonia?" IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.

Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images. arXiv preprint arXiv:2012.02238.

Objective:
This project aims to assist researchers in producing scholarly work on COVID-19 detection using chest X-ray images, contributing to the ongoing efforts to tackle the pandemic through automated screening methods.

Model Training

The model was trained with:

Batch size: 16
Learning rate: 0.002
Optimizer: Adam
Loss function: Categorical Cross-entropy
Early stopping and learning rate reduction implemented
Data augmentation applied (rotation, width/height shifts, horizontal flip)
