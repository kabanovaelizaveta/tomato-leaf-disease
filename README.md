# Tomato Leaf Disease Classification — CNN Parameter Study with MLP Performance Comparison

## Objective
Systematically study how different CNN design and training parameters affect classification accuracy on tomato leaf disease images. After establishing the best CNN configuration, compare it against a simple MLP baseline to quantify the CNN’s advantage.

## Dataset
- **Source**: [Tomato leaf disease detection (Kaggle)](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)
- **Format**: `.jpg` images of tomato leaves
- **Classes**: 10 types of disease
  - Tomato_mosaic_virus
  - Target_Spot
  - Bacterial_spot
  - Tomato_Yellow_Leaf_Curl_Virus
  - Late_blight
  - Leaf_Mold
  - Early_blight
  - Spider_mites Two-spotted_spider_miteTomato___healthy
  - Septoria_leaf_spot
- **Challenges**:
    - varied image quality;
    - risk of overfitting on small models;
    - need for careful regularization/augmentation

## Approach
- Exploratory Data Analysis (EDA): class distribution, sample visualization, image quality checks
- Image preprocessing: resizing, normalization, data augmentation (rotation, zoom, flip), one-hot labels
- Dataset split: train / validation 
- Modeling:
  - Custom Convolutional Neural Network (CNN) using Keras Sequential API
  - Multilayer Perceptron (MLP) using scikit-learn
- Regularization techniques: Dropout, Batch Normalization
- Hyperparameter tuning: strides, padding, batch size, learning rate, dropout rate

## Methods / Algorithms
- Convolutional Neural Networks (Conv2D, MaxPooling, Flatten, Dense)
- MLP (baseline): fully connected layers on flattened pixels (scikit-learn)
- Optimizers: SGD, Adam, RMSprop
- Loss: categorical crossentropy

## Experimental Factors (What We Varied)
- Architecture depth & width: number of Conv2D blocks (1 → 3), filters per block
- Kernels & downsampling: kernel size, stride (1 vs 2), padding (same vs valid), MaxPooling placement
- Normalization & regularization: BatchNorm, Dropout, L2 weight decay
- Input & color space: input resolution; color vs grayscale
- Augmentation: rotation, flips, zoom
- Optimization: SGD vs Adam vs RMSprop, learning rate, LR scheduling
- Batch size & epochs: stability vs generalization trade-offs

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Results
- **MLP**:
  - Accuracy: 0.50 on color images
  - Accuracy: 0.10 on grayscale
- **CNN (basic)**:
  - Accuracy: 0.8
- **CNN (improved)**:
  - Accuracy: 0.912
  - Loss: 0.273
  - Precision: 0.928
  - Recall: 0.897
  - AUC: 0.994
  - F1: 0.911

## Tools & Libraries
- **Deep Learning**: TensorFlow, Keras
- **Machine Learning**: scikit-learn
- **Data Handling**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Evaluation**: scikit-learn metrics

## Key Insights / Business Value
- CNNs significantly outperform MLPs in image classification tasks
- Simple MLPs are not suitable for visual recognition (performance close to random guessing)
- Regularization and augmentation effectively reduce overfitting and improve model robustness
- The solution can provide value for agritech and farmers, enabling automated disease detection in tomato plants and reducing crop losses

### Sample Predictions
- **CNN**:
<img width="793" height="812" alt="image" src="https://github.com/user-attachments/assets/90bd4111-6154-4e35-b01f-0b7e4627aec7" />

- **MLP**:
<img width="515" height="389" alt="image" src="https://github.com/user-attachments/assets/84722089-6308-4dca-8bed-7363a9374a93" />

<img width="515" height="389" alt="image" src="https://github.com/user-attachments/assets/f85f7d48-25b4-4b10-a40f-129030b00764" />
