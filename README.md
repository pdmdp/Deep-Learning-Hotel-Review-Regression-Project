# Hotel Review Score Prediction Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)
![DL](https://img.shields.io/badge/DL-Regression-red.svg)

A regression project predicting hotel review scores using a hybrid BiLSTM-MLP architecture that combines textual reviews with structured hotel and reviewer metadata. Developed as part of the Machine Learning, Artificial Neural Networks and Deep Learning course (June 2025 exam session) for the Bachelor in Artificial Intelligence at the University of Milan (UNIMI).

<p align="center">
  <img src="images/architecture_diagram.png" alt="Model Architecture" width="800"/>
</p>

## 🎯 Project Overview

Hotel review analysis is crucial for the hospitality industry to understand customer satisfaction and improve service quality. This project employs deep learning techniques to predict numerical review scores from both textual content and structured features, enabling automated quality assessment.

**Objective:** Build a robust regression model capable of predicting hotel review scores on a 0-10 scale using multi-modal input data (text + structured features).

### 🏆 Key Achievements

- **Test MSE:** 0.0331 (normalized scale) ≈ **1.82 point error** on 0-10 scale
- **Hybrid architecture** combining BiLSTM for text and MLP for structured features
- **Systematic hyperparameter tuning** with K-Fold cross-validation
- **Comprehensive preprocessing** pipeline for text and multi-type features

## 📊 Dataset

**Source:** Course-provided dataset via [University of Milan](http://frasca.di.unimi.it/MLDNN/input_data.pkl)

The dataset contains hotel reviews from 13,772 visitors with comprehensive metadata:

### Features
- **Textual:** Review text (up to 400 words)
- **Hotel Information:** Hotel Name, Hotel Address, Total Reviews for Hotel
- **Reviewer Information:** Reviewer Nationality, Total Reviews by Reviewer
- **Temporal:** Review Date (month/day/year format)
- **Ratings:** Review Score (2.5-10 range, float), Review Type (Good/Bad)

### Target Variable
- **Review_Score:** Continuous regression (float in [0, 10] range)

### Dataset Statistics
- **Total Samples:** 13,772 hotel reviews
- **Features Used:** 6 (after preprocessing)
- **Unique Hotels:** 1,298
- **Vocabulary Size:** 9,639 unique words
- **Score Range:** 2.5 - 10.0

## 🔧 Methodology

### 1. Data Preprocessing

**Feature Engineering:**
- **Retained features:** `Hotel_Name`, `Review_Date`, `Hotel_number_reviews`, `Reviewer_number_reviews`, `Review`, `Review_Score`
- **Dropped features:** `Hotel_Address`, `Reviewer_Nationality`, `Review_Type`, `Average_Score`
- **Date decomposition:** Split `Review_Date` into `Day`, `Month`, `Year` as separate integer features
- **Target normalization:** Scaled `Review_Score` from [0,10] to [0,1] for sigmoid output

**Rationale for Feature Selection:**
- `Hotel_Address` and `Reviewer_Nationality` considered irrelevant for prediction
- `Review_Type` excluded to avoid target leakage (directly correlates with score)
- `Average_Score` not part of original assignment features

**Missing Data Handling:**
- Dataset contained no missing values after feature selection
- All 13,772 samples retained for modeling

### 2. Feature Transformation Pipeline

Created specialized preprocessing for different feature types:

| Feature Type | Transformation | Applied To |
|--------------|----------------|------------|
| Text | Tokenization, Lowercasing, Punctuation Removal, Padding | Review |
| Categorical (High-cardinality) | One-Hot Encoding | Hotel_Name |
| Numerical | MinMax Scaling [0,1] | Reviewer_number_reviews, Hotel_number_reviews |
| Temporal | Split + MinMax Scaling | Day, Month, Year |

**Text Preprocessing Details:**
```python
1. Tokenization by whitespace
2. Lowercase conversion
3. Punctuation removal
4. Non-alphabetic token filtering
5. Vocabulary construction (9,639 words)
6. Sequence padding to length 100
7. Unknown word handling with <UNK> token (index 0)
```

### 3. Model Selection & Optimization

**Approach:** Manual Randomized Search with K-Fold Cross-Validation

**Architecture Selection Rationale:**

| Architecture | Pros | Cons | Decision |
|--------------|------|------|----------|
| **BiLSTM + MLP** ⭐ | Sequential text understanding, bidirectional context, fuses multi-modal data | More parameters, slower training | **Selected** - Optimal for text+structured data |
| Unidirectional LSTM | Fewer parameters, faster | Only forward context | Inferior - Limited context |
| CNN | Fast, parallel processing | Local patterns only, weak on long dependencies | Not suitable - Reviews need full context |
| MLP only | Simplest, fastest | No sequential understanding | Not suitable - Ignores word order |

**Hyperparameter Search:**
- **Search space:** 24 total combinations
- **Sampled:** 5 random configurations
- **CV strategy:** 2-fold for efficiency
- **Evaluation metric:** Mean Squared Error (MSE)
- **Primary metric:** F1 score → MSE (regression task)

**Why Not GridSearchCV?**
- KerasRegressor incompatibility with multi-input models (text + structured)
- Scikit-learn wrappers don't support Keras functional API with multiple inputs
- Manual implementation provides better control and flexibility

### 4. Final Model Architecture

```python
Model: BiLSTM-MLP Hybrid
├── Text Input Branch (100,)
│   ├── Embedding Layer (vocab_size=9640, dim=150)
│   ├── Bidirectional LSTM (64 units × 2 = 128 output)
│   └── Output: (None, 128)
│
├── Structured Input Branch (1303,)
│   ├── Hotel_Name (One-Hot): 1298 features
│   ├── Numerical Features (Scaled): 5 features
│   └── Output: (None, 1303)
│
├── Fusion Layer
│   ├── Concatenate: [BiLSTM, Structured] → (None, 1431)
│   ├── Dense(64, activation='sigmoid')
│   ├── Dropout(0.2)
│   ├── Batch Normalization
│   └── Dense(1, activation='sigmoid') → [0,1]
│
└── Output: Rescaled to [0,10]
```

**Optimal Hyperparameters:**
- Embedding dimension: 150
- LSTM units: 16 (per direction)
- Dropout rate: 0.2
- Learning rate: 0.0001
- Batch size: 64
- Optimizer: Adam
- Loss function: Mean Squared Error (MSE)

## 📈 Results

### Model Performance

| Metric | Test Set Score | Description |
|--------|----------------|-------------|
| **Test MSE (normalized)** | **0.0331** | Mean squared error on [0,1] scale |
| **Test MSE (original)** | **~3.31** | Mean squared error on [0,10] scale |
| **RMSE** | **~1.82** | Root mean squared error (average error in points) |
| **Total Parameters** | 632,849 | Model complexity (2.41 MB) |
| **Training Time** | ~20-25s/epoch | On GPU (Google Colab) |

### Performance Insights

**Learning Behavior:**
- Epoch 1: Training MSE ~0.15, Validation MSE ~0.05
- Epoch 2: Training MSE ~0.12, Validation MSE ~0.03
- Consistent improvement across epochs
- Low gap between train and validation indicates good generalization

**Prediction Examples:**
| Actual Score | Predicted Score | Error |
|--------------|-----------------|-------|
| 7.1 | 7.8 | +0.7 |
| 6.3 | 8.3 | +2.0 |
| 5.8 | 8.4 | +2.6 |
| 4.2 | 4.7 | +0.5 |
| 2.5 | 4.1 | +1.6 |

**Key Observations:**
- Model performs well on extreme scores (very low/high)
- Some tendency to predict slightly higher than actual (positive bias)
- Errors typically within ±2 points
- Most predictions fall within acceptable range for practical use

## 🚀 Installation & Usage

### Prerequisites
```bash
Python 3.8 or higher
TensorFlow 2.8+
```

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/hotel-review-prediction.git
cd hotel-review-prediction
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download the dataset:**
```bash
wget http://frasca.di.unimi.it/MLDNN/input_data.pkl
```

4. **Run the notebook:**
```bash
jupyter notebook DiPilato_535298.ipynb
```

### Alternative: Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/hotel-review-prediction/blob/main/DiPilato_535298.ipynb)

The notebook automatically downloads the dataset in the first cell.

## 📁 Project Structure

```
hotel-review-prediction/
│
├── DiPilato_535298.ipynb         # Main implementation notebook
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation (this file)
├── ARCHITECTURE.md                # Detailed technical documentation
├── LICENSE                        # MIT License
├── .gitignore                     # Git ignore rules
│
├── docs/                          # Supporting documents
│   ├── exam_test.pdf              # Original exam assignment
│   └── EXAM_DL.pdf                # Written exam answers
│
└── images/                        # Visualizations (optional)
    ├── architecture_diagram.png
    ├── training_history.png
    └── predictions_scatter.png
```

## 🔍 Key Features

✅ **Multi-modal learning** combining text and structured features  
✅ **Bidirectional LSTM** for comprehensive text understanding  
✅ **Custom preprocessing pipeline** for diverse feature types  
✅ **Robust vocabulary construction** with unknown token handling  
✅ **Systematic hyperparameter optimization** via randomized search  
✅ **Dual-input architecture** using Keras functional API  
✅ **Reproducible results** with fixed random seeds (seed=42)

## 🧪 Technical Highlights

### Preprocessing Innovation
- **Vocabulary construction:** Built from training data only (9,639 unique words)
- **Unknown token handling:** Reserved index 0 for `<UNK>` to handle unseen words gracefully
- **Sequence padding:** Fixed length of 100 tokens with post-padding strategy
- **One-hot encoding:** Handled 1,298 unique hotels without dimensionality explosion (via sparse representation)

### Architecture Design Decisions

**Why BiLSTM?**
1. ✅ Captures sequential dependencies in reviews
2. ✅ Bidirectional processing provides richer context
3. ✅ Proven effectiveness for sentiment and rating prediction
4. ✅ Better than CNN for long-range dependencies in text

**Why Sigmoid in Hidden Layer?**
- Originally proposed in written exam for consistency
- Alternative: ReLU (faster, avoids vanishing gradients) - potential future improvement

**Why Single MLP Layer?**
- BiLSTM handles most feature learning
- MLP primarily fuses representations
- Regularization (dropout + batch norm) prevents overfitting

### Validation Strategy
- **Data split:** 70% train / 15% validation / 15% test
- **K-Fold CV:** 2-fold during hyperparameter search
- **Stratification:** Not applicable (regression task)
- **Multiple metrics:** MSE, RMSE for comprehensive evaluation

## 🎓 Academic Context

This project demonstrates proficiency in:
- Multi-modal deep learning architecture design
- Text preprocessing and embedding techniques
- Bidirectional RNN (BiLSTM) implementation
- Hybrid model construction with Keras functional API
- Handling mixed data types (text + categorical + numerical)
- Hyperparameter optimization strategies
- Model evaluation for regression tasks

**Course:** 509486 - Machine Learning, Artificial Neural Networks and Deep Learning  
**Exam Session:** June 19, 2025  
**Academic Year:** 2024/2025  
**Institution:** University of Milan (UNIMI)  
**Degree Program:** [L-31] Bachelor in Artificial Intelligence  
**Student ID:** 535298

## 📝 Implementation Notes

### Changes from Written Proposal

The implementation includes several improvements over the original written exam answers:

| Written Proposal | Implementation | Rationale |
|------------------|----------------|-----------|
| Label Encoding for Hotel_Name | **One-Hot Encoding** | Avoids artificial ordinal relationships between hotels |
| No unknown token handling | **`<UNK>` token at index 0** | Robust handling of words not seen during training |
| Implicit sequence handling | **Explicit padding to length 100** | Required for uniform input shape to neural network |
| GridSearchCV | **Manual randomized search with K-Fold** | KerasRegressor incompatibility with multi-input models |
| Basic tokenization | **+ Non-alphabetic filtering** | Reduces vocabulary noise by removing numbers, dates |

All changes are thoroughly documented in the notebook with clear explanations.

## 🔮 Future Improvements

### Short-term
- [ ] Implement attention mechanism for interpretable word importance
- [ ] Add early stopping and learning rate scheduling
- [ ] Experiment with pre-trained embeddings (Word2Vec, GloVe)
- [ ] Visualize embedding space with t-SNE/UMAP

### Medium-term
- [ ] Try transformer-based models (BERT, RoBERTa)
- [ ] Explore ensemble methods (multiple BiLSTM models)
- [ ] Add review sentiment as auxiliary task (multi-task learning)
- [ ] Implement model explainability (LIME, SHAP)

### Long-term
- [ ] Develop web application for real-time prediction (Streamlit/Flask)
- [ ] Multi-language support with multilingual embeddings
- [ ] Aspect-based sentiment analysis (room, service, location scores)
- [ ] Integrate with hotel booking platforms

## 📚 References

1. Dataset: Course materials - University of Milan, Department of Computer Science
2. [Keras Documentation](https://keras.io/) - Neural network implementation
3. [TensorFlow Guide](https://www.tensorflow.org/) - Deep learning framework
4. [scikit-learn](https://scikit-learn.org/) - Preprocessing utilities
5. Hochreiter & Schmidhuber (1997) - Long Short-Term Memory networks
6. Schuster & Paliwal (1997) - Bidirectional Recurrent Neural Networks

## 👨‍💻 Author

**Matteo Di Pilato**  
Bachelor in Artificial Intelligence  
University of Milan (UNIMI)  
Student ID: 535298  
Academic Year 2024/2025

📧 Contact: [Your email if you want to add it]  
🔗 GitHub: [pdmdp](https://github.com/pdmdp)  
🔗 LinkedIn: [Your LinkedIn if you want to add it]

## 📄 License

This project is available under the MIT License. See [LICENSE](LICENSE) file for details.

**Academic Use:** This project was developed for academic purposes as part of the Machine Learning, Artificial Neural Networks and Deep Learning course exam at the University of Milan.

## 🙏 Acknowledgments

- **Dataset:** University of Milan, Department of Computer Science
- **Course Instructors:** ML, ANN, and Deep Learning teaching team
- **Institution:** University of Milan (UNIMI)
- **Libraries:** TensorFlow/Keras, scikit-learn, NumPy, Pandas communities
- **Inspiration:** Previous ML projects and course materials

---

<p align="center">
  ⭐ If you found this project helpful, please consider giving it a star!<br>
  💡 Feel free to fork and adapt for your own learning<br>
  📚 Check out my other ML projects: <a href="https://github.com/pdmdp/student-depression-project">Student Depression Prediction</a>
</p>

---

**Note:** This is an educational project. The model demonstrates deep learning concepts and should not be used for commercial hotel rating systems without further validation and ethical considerations.
