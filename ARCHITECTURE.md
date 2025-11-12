# Model Architecture Documentation

## Overview

This document provides a detailed technical breakdown of the BiLSTM-MLP hybrid architecture used for hotel review score prediction.

**Author**: Matteo Di Pilato  
**Course**: ML, Artificial Neural Networks and Deep Learning  
**Institution**: University of Milan (UNIMI)  
**Last Updated**: June 2025

## Architecture Diagram

```
Text Input (100,)              Structured Input (1303,)
      |                                |
      v                                |
 Embedding (100, 150)                  |
      |                                |
      v                                |
 BiLSTM (128)                          |
      |                                |
      +---------------+----------------+
                      |
                      v
              Concatenate (1431,)
                      |
                      v
              Dense (64, sigmoid)
                      |
                      v
              Dropout (0.2)
                      |
                      v
           Batch Normalization
                      |
                      v
            Dense (1, sigmoid)
                      |
                      v
              Output [0,1]
```

## Layer-by-Layer Breakdown

### Input Layers

#### 1. Text Input
- **Shape**: `(None, 100)`
- **Type**: Integer sequences (word indices)
- **Range**: `[0, vocab_size-1]` where vocab_size = 9,640
- **Preprocessing**: 
  - Tokenization
  - Lowercasing
  - Punctuation removal
  - Padding to fixed length 100

#### 2. Structured Input
- **Shape**: `(None, 1303)`
- **Components**:
  - Hotel name (one-hot): 1,298 dimensions
  - Reviewer_number_reviews (scaled): 1 dimension
  - Hotel_number_reviews (scaled): 1 dimension
  - Day (scaled): 1 dimension
  - Month (scaled): 1 dimension
  - Year (scaled): 1 dimension
- **Preprocessing**: MinMax scaling to [0,1]

### Text Processing Branch

#### 3. Embedding Layer
- **Input shape**: `(None, 100)`
- **Output shape**: `(None, 100, 150)`
- **Parameters**: 
  - Vocabulary size: 9,640
  - Embedding dimension: 150 (best hyperparameter)
  - Trainable: Yes
  - Mask zero: False
- **Total parameters**: 9,640 × 150 = 1,446,000

**Purpose**: Maps discrete word indices to dense continuous vectors, capturing semantic relationships.

#### 4. Bidirectional LSTM
- **Input shape**: `(None, 100, 150)`
- **Output shape**: `(None, 128)`
- **Configuration**:
  - LSTM units: 64 per direction
  - Total output: 64 × 2 = 128 (forward + backward)
  - Return sequences: False (only final state)
  - Activation functions:
    - Hidden state: tanh
    - Gates: sigmoid
- **Parameters**: ~58,880

**Purpose**: 
- Processes text bidirectionally to capture context from both directions
- Maintains information about word order and dependencies
- Outputs fixed-size representation regardless of input length

### Fusion Layer

#### 5. Concatenate
- **Input shapes**: 
  - BiLSTM output: `(None, 128)`
  - Structured input: `(None, 1303)`
- **Output shape**: `(None, 1431)`
- **Parameters**: 0 (no trainable parameters)

**Purpose**: Combines textual and structured feature representations into single vector.

### MLP Branch

#### 6. Dense Layer 1
- **Input shape**: `(None, 1431)`
- **Output shape**: `(None, 64)`
- **Activation**: Sigmoid
- **Parameters**: 1,431 × 64 + 64 = 91,648

**Purpose**: Non-linear transformation to learn complex interactions between fused features.

#### 7. Dropout
- **Rate**: 0.2 (20% dropout)
- **Applied during**: Training only
- **Parameters**: 0

**Purpose**: Regularization to prevent overfitting by randomly dropping units.

#### 8. Batch Normalization
- **Input shape**: `(None, 64)`
- **Output shape**: `(None, 64)`
- **Parameters**: 256 (gamma, beta, moving mean, moving variance)

**Purpose**: 
- Normalizes activations for stable training
- Allows higher learning rates
- Acts as additional regularization

### Output Layer

#### 9. Dense Layer 2 (Output)
- **Input shape**: `(None, 64)`
- **Output shape**: `(None, 1)`
- **Activation**: Sigmoid
- **Parameters**: 64 × 1 + 1 = 65

**Purpose**: Produces final prediction in [0,1] range (rescaled to [0,10] post-prediction).

## Total Model Statistics

- **Total parameters**: 632,849
- **Trainable parameters**: 632,721
- **Non-trainable parameters**: 128 (batch normalization)
- **Model size**: ~2.41 MB

## Mathematical Formulation

### Forward Pass

1. **Embedding**:
   ```
   E = W_emb[x_text]
   where E ∈ ℝ^(batch × 100 × 150)
   ```

2. **BiLSTM**:
   ```
   h_forward = LSTM_forward(E)
   h_backward = LSTM_backward(E)
   h_bilstm = [h_forward; h_backward] ∈ ℝ^(batch × 128)
   ```

3. **Concatenation**:
   ```
   h_concat = [h_bilstm; x_struct] ∈ ℝ^(batch × 1431)
   ```

4. **MLP**:
   ```
   h_1 = σ(W_1 · h_concat + b_1) ∈ ℝ^(batch × 64)
   h_1_drop = Dropout(h_1, p=0.2)
   h_1_norm = BatchNorm(h_1_drop)
   ```

5. **Output**:
   ```
   ŷ = σ(W_2 · h_1_norm + b_2) ∈ ℝ^(batch × 1)
   ```

### Loss Function

Mean Squared Error (MSE):
```
L = (1/n) Σᵢ (yᵢ - ŷᵢ)²
```

where:
- yᵢ: true score (normalized to [0,1])
- ŷᵢ: predicted score ([0,1])
- n: batch size

## Design Rationale

### Why BiLSTM over Alternatives?

| Architecture | Pros | Cons | Decision |
|--------------|------|------|----------|
| **BiLSTM** ✓ | • Captures sequential dependencies<br>• Bidirectional context<br>• Handles variable-length input | • More parameters<br>• Slower training | **Chosen** - Best for text understanding |
| Unidirectional LSTM | • Fewer parameters<br>• Faster | • Only forward context | Inferior - Misses backward context |
| CNN | • Fast training<br>• Parallel processing | • Local patterns only<br>• Weak on long dependencies | Not suitable - Reviews need full context |
| MLP | • Simplest<br>• Fastest | • No sequential understanding<br>• Fixed input size | Not suitable - Ignores word order |

### Why Sigmoid Activation in Hidden Layer?

Originally proposed in written exam. Alternatives:
- **ReLU**: More common, faster training, avoids vanishing gradients
- **Tanh**: Similar to sigmoid but centered at 0
- **Sigmoid**: Keeps values in [0,1], but can cause vanishing gradients

**Future improvement**: Consider ReLU for hidden layers.

### Why Single Dense Layer in MLP?

- Keeps model simpler (fewer parameters)
- Main feature learning happens in BiLSTM
- MLP primarily fuses representations
- Regularization (dropout + batch norm) prevents overfitting

## Hyperparameter Selection

### Search Space

| Hyperparameter | Values Tested | Best Value | Rationale |
|----------------|---------------|------------|-----------|
| Embedding dim | 50, 100, 150 | **150** | Richer word representations |
| LSTM units | 16, 32 | **16** | Sufficient capacity, avoids overfitting |
| Dropout | 0.1, 0.2 | **0.2** | Better regularization |
| Learning rate | 0.0001, 0.0005 | **0.0001** | More stable convergence |

### Search Method: Manual Randomized Search

**Why not GridSearchCV?**
- KerasRegressor incompatibility with multi-input models
- Keras functional API not directly compatible with sklearn wrappers
- Manual approach provides better control

**Implementation**:
- 2-fold cross-validation per configuration
- 5 random samples from hyperparameter space
- Average validation MSE as selection criterion

## Training Configuration

- **Optimizer**: Adam
  - Adaptive learning rates
  - Combines momentum and RMSprop
  - Default β₁=0.9, β₂=0.999, ε=10⁻⁷

- **Batch size**: 64
  - Balances memory usage and gradient stability
  - Fits in standard GPU memory

- **Epochs**: 2 (limited for quick tuning)
  - Production model would use more epochs with early stopping

- **Weight initialization**: 
  - Glorot/Xavier (Keras default for Dense layers)
  - Suitable for sigmoid and tanh activations

## Data Flow Example

```python
# Input
text = "Great hotel excellent service"
hotel_name = "Hilton London"
review_date = "5/2/2017"
reviewer_reviews = 3
hotel_reviews = 1500

# Preprocessing
text_ids = [245, 892, 1043, 567]  # After tokenization
text_padded = [245, 892, 1043, 567, 0, 0, ..., 0]  # Length 100

hotel_encoded = [0, 0, ..., 1, ..., 0]  # One-hot, length 1298
month, day, year = 5, 2, 2017
month_scaled = 0.42  # (5-1)/(12-1)
day_scaled = 0.03    # (2-1)/(31-1)
year_scaled = 0.67   # Scaled to training range

structured = [hotel_encoded, reviewer_scaled, hotel_scaled, 
              day_scaled, month_scaled, year_scaled]  # Length 1303

# Model output
output_normalized = 0.73  # From sigmoid
output_rescaled = 7.3     # × 10 for final score
```

## Memory Requirements

### Training
- **Batch size**: 64
- **Text input**: 64 × 100 × 4 bytes = 25.6 KB
- **Structured input**: 64 × 1303 × 4 bytes = 333.6 KB
- **Embeddings**: 64 × 100 × 150 × 4 bytes = 3.84 MB
- **Activations**: ~5-10 MB
- **Model parameters**: 2.41 MB
- **Total**: ~15-20 MB per batch

### Inference
- Single sample: ~50 KB
- Suitable for real-time prediction

## Performance Characteristics

- **Training time**: ~20-25 seconds per epoch (GPU)
- **Inference time**: ~17ms per sample
- **Throughput**: ~60 samples/second

## Future Architecture Improvements

1. **Add Attention Mechanism**
   ```
   BiLSTM → Attention → MLP
   ```
   - Allows model to focus on important words
   - Interpretable attention weights

2. **Use Pre-trained Embeddings**
   - Word2Vec, GloVe, or BERT
   - Transfer learning from larger corpora

3. **Add Residual Connections**
   ```
   x → Dense → x + f(x)
   ```
   - Helps with gradient flow
   - Enables deeper networks

4. **Multi-task Learning**
   - Predict both score and review type
   - Shared representations

5. **Ensemble Methods**
   - Train multiple models
   - Average predictions

## Reproducibility

Random seeds set:
```python
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
```

Ensures deterministic behavior for:
- Weight initialization
- Dropout masks
- Data shuffling
