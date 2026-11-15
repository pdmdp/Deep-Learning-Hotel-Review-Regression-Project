# Hotel review score regression

This project predicts hotel review scores from review text and hotel/reviewer metadata. A bidirectional LSTM reads the text, a second branch supplies structured features, and their combined prediction is compared with a training-mean baseline.

The recorded neural model has an RMSE of **approximately 1.819 score points**, compared with **2.056** for always predicting the training mean on the same split.

[Notebook](Deep-Learning-Hotel-Review-Regression-Project.ipynb) · [Architecture and training](ARCHITECTURE.md) · [Project design](docs/EXAM_DL.pdf)

## How the model uses both inputs

The text branch learns word embeddings and reads each review in both directions. Its representation is concatenated with hotel indicators and scaled numerical/date features, then passed through a dense network to predict a score.

The selected model uses 150-dimensional embeddings, 16 LSTM units per direction, a 64-unit dense layer, dropout and batch normalization. Review sequences are limited to 100 tokens. Scores are divided by ten during training and converted back for the RMSE comparison.

The 13,772 reviews are split into 9,640 training, 2,066 validation and 2,066 test examples with seed 42. The search samples five configurations with two-fold cross-validation and two epochs per configuration. `Review_Type` is excluded because it is directly related to the target score.

## Results

![Test error for the recorded neural model and the training-mean baseline](assets/test-comparison.png)

| Model | MSE on normalized target | RMSE in score points |
| :--- | ---: | ---: |
| Training-mean baseline | 0.04226 | 2.056 |
| BiLSTM + structured features | 0.0331 | approximately 1.819 |

The neural score is a saved coursework result; the baseline was computed later using the same split and only the training mean. The neural MSE was printed to four decimal places, so its converted RMSE is approximate. [The comparison record](results/comparison.json) contains the values and split details.

The combined model improves on this simple baseline. The experiment does not isolate how much comes from text versus metadata; separate text-only and metadata-only models would answer that question.

## Contribution

**Matteo Pietro Di Pilato** · Deep Learning coursework, BSc in Artificial Intelligence. The project implements the dual-input model, preprocessing, configuration search and evaluation.

## Limits and next steps

The random review split does not test unseen hotels or future dates. Preprocessing is shared across the inner validation folds, padding and unknown words use the same unmasked index, and the two-epoch search is small. The trained weights were not saved.

The next comparison should fit preprocessing inside each search fold, add text-only and metadata-only baselines, and evaluate on a hotel-disjoint or chronological split. [Architecture notes](ARCHITECTURE.md#evaluation-limits) explain the current limitations.

## Run

```bash
git clone https://github.com/pdmdp/Deep-Learning-Hotel-Review-Regression-Project.git
cd Deep-Learning-Hotel-Review-Regression-Project
python -m pip install -r requirements.txt
python -m jupyter lab Deep-Learning-Hotel-Review-Regression-Project.ipynb
```

The notebook loads the included course CSV and needs a compatible TensorFlow environment. To reproduce the baseline and figure without TensorFlow:

```bash
python -m pip install -r requirements-baseline.txt
python scripts/evaluate_baseline.py
```

[Reproduction notes](ARCHITECTURE.md#reproduction-notes) describe the available environment and saved results. The [MIT license](LICENSE) covers project code; course data and materials retain their own terms.
