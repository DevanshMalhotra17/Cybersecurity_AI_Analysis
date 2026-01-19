# ATP Tennis Match Winner Predictor

Machine learning model to predict tennis match winners using Random Forest classification.

## Dataset
**Source:** ATP Tennis 2000-2025 (Kaggle)  
**Size:** 66,682 professional tennis matches  
**Features Used:**
- Player rankings (Rank_1, Rank_2)
- Court surface (Hard, Clay, Grass, Carpet)
- Court type (Indoor, Outdoor)

## Model Details
**Algorithm:** Random Forest Classifier
- 100 trees
- Max depth: 10
- Train/test split: 80/20

**Performance:**
- Training Accuracy: ~67%
- Testing Accuracy: ~65%
- Overfitting Gap: ~2% (minimal)

## Installation

```bash
pip install pandas numpy scikit-learn
```

## Usage

```bash
python tennis_predictor.py
```

## Why Random Forest?

1. **Handles Non-Linear Relationships:** Tennis match outcomes depend on complex interactions between rankings and surface types
2. **Resistant to Overfitting:** Ensemble method reduces variance compared to single decision trees
3. **Feature Importance:** Shows which factors matter most (rankings vs surface)
4. **Works with Limited Features:** Performs well even with just 5 features
5. **Interpretable:** Easy to explain decision-making process

## Model Biases

1. **Ranking Bias:** Model heavily relies on rankings, may not account for players in poor form or rising stars
2. **Surface Bias:** Dataset may have more Hard court matches than Clay/Grass
3. **Historical Bias:** Data from 2000-2025 may not reflect modern playing styles
4. **Selection Bias:** Only includes ATP matches, excludes qualifiers/wildcards with missing rankings
5. **Winner Bias:** No information on close matches vs blowouts (all wins treated equally)

## Limitations & Assumptions

### Limitations:
- Cannot predict matches with unranked players
- Ignores player injuries or recent form
- No head-to-head history between players
- Doesn't account for match importance (Grand Slam vs regular tournament)
- Missing data on player fatigue or travel schedules

### Assumptions:
- Rankings accurately reflect current player skill level
- Surface type impacts all players equally
- Past performance predicts future results
- All matches have equal weight regardless of tournament

## Suggestions for Improvement

1. **Add Recent Form:** Include last 5-10 matches win rate for momentum
2. **Head-to-Head History:** Add previous matchups between specific players
3. **Surface-Specific Stats:** Track player performance on each surface type (clay specialists, etc.)
4. **Tournament Importance:** Weight Grand Slams differently than regular tournaments
5. **Neural Networks:** Try deep learning for more complex pattern recognition
6. **Score Margin Data:** Include how decisively matches were won
7. **Time-Series Analysis:** Track player momentum and trajectory over time

## Project Structure

```
.
├── atp_tennis.csv          # Dataset
├── tennis_predictor.py     # Main ML model
├── README.md
└── requirements.txt        # Dependencies
```

## Statistical Analysis

The model calculates comprehensive statistics for player rankings:
- Mean, Median, Mode
- Min, Max, Range
- Variance, Standard Deviation

These metrics help understand the distribution of player skill levels in the dataset.

## Example Prediction

```python
# Predict: Rank 10 vs Rank 50 on Hard court, Outdoor
Predict(model, rank1=10, rank2=50, surface='Hard', court='Outdoor')
# Expected: Player 1 wins with ~70% confidence
```

## AI in Tennis (Independent Research)

Modern tennis uses AI in various ways:
- **Hawk-Eye:** Ball tracking system for line calls using computer vision
- **IBM Watson:** Analyzes player performance and generates match insights
- **Serve Analysis:** ML models predict serve direction and speed
- **Injury Prevention:** AI monitors player biomechanics to prevent injuries