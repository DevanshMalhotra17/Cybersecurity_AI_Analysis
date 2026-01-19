import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def LoadData(filepath):
    # Load only what we need
    columns_needed = ['Rank_1', 'Rank_2', 'Surface', 'Court', 'Winner', 'Player_1', 'Player_2']
    
    df = pd.read_csv(filepath, usecols=columns_needed)
    print(f"Loaded {len(df)} matches")
    
    # Remove invalid data
    df = df.dropna()
    df = df[df['Rank_1'] != -1]
    df = df[df['Rank_2'] != -1]
    
    print(f"Clean dataset: {len(df)} matches")
    
    # Create target (1 if Player_1 won, 0 if Player_2 won)
    df['Target'] = (df['Winner'] == df['Player_1']).astype(int)
    
    return df

def CreateFeatures(df):
    # Rank difference
    df['Rank_Diff'] = df['Rank_1'] - df['Rank_2']
    
    # Surface (Hard, Clay, Grass, etc.)
    surface = LabelEncoder()
    df['Surface'] = surface.fit_transform(df['Surface'])
    
    # Court (Indoor, Outdoor)
    court = LabelEncoder()
    df['Court'] = court.fit_transform(df['Court'])
    
    return df

def TrainModel(df):
    feature_cols = ['Rank_1', 'Rank_2', 'Rank_Diff', 'Surface', 'Court']
    X = df[feature_cols].values
    y = df['Target'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining on {len(X_train)} matches")
    print(f"Testing on {len(X_test)} matches")
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\nTraining Accuracy: {train_acc*100:.2f}%")
    print(f"Testing Accuracy: {test_acc*100:.2f}%")
    print(f"Overfitting Gap: {(train_acc - test_acc)*100:.2f}%")
    
    return model

if __name__ == "__main__":
    df = LoadData('atp_tennis.csv')
    df = CreateFeatures(df)
    model = TrainModel(df)
    print("\nModel trained successfully!")