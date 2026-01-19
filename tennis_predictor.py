import pandas as pd
import numpy as np

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

if __name__ == "__main__":
    df = LoadData('atp_tennis.csv')
    print(f"\nPlayer 1 wins: {df['Target'].sum()}")
    print(f"Player 2 wins: {len(df) - df['Target'].sum()}")