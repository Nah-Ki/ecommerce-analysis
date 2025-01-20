from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def generate_lookalike_model(data, top_n=3):
    features = data.groupby('CustomerID').agg({
        'TotalValue': 'sum',
        'Quantity': 'sum',
        'Price': 'mean'
    }).reset_index()
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features.iloc[:, 1:])
    similarity = cosine_similarity(features_scaled)
    return similarity, features
