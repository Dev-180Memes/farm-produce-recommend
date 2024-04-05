from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

class FarmRecommendationSystem:
    def __init__(self, data):
        self.data = data
        self.scaler = MinMaxScaler()

    def preprocess_data(self):
        self.data.fillna(0, inplace=True)
        numeric_columns = ['Value']
        self.data[numeric_columns] = self.scaler.fit_transform(self.data[numeric_columns])

    def calculate_similarity(self, farmer_value, item_value):
        if farmer_value and item_value:
            similarity_score = 1 - abs(farmer_value - item_value) / max(farmer_value, item_value)
        else:
            similarity_score = 0
        return similarity_score

    def get_recommendations(self, farmer_profile, top_n=5):
        farmer_item = farmer_profile['Item']
        farmer_value = farmer_profile['Value']
        filtered_data = self.data[self.data['Item'] == farmer_item]
        filtered_data['similarity_score'] = filtered_data['Value'].apply(
            lambda item_value: self.calculate_similarity(farmer_value, item_value))
        unique_items = filtered_data.groupby('Item', as_index=False).max()
        recommendations = unique_items.sort_values(by='similarity_score', ascending=False).head(top_n)
        return recommendations[['Element', 'Item', 'similarity_score']]

# Global variable for the recommendation system instance
recommendation_system = None

@app.route('/load_data', methods=['POST'])
def load_data():
    global recommendation_system
    data = pd.read_csv("/dataset.csv")
    recommendation_system = FarmRecommendationSystem(data)
    recommendation_system.preprocess_data()
    return jsonify(success=True)

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    farmer_profile = request.json
    if recommendation_system is None:
        return jsonify(error="Data not loaded"), 400
    recommendations = recommendation_system.get_recommendations(farmer_profile)
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
