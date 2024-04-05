from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)


class FarmRecommendationSystem:
    def __init__(self):
        self.data = pd.read_csv("./dataset.csv")
        self.scaler = MinMaxScaler()
        self.tfidf_vectorizer = TfidfVectorizer()

    def preprocess_data(self):
        self.data.fillna(0, inplace=True)
        numeric_columns = ['Value']
        self.data[numeric_columns] = self.scaler.fit_transform(self.data[numeric_columns])
        self.item_name_vectors = self.tfidf_vectorizer.fit_transform(self.data['Item'])

    @staticmethod
    def calculate_similarity(farmer_value, item_value):
        # Calculate numeric similarity
        if farmer_value and item_value:
            numeric_similarity = np.dot([farmer_value], [item_value]) / (
                    np.linalg.norm([farmer_value]) * np.linalg.norm([item_value]))
        else:
            numeric_similarity = 0
        return numeric_similarity

    def get_recommendations(self, farmer_profile, top_n=5):
        farmer_item = farmer_profile['Item']
        farmer_value = float(farmer_profile['Value'])

        # Calculate textual similarity
        farmer_item_vector = self.tfidf_vectorizer.transform([farmer_item])
        textual_similarity_scores = cosine_similarity(farmer_item_vector, self.item_name_vectors).flatten()

        # Calculate value similarity
        value_similarity_scores = self.data['Value'].apply(
            lambda item_value: self.calculate_similarity(farmer_value, item_value))

        # Combine similarities (simple average or weighted average could be used)
        combined_similarity_scores = (textual_similarity_scores + value_similarity_scores) / 2

        # Sort data based on combined similarity scores and get top_n items
        self.data['similarity_score'] = combined_similarity_scores
        recommendations = self.data.sort_values(by='similarity_score', ascending=False).head(top_n)

        return recommendations[['Element', 'Item', 'similarity_score']]


recommendation_system = FarmRecommendationSystem()
recommendation_system.preprocess_data()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    farmer_profile = request.json
    if recommendation_system is None:
        return jsonify(error="Data not loaded"), 400
    recommendations = recommendation_system.get_recommendations(farmer_profile)
    # Make sure recommendation only contains unique items
    recommendations = recommendations.drop_duplicates(subset=['Item'])
    return jsonify(recommendations.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(debug=True)
