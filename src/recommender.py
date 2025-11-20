"""
Recommender
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class StudyRecommender:
    def __init__(self):
        # Данные прямо в коде (без чтения из CSV)
        self.materials_data = [
            {
                "id": 1,
                "title": "Introduction to Python",
                "description": "Concepts of Python programming",
                "category": "Programming",
                "tags": "python,basics"
            },
            {
                "id": 2,
                "title": "ML with scikit-learn",
                "description": "ML methods description",
                "category": "ML",
                "tags": "ml,scikit"
            },
            {
                "id": 3,
                "title": "NLP Text Processing",
                "description": "Text data processing",
                "category": "NLP",
                "tags": "nlp,text"
            },
            {
                "id": 4,
                "title": "Pandas Analysis",
                "description": "Data analysis tools",
                "category": "Data",
                "tags": "pandas,data-analysis"
            },
            {
                "id": 5,
                "title": "Data Visualization",
                "description": "Charts and diagrams",
                "category": "Data",
                "tags": "matplotlib,visualization"
            },
            {
                "id": 6,
                "title": "Deep Learning Basics",
                "description": "Neural networks",
                "category": "ML",
                "tags": "deep-learning,neural-networks"
            },
            {
                "id": 7,
                "title": "Recommendation Systems",
                "description": "Recommendation systems",
                "category": "ML",
                "tags": "recommendation-systems"
            },
            {
                "id": 8,
                "title": "Python for Data Science",
                "description": "Data analysis tools",
                "category": "Programming",
                "tags": "python,data-science"
            },
            {
                "id": 9,
                "title": "C++ Programming",
                "description": "C++ language concepts",
                "category": "Programming",
                "tags": "c++,programming"
            },
            {
                "id": 10,
                "title": "Time Series Analysis",
                "description": "Forecasting methods",
                "category": "Data",
                "tags": "time-series,forecasting"
            }
        ]

        # Создаем DataFrame из данных
        self.data = pd.DataFrame(self.materials_data)

        # Создаем combined_text для векторизации
        self.data['combined_text'] = self._create_combined_text()

        # Инициализируем векторизатор
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            token_pattern=r'\b\w+\b'
        )

        # Создаем TF-IDF матрицу
        if len(self.data) > 0:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.data['combined_text'])

        print(f"✅ Инициализирована система рекомендаций с {len(self.data)} материалами")

    def _create_combined_text(self):
        """Создает объединенный текст для векторизации"""
        return (self.data['title'].fillna('') + ' ' +
                self.data['description'].fillna('') + ' ' +
                self.data['tags'].fillna(''))

    def recommend(self, query, top_n=3):
        """Рекомендует материалы по запросу"""
        if len(self.data) == 0:
            return []

        # Векторизуем запрос
        query_vector = self.vectorizer.transform([query])

        # Вычисляем косинусное сходство
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Находим топ-N наиболее похожих материалов
        top_indices = similarities.argsort()[::-1][:top_n]
        top_indices = [i for i in top_indices if similarities[i] > 0]

        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'id': self.data.iloc[idx]['id'],
                'title': self.data.iloc[idx]['title'],
                'description': self.data.iloc[idx]['description'],
                'category': self.data.iloc[idx]['category'],
                'tags': self.data.iloc[idx]['tags'],
                'similarity': float(similarities[idx])
            })

        return recommendations

    def search_by_category(self, category):
        """Ищет материалы по категории"""
        results = self.data[self.data['category'].str.contains(category, case=False, na=False)]
        return results.to_dict('records')

    def search_by_tag(self, tag):
        """Ищет материалы по тегу"""
        results = self.data[self.data['tags'].str.contains(tag, case=False, na=False)]
        return results.to_dict('records')

    def get_all_materials(self):
        """Возвращает все материалы"""
        return self.data.to_dict('records')


# Пример использования
if __name__ == "__main__":
    recommender = StudyRecommender()

    # Тест рекомендаций
    print("\nРекомендации для 'python programming':")
    recommendations = recommender.recommend("python programming", 3)
    for rec in recommendations:
        print(f"  {rec['title']} - {rec['similarity']:.2f}")

    # Тест поиска по категории
    print("\nМатериалы по категории 'ML':")
    ml_materials = recommender.search_by_category("ML")
    for material in ml_materials:
        print(f"  {material['title']}")
