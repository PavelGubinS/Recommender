"""
Recommender
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class StudyRecommender:
    def __init__(self):
        self.materials_data = [
            {
                "id": 1,
                "title": "Introduction to Python",
                "description": "Python concepts",
                "category": "Programming",
                "tags": "python,basics",
            },
            {
                "id": 2,
                "title": "ML with scikit-learn",
                "description": "ML methods",
                "category": "ML",
                "tags": "ml,scikit",
            },
            {
                "id": 3,
                "title": "NLP Text Processing",
                "description": "Text processing",
                "category": "NLP",
                "tags": "nlp,text",
            },
            {
                "id": 4,
                "title": "Pandas Analysis",
                "description": "Data tools",
                "category": "Data",
                "tags": "pandas,data-analysis",
            },
            {
                "id": 5,
                "title": "Data Visualization",
                "description": "Charts and diagrams",
                "category": "Data",
                "tags": "matplotlib,visualization",
            },
            {
                "id": 6,
                "title": "Deep Learning Basics",
                "description": "Neural networks",
                "category": "ML",
                "tags": "deep-learning,neural-networks",
            },
            {
                "id": 7,
                "title": "Recommendation Systems",
                "description": "Recommendation systems",
                "category": "ML",
                "tags": "recommendation-systems",
            },
            {
                "id": 8,
                "title": "Python for Data Science",
                "description": "Data tools",
                "category": "Programming",
                "tags": "python,data-science",
            },
            {
                "id": 9,
                "title": "C++ Programming",
                "description": "C++ concepts",
                "category": "Programming",
                "tags": "c++,programming",
            },
            {
                "id": 10,
                "title": "Time Series Analysis",
                "description": "Forecasting methods",
                "category": "Data",
                "tags": "time-series,forecasting",
            },
        ]

        self.data = pd.DataFrame(self.materials_data)
        self.data["combined_text"] = self._create_combined_text()

        self.vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2), token_pattern=r"\b\w+\b"
        )

        texts = self.data["combined_text"]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        print(f"✅ Инициализирован с {len(self.data)} материалами")

    def _create_combined_text(self):
        return (
            self.data["title"].fillna("")
            + " "
            + self.data["description"].fillna("")
            + " "
            + self.data["tags"].fillna("")
        )

    def recommend(self, query, top_n=3):
        if len(self.data) == 0:
            return []

        query_vector = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        top_idxs = sims.argsort()[::-1][:top_n]
        top_idxs = [i for i in top_idxs if sims[i] > 0]

        res = []
        for i in top_idxs:
            res.append(
                {
                    "id": self.data.iloc[i]["id"],
                    "title": self.data.iloc[i]["title"],
                    "description": self.data.iloc[i]["description"],
                    "category": self.data.iloc[i]["category"],
                    "tags": self.data.iloc[i]["tags"],
                    "similarity": float(sims[i]),
                }
            )
        return res

    def search_by_category(self, category):
        category_mask = self.data["category"].str.contains(
            category, case=False, na=False
        )
        return self.data[category_mask].to_dict("records")

    def search_by_tag(self, tag):
        tags_mask = self.data["tags"].str.contains(
            tag, case=False, na=False
        )
        return self.data[tags_mask].to_dict("records")

    def get_all_materials(self):
        return self.data.to_dict("records")


if __name__ == "__main__":
    recommender = StudyRecommender()
    print("\nРекомендации по запросу 'python prog':")
    recs = recommender.recommend("python programming", 3)
    for rec in recs:
        print(f"  {rec['title']} - {rec['similarity']:.2f}")
    print("\nМатериалы по 'ML':")
    ml_list = recommender.search_by_category("ML")
    for m in ml_list:
        print(f"  {m['title']}")
