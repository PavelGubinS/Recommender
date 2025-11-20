from src.recommender import StudyRecommender

recommender = StudyRecommender("data/materials.csv")
results = recommender.recommend("Python for beginners")
print(results)