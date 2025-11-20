"""
Recommender
"""

from recommender import StudyRecommender

def main():
    print("=== Study Recommender System ===")
    
    # Инициализация системы
    recommender = StudyRecommender()
    
    print(f"Загружено {len(recommender.data)} учебных материалов")
    
    # Примеры использования
    print("\n1. Рекомендации по запросу 'python':")
    recommendations = recommender.recommend("python", 3)
    for rec in recommendations:
        print(f"   - {rec['title']} (similarity: {rec['similarity']:.2f})")
    
    print("\n2. Поиск по категории 'ML':")
    ml_materials = recommender.search_by_category("ML")
    for material in ml_materials:
        print(f"   - {material['title']}")
    
    print("\n3. Поиск по тегу 'data':")
    data_materials = recommender.search_by_tag("data")
    for material in data_materials:
        print(f"   - {material['title']}")

if __name__ == "__main__":
    main()
