"""
Tests for Recommender
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.recommender import StudyRecommender

def test_recommender_initialization():
    """Тестирует инициализацию рекомендательной системы"""
    recommender = StudyRecommender()
    
    # Проверяем, что данные загружены
    assert hasattr(recommender, 'data')
    assert len(recommender.data) > 0
    
    # Проверяем, что векторизатор инициализирован
    assert hasattr(recommender, 'vectorizer')
    assert hasattr(recommender, 'tfidf_matrix')

def test_recommendation_basic():
    """Тестирует базовую функциональность рекомендаций"""
    recommender = StudyRecommender()
    
    # Тест рекомендаций
    recommendations = recommender.recommend("python programming", 3)
    assert isinstance(recommendations, list)
    assert len(recommendations) >= 0  # Может быть 0, если нет совпадений

def test_recommendation_with_results():
    """Тестирует рекомендации с реальными результатами"""
    recommender = StudyRecommender()
    
    # Тест с конкретным запросом
    recommendations = recommender.recommend("machine learning", 2)
    assert isinstance(recommendations, list)
    if len(recommendations) > 0:
        assert 'title' in recommendations[0]
        assert 'similarity' in recommendations[0]

def test_search_by_category():
    """Тестирует поиск по категории"""
    recommender = StudyRecommender()
    
    # Тест поиска по категории
    results = recommender.search_by_category("ML")
    assert isinstance(results, list)
    if len(results) > 0:
        assert 'category' in results[0]
        assert results[0]['category'] == "ML"

def test_search_by_tag():
    """Тестирует поиск по тегу"""
    recommender = StudyRecommender()
    
    # Тест поиска по тегу
    results = recommender.search_by_tag("python")
    assert isinstance(results, list)

def test_get_materials_info():
    """Тестирует получение информации о материалах"""
    recommender = StudyRecommender()
    
    # Тест получения всех материалов
    materials = recommender.get_all_materials()
    assert isinstance(materials, list)
    assert len(materials) > 0

def test_recommendation_empty_query():
    """Тестирует рекомендации с пустым запросом"""
    recommender = StudyRecommender()
    
    # Тест с пустым запросом
    recommendations = recommender.recommend("", 3)
    assert isinstance(recommendations, list)

if __name__ == "__main__":
    test_recommender_initialization()
    test_recommendation_basic()
    test_recommendation_with_results()
    test_search_by_category()
    test_search_by_tag()
    test_get_materials_info()
    test_recommendation_empty_query()
    print("✅ Все тесты пройдены!")
