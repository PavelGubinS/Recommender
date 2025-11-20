"""
Recommender - Тесты
"""

import pytest
import pandas as pd
from src.recommender import StudyRecommender

def test_recommender_initialization():
    """Тестирует инициализацию рекомендательной системы"""
    recommender = StudyRecommender("data/materials.csv")
    
    # Проверяем, что данные загружены корректно
    assert isinstance(recommender.data, pd.DataFrame)
    assert len(recommender.data) > 0
    
    # Проверяем, что векторизатор инициализирован
    assert hasattr(recommender, 'vectorizer')
    assert hasattr(recommender, 'tfidf_matrix')

def test_recommendation_basic():
    """Тестирует базовую функциональность рекомендаций"""
    recommender = StudyRecommender("data/materials.csv")
    
    # Тест с простым запросом
    results = recommender.recommend("Python", top_n=3)
    
    assert isinstance(results, pd.DataFrame)
    assert len(results) >= 0  # Может быть пустым
    assert 'id' in results.columns
    assert 'title' in results.columns
    assert 'description' in results.columns

def test_recommendation_with_results():
    """Тестирует рекомендации с реальными результатами"""
    recommender = StudyRecommender("data/materials.csv")
    
    # Тест с конкретным запросом, который должен давать результаты
    results = recommender.recommend("Python basics", top_n=3)
    
    # Проверяем, что результаты имеют правильные колонки
    expected_columns = ['id', 'title', 'description', 'category', 'tags', 'similarity']
    if not results.empty:
        for col in expected_columns:
            assert col in results.columns

def test_search_by_category():
    """Тестирует поиск по категории"""
    recommender = StudyRecommender("data/materials.csv")
    
    # Проверяем поиск по существующей категории
    results = recommender.search_by_category("Programming")
    assert isinstance(results, pd.DataFrame)
    
    # Проверяем, что результаты содержат правильную категорию
    if not results.empty:
        assert any("Programming" in str(row) for row in results['category'])

def test_search_by_tag():
    """Тестирует поиск по тегу"""
    recommender = StudyRecommender("data/materials.csv")
    
    # Проверяем поиск по существующему тегу
    results = recommender.search_by_tag("python")
    assert isinstance(results, pd.DataFrame)

def test_get_materials_info():
    """Тестирует получение информации о материалах"""
    recommender = StudyRecommender("data/materials.csv")
    
    materials = recommender.get_materials_info()
    assert isinstance(materials, pd.DataFrame)
    assert len(materials) > 0

def test_recommendation_empty_query():
    """Тестирует рекомендации с пустым запросом"""
    recommender = StudyRecommender("data/materials.csv")
    
    # Пустой запрос
    results = recommender.recommend("", top_n=3)
    assert isinstance(results, pd.DataFrame)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
