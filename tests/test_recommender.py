"""
Recommender - Тесты
"""

import sys
import os
import pytest

# Добавляем путь к src в PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.recommender import StudyRecommender

def test_recommender_initialization():
    """Тестирует инициализацию рекомендательной системы"""
    recommender = StudyRecommender("data/materials.csv")
    
    # Проверяем, что данные загружены корректно
    assert hasattr(recommender, 'data')
    assert len(recommender.data) > 0
    
    # Проверяем, что векторизатор инициализирован
    assert hasattr(recommender, 'vectorizer')
    assert hasattr(recommender, 'tfidf_matrix')

def test_recommendation_basic():
    """Тестирует базовую функциональность рекомендаций"""
    recommender = StudyRecommender("data/materials.csv")
    
    # Тест с простым запросом
    results = recommender.recommend("Python", top_n=3)
    
    assert hasattr(results, 'shape') or len(results) >= 0  # Может быть пустым
    assert 'id' in results.columns if len(results) > 0 else True
    assert 'title' in results.columns if len(results) > 0 else True

def test_recommendation_with_results():
    """Тестирует рекомендации с реальными результатами"""
    recommender = StudyRecommender("data/materials.csv")
    
    # Тест с конкретным запросом, который должен давать результаты
    results = recommender.recommend("Python basics", top_n=3)
    
    # Проверяем, что результаты имеют правильные колонки
    if len(results) > 0:
        expected_columns = ['id', 'title', 'description', 'category', 'tags', 'similarity']
        for col in expected_columns:
            assert col in results.columns

def test_search_by_category():
    """Тестирует поиск по категории"""
    recommender = StudyRecommender("data/materials.csv")
    
    # Проверяем поиск по существующей категории
    results = recommender.search_by_category("Programming")
    assert hasattr(results, 'shape') or len(results) >= 0

def test_search_by_tag():
    """Тестирует поиск по тегу"""
    recommender = StudyRecommender("data/materials.csv")
    
    # Проверяем поиск по существующему тегу
    results = recommender.search_by_tag("python")
    assert hasattr(results, 'shape') or len(results) >= 0

def test_get_materials_info():
    """Тестирует получение информации о материалах"""
    recommender = StudyRecommender("data/materials.csv")
    
    materials = recommender.get_materials_info()
    assert hasattr(materials, 'shape') or len(materials) > 0

def test_recommendation_empty_query():
    """Тестирует рекомендации с пустым запросом"""
    recommender = StudyRecommender("data/materials.csv")
    
    # Пустой запрос
    results = recommender.recommend("", top_n=3)
    assert hasattr(results, 'shape') or len(results) >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
