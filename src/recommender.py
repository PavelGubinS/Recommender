"""
Recommender - Логика рекомендаций
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class StudyRecommender:
    """
    Рекомендательная система для учебных материалов
    """
    
    def __init__(self, data_path):
        """
        Инициализация рекомендательной системы
        
        Args:
            data_path (str): Путь к CSV файлу с материалами
        """
        self.data = pd.read_csv(data_path)
        # Создаем комбинированный текст для векторизации
        self.data['combined_text'] = (
            self.data['title'].fillna('') + ' ' + 
            self.data['description'].fillna('') + ' ' + 
            self.data['tags'].fillna('')
        )
        
        # Инициализируем векторизатор TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Создаем TF-IDF матрицу
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['combined_text'])
        
        print(f"✅ Инициализирована система рекомендаций с {len(self.data)} материалами")
    
    def recommend(self, query, top_n=3):
        """
        Рекомендует материалы по запросу
        
        Args:
            query (str): Поисковый запрос пользователя
            top_n (int): Количество рекомендаций
            
        Returns:
            pandas.DataFrame: Топ N рекомендованных материалов
        """
        # Векторизуем запрос
        query_vec = self.vectorizer.transform([query])
        
        # Вычисляем косинусное сходство
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Получаем индексы топ N материалов
        top_indices = similarities.argsort()[::-1][:top_n]
        
        # Фильтруем по минимальному уровню сходства (опционально)
        min_similarity = 0.1
        filtered_indices = [i for i in top_indices if similarities[i] >= min_similarity]
        
        # Возвращаем результаты
        if len(filtered_indices) == 0:
            return pd.DataFrame()
            
        results = self.data.iloc[filtered_indices].copy()
        results['similarity'] = [similarities[i] for i in filtered_indices]
        
        # Сортируем по сходству
        results = results.sort_values('similarity', ascending=False)
        
        return results[['id', 'title', 'description', 'category', 'tags', 'similarity']]
    
    def get_materials_info(self):
        """
        Возвращает информацию о всех материалах
        
        Returns:
            pandas.DataFrame: Все учебные материалы
        """
        return self.data[['id', 'title', 'description', 'category', 'tags']]
    
    def search_by_category(self, category):
        """
        Ищет материалы по категории
        
        Args:
            category (str): Категория для поиска
            
        Returns:
            pandas.DataFrame: Материалы в указанной категории
        """
        return self.data[self.data['category'].str.contains(category, case=False, na=False)]
    
    def search_by_tag(self, tag):
        """
        Ищет материалы по тегу
        
        Args:
            tag (str): Тег для поиска
            
        Returns:
            pandas.DataFrame: Материалы с указанным тегом
        """
        return self.data[self.data['tags'].str.contains(tag, case=False, na=False)]
