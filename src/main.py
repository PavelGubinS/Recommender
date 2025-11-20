#!/usr/bin/env python3
"""
Study Material Recommender - Главный файл проекта
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommender import StudyRecommender

def main():
    """Основная функция приложения"""
    print("🎓 Добро пожаловать в Study Material Recommender!")
    print("=" * 50)
    
    # Создаем рекомендательную систему
    recommender = StudyRecommender("data/materials.csv")
    
    # Примеры запросов для демонстрации
    examples = [
        "Python для начинающих",
        "Машинное обучение с использованием scikit-learn",
        "Анализ данных с помощью Pandas",
        "Обработка текста в NLP"
    ]
    
    print("\n🔍 Примеры запросов:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    
    print("\nВведите свой запрос (или 'quit' для выхода):")
    
    while True:
        user_input = input("\n> ").strip()
        
        if user_input.lower() in ['quit', 'выход', 'exit']:
            print("👋 До свидания!")
            break
            
        if not user_input:
            print("Пожалуйста, введите запрос")
            continue
            
        print(f"\n🔍 Поиск материалов по запросу: '{user_input}'")
        results = recommender.recommend(user_input, top_n=3)
        
        if len(results) == 0:
            print("❌ Не удалось найти подходящие материалы")
        else:
            print("\n📚 Рекомендуемые материалы:")
            for index, row in results.iterrows():
                print(f"   • {row['title']}")
                print(f"     Описание: {row['description']}")
                print(f"     Категория: {row['category']}")
                print(f"     Теги: {row['tags']}")
                print()

if __name__ == "__main__":
    main()
