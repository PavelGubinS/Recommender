#!/usr/bin/env python3
"""
Recommender - Главный файл проекта
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommender import StudyRecommender

def main():
    """Основная функция приложения"""
    print("🎓 Добро пожаловать в Study Material Recommender!")
    print("=" * 50)
    
    try:
        # Создаем рекомендательную систему
        recommender = StudyRecommender("data/materials.csv")
    except Exception as e:
        print(f"❌ Ошибка при инициализации системы: {e}")
        return
    
    # Примеры запросов для демонстрации
    examples = [
        "Python for beginners",
        "Machine learning with scikit-learn",
        "Data analysis with Pandas",
        "Text processing in NLP"
    ]
    
    print("\n🔍 Примеры запросов:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    
    print("\nВведите свой запрос (или 'quit' для выхода):")
    
    while True:
        try:
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
                    if 'similarity' in row:
                        print(f"     Сходство: {row['similarity']:.3f}")
                    print()
                    
        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка при обработке запроса: {e}")

if __name__ == "__main__":
    main()
