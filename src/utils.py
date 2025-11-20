"""
Recommender - Вспомогательные функции
"""

import pandas as pd


def load_materials(data_path: str) -> pd.DataFrame:
    """
    Загружает данные о материалах

    Args:
        data_path (str): Путь к CSV файлу

    Returns:
        pandas.DataFrame: Данные о материалах
    """
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return pd.DataFrame()


def validate_query(query: str) -> bool:
    """
    Проверяет корректность запроса

    Args:
        query (str): Запрос пользователя

    Returns:
        bool: Валиден ли запрос
    """
    if not isinstance(query, str):
        return False
    if len(query.strip()) < 2:
        return False
    return True


def format_results(results: pd.DataFrame) -> list:
    """
    Форматирует результаты для вывода

    Args:
        results (pandas.DataFrame): Результаты поиска

    Returns:
        List[Dict]: Список словарей с форматированными данными
    """
    if results.empty:
        return []

    formatted = []
    for _, row in results.iterrows():
        formatted.append({
            'id': int(row['id']),
            'title': str(row['title']),
            'description': str(row['description']),
            'category': str(row['category']),
            'tags': str(row['tags']),
            'similarity': float(row.get('similarity', 0))
        })

    return formatted


def print_results(results: pd.DataFrame, query: str):
    """
    Печатает результаты поиска
    Args:
        results (pandas.DataFrame): Результаты поиска
        query (str): Запрос пользователя
    """
    if results.empty:
        print(f"❌ По запросу '{query}' ничего не найдено")
        return

    print(f"\n📚 Рекомендуемые материалы по запросу '{query}':")
    print("-" * 60)

    for _, row in results.iterrows():
        print(f"ID: {row['id']}")
        print(f"Название: {row['title']}")
        print(f"Описание: {row['description']}")
        print(f"Категория: {row['category']}")
        print(f"Теги: {row['tags']}")

        if 'similarity' in row:
            print(f"Сходство: {row['similarity']:.3f}")

        print("-" * 60)
