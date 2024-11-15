import pandas as pd
from app import titanic_data


data = {
        'Sex': ['male', 'female', 'male'],
        'Survived': [0, 1, 1]
    }
df = pd.DataFrame(data)


def test_data_lives_number():
    assert titanic_data(df, 'Спасенные', 'Число') == (
        {
            "Пол": ["Мужчины", "Женщины"],
            "Количество": [1, 1],
        }
    )


def test_data_dead_number():
    assert titanic_data(df, 'Погибшие', 'Число') == (
        {
            "Пол": ["Мужчины", "Женщины"],
            "Количество": [1, 0],
        }
    )


def test_data_lives_percentage():
    assert titanic_data(df, 'Спасенные', 'Процент') == (
        {
            "Пол": ["Мужчины", "Женщины"],
            "Процент": [50, 100],
        }
    )


def test_data_dead_percentage():
    assert titanic_data(df, 'Погибшие', 'Процент') == (
        {
            "Пол": ["Мужчины", "Женщины"],
            "Процент": [50, 0],
        }
    )
