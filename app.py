from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from io import BytesIO
import streamlit as st
import pandas as pd


processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224"
)


def process_uploaded_files(up_file):
    image = Image.open(up_file)

    if image.format.lower not in ['png', 'jpeg', 'jpg']:
        image = image.convert('RGB')
        image_bytes = BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        image = Image.open(image_bytes)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    return image_bytes, model.config.id2label[predicted_class_idx]


def lode_image():
    st.title('Генерация описания для загруженных пользователем картинок')
    up_file = st.file_uploader('Загрузите изображение', type=['png', 'jpg', 'jpeg'])
    if up_file is not None:
        # Обработка загруженного файла
        image_bytes, caption = process_uploaded_files(up_file)

        # Отображение изображения и описания

        return image_bytes, caption
    # url = "https://i.pinimg.com/originals/24/32/68/243268a10cc4a8903d07d6d89a4221bf.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)


def read_data():
    df = pd.read_csv('titanic_train.csv', delimiter=',')
    st.image('t.jpg', use_column_width=True)
    # Заголовок приложения
    st.title("Статистика спасенных и погибших")
    return df


def select_options():
    # Выбор между живыми и мертвыми
    status_option = st.selectbox("Выберите статус:", ["Спасенные", "Погибшие"])
    # Выбор между числом и процентом
    value_option = st.selectbox("Выберите отображение:", ["Число", "Процент"])
    return status_option, value_option


def titanic_data(df, status_option, value_option):

    # Преобразование статуса в значения для фильтрации
    if status_option == "Спасенные":
        survived_value = 1
    else:
        survived_value = 0

    # Подсчет количества мужчин и женщин
    male_count = len(df[(df['Sex'] == 'male') & (df['Survived'] == survived_value)])
    female_count = len(df[(df['Sex'] == 'female') & (df['Survived'] == survived_value)])

    # Подсчет общего количества мужчин и женщин для расчета процентов
    total_males = len(df[df['Sex'] == 'male'])
    total_females = len(df[df['Sex'] == 'female'])

    # Создание DataFrame для отображения результатов
    if value_option == "Число":
        results = {
            "Пол": ["Мужчины", "Женщины"],
            "Количество": [male_count, female_count],
        }
    else:  # Проценты
        male_percentage = (male_count / total_males * 100) if total_males > 0 else 0
        female_percentage = (
            (female_count / total_females * 100) if total_females > 0 else 0
        )
        results = {
            "Пол": ["Мужчины", "Женщины"],
            "Процент": [round(male_percentage, 2), round(female_percentage, 2)],
        }
        
    return results


def main():
    df = read_data()
    status_option, value_option = select_options()
    results = titanic_data(df, status_option, value_option)
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)
    image_bytes, caption = lode_image()
    if image_bytes is not None:
        st.image(image_bytes, caption="Загруженное изображение", use_column_width=True)
        st.write(f"Сгенерированное описание: {caption}")


if __name__ == '__main__':
    main()
