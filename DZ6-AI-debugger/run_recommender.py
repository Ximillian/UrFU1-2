# run_recommender.py
import pandas as pd
from pathlib import Path
import numpy as np

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

print("Запуск приложения для рекомендаций...")

ARTIFACTS_DIR = Path("artifacts_books_reco")
CHROMA_DB_DIR = ARTIFACTS_DIR / "chroma_db"
EMBEDDINGS_MODEL_DIR = ARTIFACTS_DIR / "embeddings_model"
SENTIMENT_MODEL_DIR = ARTIFACTS_DIR / "sentiment_model"
PROCESSED_DATA_FILE = ARTIFACTS_DIR / "books_with_emotions.csv"

print(f"Загрузка обработанных данных о книгах из {PROCESSED_DATA_FILE}...")
if not PROCESSED_DATA_FILE.exists():
    print(f"ОШИБКА: Файл обработанных данных '{PROCESSED_DATA_FILE}' не найден.")
    print("Пожалуйста, сначала запустите 'prepare_model.py' для генерации артефактов.")
    exit()
df_books_processed = pd.read_csv(PROCESSED_DATA_FILE)
print(f"Загружено {len(df_books_processed)} книг с данными об эмоциях.")

print(f"Загрузка модели эмбеддингов из локальной директории: {EMBEDDINGS_MODEL_DIR}...")
if not EMBEDDINGS_MODEL_DIR.exists() or not any(EMBEDDINGS_MODEL_DIR.iterdir()):
    print(f"ОШИБКА: Директория модели эмбеддингов '{EMBEDDINGS_MODEL_DIR}' не найдена или пуста.")
    print("Пожалуйста, убедитесь, что 'prepare_model.py' корректно сохранил файлы модели эмбеддингов.")
    print("И что 'prepare_model.py' использовал `embeddings.client.save(str(EMBEDDINGS_MODEL_DIR))`.")
    exit()

try:
    embeddings = HuggingFaceEmbeddings(
        model_name=str(EMBEDDINGS_MODEL_DIR),
        model_kwargs={'device': 'cpu'}, # или 'cuda' если есть GPU
    )
    print("Модель эмбеддингов инициализирована с использованием локальной директории.")
except Exception as e:
    print(f"ОШИБКА: Не удалось инициализировать эмбеддинги из локальной директории {EMBEDDINGS_MODEL_DIR}: {e}")
    print("Попытка загрузки с использованием оригинального имени модели из Hugging Face Hub (требуется интернет)...")
    try:
        with open(EMBEDDINGS_MODEL_DIR / "model_name.txt", "r") as f: # Предполагаем, что prepare_model.py сохраняет это
            embedding_model_name_original = f.read().strip()
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name_original,
            model_kwargs={'device': 'cpu'}
            )
        print(f"Модель эмбеддингов инициализирована с использованием model_name из Hub: {embedding_model_name_original} (запасной вариант).")
    except Exception as e_fallback:
        print(f"ОШИБКА: Запасной вариант инициализации эмбеддингов также не удался: {e_fallback}")
        print("Пожалуйста, убедитесь, что 'prepare_model.py' правильно сохранил модель эмбеддингов или у вас есть доступ в интернет.")
        exit()

print(f"Загрузка векторной базы данных Chroma из {CHROMA_DB_DIR}...")
if not CHROMA_DB_DIR.exists() or not any(CHROMA_DB_DIR.iterdir()):
    print(f"ОШИБКА: Директория Chroma DB '{CHROMA_DB_DIR}' не найдена или пуста.")
    print("Пожалуйста, сначала запустите 'prepare_model.py'.")
    exit()
db_books = Chroma(
    persist_directory=str(CHROMA_DB_DIR),
    embedding_function=embeddings
)
print("База данных Chroma загружена.")

print(f"Загрузка модели анализа тональности из {SENTIMENT_MODEL_DIR}...")
if not SENTIMENT_MODEL_DIR.exists():
    print(f"ОШИБКА: Директория модели анализа тональности '{SENTIMENT_MODEL_DIR}' не найдена.")
    print("Пожалуйста, сначала запустите 'prepare_model.py'.")
    exit()
try:
    sent_tokenizer = AutoTokenizer.from_pretrained(str(SENTIMENT_MODEL_DIR), local_files_only=True)
    sent_model = AutoModelForSequenceClassification.from_pretrained(str(SENTIMENT_MODEL_DIR), local_files_only=True)
    sentiment_classifier = pipeline(
        "text-classification",
        model=sent_model,
        tokenizer=sent_tokenizer,
        top_k=None,
        truncation=True,
        device=-1 # -1 для CPU, 0 для первого GPU
    )
    print("Модель анализа тональности загружена с использованием локальных файлов.")
except Exception as e:
    print(f"Ошибка загрузки модели анализа тональности с использованием local_files_only: {e}")
    print("Попытка загрузки модели анализа тональности без local_files_only (может потребоваться загрузка из интернета).")
    try:
        sent_tokenizer = AutoTokenizer.from_pretrained(str(SENTIMENT_MODEL_DIR))
        sent_model = AutoModelForSequenceClassification.from_pretrained(str(SENTIMENT_MODEL_DIR))
        sentiment_classifier = pipeline(
            "text-classification",
            model=sent_model,
            tokenizer=sent_tokenizer,
            top_k=None,
            truncation=True,
            device=-1
        )
        print("Модель анализа тональности загружена (возможно, с загрузками из интернета).")
    except Exception as e_sent_fallback:
        print(f"Ошибка загрузки модели анализа тональности в запасном варианте: {e_sent_fallback}")
        print("Функциональность анализа тональности может быть нарушена.")
        sentiment_classifier = None

def retrieve_semantic_recommendations(query: str, top_k: int = 10, books_data: pd.DataFrame = df_books_processed) -> pd.DataFrame:
    """
    Извлекает рекомендации книг на основе семантического сходства с запросом.
    """
    if not db_books:
        print("Ошибка: Векторная база данных (db_books) не загружена.")
        return pd.DataFrame()

    print(f"\nПоиск книг, похожих на: '{query}' (топ {top_k})")
    recs_docs = db_books.similarity_search(query, k=top_k)

    if not recs_docs:
        print("Рекомендации не найдены.")
        return pd.DataFrame()

    book_ids = []
    seen_ids = set()
    for doc in recs_docs:
        try:
            book_id_str = doc.page_content.strip('"').split()[0]
            if book_id_str.isdigit():
                book_id = int(book_id_str)
                if book_id not in seen_ids:
                    book_ids.append(book_id)
                    seen_ids.add(book_id)
            else:
                # print(f"Предупреждение: В документе найден нечисловой ID: '{book_id_str}' из '{doc.page_content[:50]}...'")
                pass
        except (ValueError, IndexError) as e:
            # print(f"Предупреждение: Не удалось извлечь ID книги из документа: {doc.page_content[:50]}... Ошибка: {e}")
            pass
        if len(book_ids) >= top_k:
            break
    
    if not book_ids:
        print("Не удалось извлечь действительные ID книг из результатов поиска.")
        return pd.DataFrame()

    recommendations_df = books_data[books_data["Number"].isin(book_ids)].copy()
    
    if not recommendations_df.empty:
        recommendations_df['sort_order'] = recommendations_df['Number'].apply(lambda x: book_ids.index(x) if x in book_ids else float('inf'))
        recommendations_df = recommendations_df.sort_values('sort_order').drop(columns=['sort_order'])
    
    return recommendations_df.head(top_k)


# --- Интерактивный цикл запросов ---
def main():
    print("\n--- Рекомендатель книг готов ---")
    print("Введите ваш запрос для получения рекомендаций книг или 'quit' для выхода.")

    while True:
        user_query = input("\nВведите ваш книжный запрос: ").strip()
        if user_query.lower() == 'quit':
            break
        if not user_query:
            continue

        num_results_str = input(f"Сколько рекомендаций (по умолчанию 5)? ")
        try:
            num_results = int(num_results_str) if num_results_str else 5
            if num_results <= 0: num_results = 5
        except ValueError:
            num_results = 5
            print("Неверное число, используется значение по умолчанию 5.")


        recommendations = retrieve_semantic_recommendations(user_query, top_k=num_results)

        if not recommendations.empty:
            print(f"\n--- Топ {len(recommendations)} рекомендаций для '{user_query}' ---")
            for idx, row in recommendations.iterrows():
                print(f"\nНазвание: {row.get('Title', 'Н/Д')}")
                print(f"  Номер (ID): {row['Number']}")
                print(f"  Автор: {row.get('Author', 'Н/Д')}")
                print(f"  Описание: {row['Description'][:200]}...")
                
                emotion_cols = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
                print("  Доминирующие эмоции (макс. оценка на предложение в описании):")
                relevant_emotions = []
                for emotion in emotion_cols:
                    if emotion in row and pd.notna(row[emotion]) and row[emotion] > 0.0: # Показываем только если есть балл
                        relevant_emotions.append((emotion, row[emotion]))
                # Сортируем по убыванию балла
                relevant_emotions.sort(key=lambda x: x[1], reverse=True)
                for emotion, score in relevant_emotions:
                     print(f"    {emotion.capitalize()}: {score:.4f}")
                if not relevant_emotions:
                    print("    (Значимые эмоции не обнаружены или данные отсутствуют)")
        else:
            print(f"К сожалению, рекомендации для '{user_query}' не найдены. Попробуйте другой запрос.")

    print("Выход из рекомендателя. До свидания!")

if __name__ == "__main__":
    main()