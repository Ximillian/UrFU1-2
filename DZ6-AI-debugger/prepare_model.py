# prepare_model.py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from transformers import pipeline

print("Начинаем подготовку модели...")

ARTIFACTS_DIR = Path("artifacts_books_reco")
ARTIFACTS_DIR.mkdir(exist_ok=True)

CHROMA_DB_DIR = ARTIFACTS_DIR / "chroma_db"
EMBEDDINGS_MODEL_DIR = ARTIFACTS_DIR / "embeddings_model"
SENTIMENT_MODEL_DIR = ARTIFACTS_DIR / "sentiment_model"
PROCESSED_DATA_FILE = ARTIFACTS_DIR / "books_with_emotions.csv"
TAGGED_DESC_FILE = ARTIFACTS_DIR / "tagged_description.txt" # Временный файл

print("Загрузка и предварительная обработка данных...")
try:
    df_books = pd.read_csv('Books_FINISH_04-05-25.csv')
except FileNotFoundError:
    print("ОШИБКА: 'Books_FINISH_04-05-25.csv' не найден.")
    exit()

df_books = df_books.rename(columns={'Number_Description': 'tagged_description'})

df_books['tagged_description'].to_csv(TAGGED_DESC_FILE,
                                      sep="\n",
                                      index=False,
                                      header=False)

print("Загрузка документов и инициализация модели эмбеддингов...")
raw_doc = TextLoader(str(TAGGED_DESC_FILE)).load() # Используем сохраненный файл
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n") # chunk_size=0 выглядит странно, обычно >0
documents = text_splitter.split_documents(raw_doc)

# Проверяем, правильно ли загружены документы
if not documents:
    print("ОШИБКА: Документы не были загружены. Проверьте 'tagged_description.txt' и TextLoader.")
    exit()
# print(f"Содержимое первого документа: {documents[0].page_content[:100]}...") # Отладочная печать

embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name
)

print("Создание и сохранение векторной базы данных Chroma...")
db_books = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=str(CHROMA_DB_DIR)
)
print(f"База данных Chroma сохранена в {CHROMA_DB_DIR}")

print(f"Сохранение клиента SentenceTransformer модели эмбеддингов в {EMBEDDINGS_MODEL_DIR}...")
EMBEDDINGS_MODEL_DIR.mkdir(exist_ok=True)
# Атрибут 'client' содержит фактическую модель SentenceTransformer
if hasattr(embeddings, 'client') and hasattr(embeddings.client, 'save'):
    embeddings.client.save(str(EMBEDDINGS_MODEL_DIR))
    print(f"Клиент модели эмбеддингов сохранен. Вы можете загрузить его с помощью SentenceTransformer('{EMBEDDINGS_MODEL_DIR}')")
else:
    print(f"Предупреждение: Не удалось сохранить embeddings.client. Для загрузки будет использоваться model_name '{embedding_model_name}'.")
    # Сохраняем имя модели для последующего использования, если прямое сохранение не удалось
    with open(EMBEDDINGS_MODEL_DIR / "model_name.txt", "w") as f:
        f.write(embedding_model_name)


print("Инициализация и сохранение модели анализа тональности...")
sentiment_model_name = "j-hartmann/emotion-english-distilroberta-base"
sentiment_classifier = pipeline(
    "text-classification",
    model=sentiment_model_name,
    top_k=None,
    truncation=True
)

SENTIMENT_MODEL_DIR.mkdir(exist_ok=True)
sentiment_classifier.model.save_pretrained(str(SENTIMENT_MODEL_DIR))
sentiment_classifier.tokenizer.save_pretrained(str(SENTIMENT_MODEL_DIR))
print(f"Модель анализа тональности и токенизатор сохранены в {SENTIMENT_MODEL_DIR}")

print("Выполнение анализа тональности описаний книг (это может занять некоторое время)...")
books_for_sentiment = df_books.copy() # Используем df_books, загруженный ранее

emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

isbn_list = [] 
emotion_scores_dict = {label: [] for label in emotion_labels}

def calculate_max_emotion_scores(predictions_list, labels_order):
    per_emotion_scores = {label: [] for label in labels_order}
    for sentence_prediction_list in predictions_list:
        current_sentence_scores = {p['label']: p['score'] for p in sentence_prediction_list}
        for label in labels_order:
            per_emotion_scores[label].append(current_sentence_scores.get(label, 0.0)) # По умолчанию 0.0, если метка отсутствует

    # Вычисляем максимальную оценку для каждой эмоции по всем предложениям описания
    max_scores = {label: np.max(scores) if scores else 0.0 for label, scores in per_emotion_scores.items()}
    return max_scores


for i in tqdm(range(len(books_for_sentiment))):
    isbn_list.append(books_for_sentiment["Number"][i])
    description_text = books_for_sentiment["Description"][i]
    if pd.isna(description_text) or not isinstance(description_text, str) or description_text.strip() == "":
        # Обработка пустых или NaN описаний
        max_scores = {label: 0.0 for label in emotion_labels}
    else:
        sentences = [s.strip() for s in description_text.split(".") if s.strip()]
        if not sentences: # Если разделение не приводит к допустимым предложениям
             max_scores = {label: 0.0 for label in emotion_labels}
        else:
            predictions = sentiment_classifier(sentences) # Это возвращает список списков словарей
            max_scores = calculate_max_emotion_scores(predictions, emotion_labels)

    for label in emotion_labels:
        emotion_scores_dict[label].append(max_scores[label])

emotions_df = pd.DataFrame(emotion_scores_dict)
emotions_df["Number"] = isbn_list # Используем то же имя столбца ID, что и в df_books, для слияния

books_processed = pd.merge(df_books, emotions_df, on="Number", how="left")

print(f"Сохранение обработанного DataFrame с эмоциями в {PROCESSED_DATA_FILE}...")
books_processed.to_csv(PROCESSED_DATA_FILE, index=False)

if TAGGED_DESC_FILE.exists():
    TAGGED_DESC_FILE.unlink()

print("Подготовка модели завершена. Артефакты сохранены в:", ARTIFACTS_DIR)