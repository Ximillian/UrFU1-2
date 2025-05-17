# server.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any
import os
import uvicorn

from fastapi import FastAPI, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware # Для UI

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts_books_reco")
CHROMA_DB_DIR = ARTIFACTS_DIR / "chroma_db"
EMBEDDINGS_MODEL_DIR = ARTIFACTS_DIR / "embeddings_model"
# SENTIMENT_MODEL_DIR = ARTIFACTS_DIR / "sentiment_model" # Модель эмоций не нужна для поиска, только данные
PROCESSED_DATA_FILE = ARTIFACTS_DIR / "books_with_emotions.csv"

# ЗАГРУЗКА РЕСУРСОВ ПРИ СТАРТЕ

def load_artifacts():
    """ Загружает модели и данные в память. """
    logger.info("Начинаем загрузку артефактов...")
    
    # 1. Данные о книгах
    if not PROCESSED_DATA_FILE.exists():
         logger.error(f"ОШИБКА: Файл данных '{PROCESSED_DATA_FILE}' не найден.")
         raise RuntimeError(f"Файл данных '{PROCESSED_DATA_FILE}' не найден.")
    df_books = pd.read_csv(PROCESSED_DATA_FILE)
     # Замена NaN на None (null в JSON) или пустые строки, чтобы JSON был валидным
    df_books = df_books.replace({np.nan: None})
    logger.info(f"Загружено {len(df_books)} книг с данными об эмоциях.")

    # 2. Модель Эмбеддингов
    if not EMBEDDINGS_MODEL_DIR.exists() or not any(EMBEDDINGS_MODEL_DIR.iterdir()):
         logger.error(f"ОШИБКА: Директория эмбеддингов '{EMBEDDINGS_MODEL_DIR}' не найдена или пуста.")
         raise RuntimeError(f"Директория эмбеддингов '{EMBEDDINGS_MODEL_DIR}' не найдена или пуста.")
         
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=str(EMBEDDINGS_MODEL_DIR),
            model_kwargs={'device': 'cpu'}, # или 'cuda'
         )
        logger.info("Модель эмбеддингов инициализирована из локальной директории.")
    except Exception as e:
        logger.warning(f"Не удалось загрузить эмбеддинги локально ({e}). Пробуем fallback через имя...")
        try:
           # Запасной вариант, если локальная загрузка не сработала
           with open(EMBEDDINGS_MODEL_DIR / "model_name.txt", "r") as f: 
                embedding_model_name_original = f.read().strip()
           embeddings = HuggingFaceEmbeddings(
              model_name=embedding_model_name_original,
              model_kwargs={'device': 'cpu'}
            )
           logger.info(f"Модель эмбеддингов инициализирована с использованием Hub: {embedding_model_name_original}.")
        except Exception as e_fallback:
            logger.error(f"ОШИБКА: Запасной вариант инициализации эмбеддингов также не удался: {e_fallback}")
            raise RuntimeError(f"Не удалось загрузить модель эмбеддингов: {e_fallback}")
            
    # 3. Векторная БД
    if not CHROMA_DB_DIR.exists() or not any(CHROMA_DB_DIR.iterdir()):
       logger.error(f"ОШИБКА: Директория Chroma DB '{CHROMA_DB_DIR}' не найдена или пуста.")
       raise RuntimeError(f"Директория Chroma DB '{CHROMA_DB_DIR}' не найдена или пуста.")
       
    db = Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embeddings
     )
    logger.info("База данных Chroma загружена.")
    
    return df_books, db, embeddings

# Глобальные переменные для хранения загруженных артефактов
try:
  df_books_processed, db_books, embeddings_model = load_artifacts()
except Exception as e:
   logger.error(f"!!! КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАГРУЗКЕ АРТЕФАКТОВ. СЕРВЕР НЕ МОЖЕТ РАБОТАТЬ. {e} !!!")
   df_books_processed = None 
   db_books = None
   embeddings_model = None
   # exit(1) 

# ЛОГИКА РЕКОМЕНДАЦИЙ

def retrieve_semantic_recommendations(
    query: str, 
    top_k: int, 
    db: Chroma, 
    books_data: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Извлекает рекомендации книг на основе семантического сходства с запросом.
    Возвращает DataFrame.
    """
    if not db:
       raise RuntimeError("Векторная база данных не была загружена.")
       
    logger.info(f"Выполняется поиск для: '{query}' (k={top_k})")
    recs_docs = db.similarity_search(query, k=top_k*2) # Берем с запасом на случай дублей/ошибок ID

    if not recs_docs:
        return pd.DataFrame() # Возвращаем пустой DataFrame

    book_ids = []
    seen_ids = set()
    for doc in recs_docs:
        try:
            # Ожидаем, что page_content начинается с ID
            book_id_str = doc.page_content.strip('"').split()[0]
            if book_id_str.isdigit():
                book_id = int(book_id_str)
                if book_id not in seen_ids:
                   book_ids.append(book_id)
                   seen_ids.add(book_id)
            else:
                 logger.warning(f"В документе найден нечисловой ID: '{book_id_str}'")
        except (ValueError, IndexError) as e:
            logger.warning(f"Не удалось извлечь ID из документа: {doc.page_content[:70]}... Ошибка: {e}")
            
        if len(book_ids) >= top_k: # Набрали нужное количество
            break
            
    if not book_ids:
         return pd.DataFrame()

    recommendations_df = books_data[books_data["Number"].isin(book_ids)].copy()
    
     # Восстанавливаем порядок релевантности из ChromaDB
    if not recommendations_df.empty:
        recommendations_df['sort_order'] = recommendations_df['Number'].apply(lambda x: book_ids.index(x) if x in book_ids else float('inf'))
        recommendations_df = recommendations_df.sort_values('sort_order').drop(columns=['sort_order'])
    
    return recommendations_df.head(top_k)


# ИНИЦИАЛИЗАЦИЯ FastAPI

app = FastAPI(
    title="Book Recommender API",
    description="API для получения рекомендаций книг по семантической близости.",
     version="1.0.0"
)

# CORS Middleware
origins = [
    "http://localhost",
    "http://localhost:8080",
     "http://127.0.0.1",
     "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Разрешить GET, POST и т.д.
    allow_headers=["*"],
)

# --- API ENDPOINTS ---

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
     """ Проверка, что сервис жив и модели загружены """
     if db_books and df_books_processed is not None:
       return {"status": "OK", "message": "Service and models are loaded."}
     else:
        raise HTTPException(
           status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
           detail="Service is starting or artifacts failed to load.",
        )


@app.get("/recommend",
         summary="Получить рекомендации книг",
         response_description="Список рекомендованных книг"
         )
# FastAPI запустит синхронную функцию в своем пуле потоков, не блокируя основной цикл
def get_recommendations(
    query: str = Query(..., min_length=3, description="Текстовый запрос для поиска похожих книг."), 
    top_k: int = Query(5, ge=1, le=50, description="Количество рекомендаций для возврата.")
    ) -> List[Dict[str, Any]]: # Явно указываем, что вернется Список Словарей
    """
    Ищет книги, семантически близкие к **query**, 
    и возвращает **top_k** наиболее подходящих.
    """
    if not db_books or df_books_processed is None:
       raise HTTPException(
           status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
           detail="Сервис недоступен, модели не загружены.",
        )
        
    try:
      recommendations_df = retrieve_semantic_recommendations(
          query, top_k, db_books, df_books_processed
          )
          
      if recommendations_df.empty:
          return [] # Возвращаем пустой список, если ничего не найдено
          
      results = recommendations_df.to_dict(orient='records')
      return results
      
    except Exception as e:
       logger.exception(f"Ошибка при обработке запроса: {query}")
       raise HTTPException(
           status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
           detail=f"Внутренняя ошибка сервера при обработке запроса: {e}",
        )

logger.info("Настройка FastAPI завершена.")

if __name__ == "__main__":
    app_port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=app_port)