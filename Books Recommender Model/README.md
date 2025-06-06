**Books Recommender**

*Представленная программа реализует семантическую рекомендательную систему для книг на английском языке с использованием эмбеддингов и векторной базы данных Chroma.*

Ниже будут рассмотрены два файла: vector_search_and_embeddings.ipynb и main.py.

*Ключевые особенности и общий порядок исполнения файла **vector_search_and_embeddings.ipynb**:*
---
1. Загрузка и подготовка данных
Чтение подготовленного CSV-файла с данными о книгах и извлечение колонки tagged_description (текстовые описания книг с уникальными идентификаторами), сохранение в текстовый файл. В результате описания книг обрабатываются как отдельные документы.
Именно описания (кстати говоря, содержащие не менее 24 слов в результате принудительного ограничения как минимально осмысленные).

2. Создание документов

Загрузка текстовых описаний через TextLoader.
Разделение текста на отдельные документы с помощью CharacterTextSplitter - каждое описание становится самостоятельным документом.

3. Генерация векторных эмбеддингов 
 
Для векторизации текста и создания эмбеддингов используется модель paraphrase-multilingual-MiniLM-L12-v2 из библиотеки sentence-transformers.
<br>Что в ней приглянулось:
<br>a) Поддержка мультиязычности (пускай мы и остановились на использовании исключительно английского)&
<br>b) Компактна и шустро работает.
<br>c) Для своего небольшого размера качесвтенно сохраняет семантическое сходство между словами.

4. Создание векторной базы данных

Ключевой этап работы нашего проекта.
Векторизованные описания сохраняются в базе данных Chroma - специализированном хранилище, оптимизированном для работы с векторными представлениями данных.
Chroma позволяет эффективно искать семантически близкие тексты по запросу, за что благодарим индексирование.
<br>Кратко и не очень содержательно - как это работает?
При вызове Chroma.from_documents() база автоматически индексирует векторы, используя выбранный алгоритм(у нас Approximate Nearest Neighbor - ANN - кажется), что ускоряет поиск и выдает достаточно точный результат.

5. Семантический поиск рекомендаций

Функция retrieve_semantic_recommendations:
<br>a) Запрос (например, "Pulp fiction") преобразуется в вектор с помощью той же модели, что использовалась ранее для описаний (HuggingFaceEmbeddings).
<br>b) Вектор запроса сравнивается с векторами в базе через косинусное сходство.
<br>e) Извлекает уникальные идентификаторы книг из найденных документов для получения сведений о рекомендуемой литературе (таких как жанр, автор, рейтинг и название.
<br>d) Возвращает DataFrame с полной информацией о рекомендованных книгах.


Ключевые технологии

<br>LangChain: Для работы с документами и интеграции моделей.
<br>Chroma: Векторная база данных для хранения и поиска эмбеддингов.
<br>Hugging Face Transformers: Предобученная модель для генерации эмбеддингов.

*main.py*
---
Убрано все ненужное, сохранено все необходимое для запуска модели:
<br>Порядок исполнения:
1. Создается виртуальное окружение, загружаются через pip и импортируются необходимые зависимости из requirements.txt.
2. Загружается csv-файл с книгами, на основе которого ранее была создана Chroma.
3. Загружаются эмбеддинги:
   <br>embeddings = HuggingFaceEmbeddings(
    model_name =  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
4. Загружается предварительно сохраненная база Croma:
   <br>db_books = Chroma(
    persist_directory = "/путь/к/базе",
    embedding_function = embeddings
)
5. Инициализируется уже знакомая нам функция retrieve_semantic_recommendations. Именно здесь задается количество книг, которое мы хотим получить в рамках рекомендации (top_k, по умолчанию - 7)
6. Вызов функции с передачей запроса в качестве аргумента:
   <br>retrieve_semantic_recommendations(input("Enter some words describing the book you'd like to read(e.g. 'A funny and easy-going fairy tale'): "))
7. На выходе получаем остортированные по рейтингу книги в формате DataFrame с признаками: Title	Authors	Description	Rating	Category	Link	Number	Number_Description.
<br> Title,	Authors,	Description,	Rating,	Category, и	Link (ссылки на обложки) будут в дальнейшем использованы в разработке приложения в качестве предложения к ознакомлению с рекомендованной литературой пользователю.

*Слабые места и предложения к улучшению*
---
Как и везде - чем точнее запрос, тем релевантнее результат, а в рамках работы модели выражение становится еще более насущным.
<br>На пользователя ложится большая ответственность за "фильтрацию базара" - что пользователь введет, то модель и будет рассматривать как вопрос вида "найди мне семантически похожие тексты на основе того, что я ввел".
Запрос, как на приеме у порядочных гадалок, должен быть максимально информативен и корректно оформлен (как выше, например). Все сказанное может быть использовано против вас - перед нами не большая языковая модель а, по сути, система поиска, которая требует определенной инструкции. Например, введя что-нибудь вроде "Привет, я расстался с женой, а в остальном у меня все хорошо, дай мне книжку об истории СССР " модель выдаст, условно, список книг о быте граждан времен СССР и, возможно, сборник стихотворений Маяковского.  
<br> В качестве улучшения могу предложить использование других признаков, а не только Description, для повышения гибкости поиска(позволит использовать, например, рейтинг в запросах, по которому также будет осуществляться поиск)
   
