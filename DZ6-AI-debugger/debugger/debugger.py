import os
import requests
import re 
import difflib

# Настройки для ProxyAPI
PROXY_API_URL = 'https://api.proxyapi.ru/openai/v1/chat/completions'
PROXY_API_KEY = '...' # <- Перед использованием нужно добавить API ключ
MODEL_NAME = 'o4-mini'  # Используемая модель

def check_api_available():
    """Проверяет доступность ProxyAPI"""
    if not PROXY_API_KEY or PROXY_API_KEY == '...':
        print("Ошибка: PROXY_API_KEY не добавлен. Добавьте его в коде.")
        return False
    try:
        response = requests.get(
            PROXY_API_URL.replace('/chat/completions', '/models'),
            headers={"Authorization": f"Bearer {PROXY_API_KEY}"}
        )
        if response.status_code == 401:
            print("Ошибка: Неверный API ключ (401 Unauthorized). Проверьте PROXY_API_KEY.")
            return False
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Ошибка соединения с API: {e}")
        return False

def load_logs(log_path: str) -> str:
    """Читает файл с логами и возвращает текст."""
    with open(log_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_error_details_from_logs(logs: str) -> tuple[str | None, int | None, str | None]:
    """
    Извлекает путь к файлу, номер строки и сообщение об ошибке из последней записи Traceback.
    """
    # Ищем последнюю запись "File "...", line ..., in ..."
    # Затем ищем следующую строку, которая обычно содержит тип ошибки и сообщение
    traceback_pattern = re.compile(r'File "([^"]+)", line (\d+)(?:, in .*)?\n(?:.*\n)*?([A-Za-z0-9_]+Error: .+)')
    
    matches = list(traceback_pattern.finditer(logs))
    if not matches:
        # Более простой паттерн, если сложный не сработал (для простых ошибок без полного стека)
        simple_pattern = re.compile(r'File "([^"]+)", line (\d+)\n\s*.*\n([A-Za-z0-9_]+Error: .*)')
        matches = list(simple_pattern.finditer(logs))

    if matches:
        last_match = matches[-1] # Берем последнее совпадение, оно самое глубокое в стеке
        file_path = last_match.group(1)
        line_number = int(last_match.group(2))
        error_message_line = last_match.group(3).strip()

        return file_path, line_number, error_message_line
    return None, None, None

def load_specific_source_file(file_path: str) -> str | None:
    """Загружает содержимое указанного файла."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Ошибка: Файл исходного кода {file_path} не найден.")
        return None
    except Exception as e:
        print(f"Ошибка при чтении файла {file_path}: {e}")
        return None

def build_prompt_for_fix(logs: str, error_file_path: str, error_file_content: str, error_line_number: int, error_message: str) -> dict:
    """Формирует prompt для модели с логами и кодом для исправления конкретного файла."""
    system_message = (
        "Ты — AI‑ассистент для программиста на Python. "
        "Твоя задача — проанализировать логи программы, указанный файл с ошибкой и предложить "
        "ИСПРАВЛЕННЫЙ КОД для этого файла, внеся ТОЛЬКО МИНИМАЛЬНО НЕОБХОДИМЫЕ ИЗМЕНЕНИЯ для устранения указанной ошибки. " 
        "Сосредоточься ИСКЛЮЧИТЕЛЬНО на исправлении ошибки, упомянутой в логах. "
        "В твоем ответе должен содержаться ТОЛЬКО ИСПРАВЛЕННЫЙ КОД для указанного файла. "
        "Твоя цель - исправить ошибку, сохранив остальной код максимально нетронутым. "
        "Не добавляй никаких пояснений, комментариев, приветствий или Markdown-разметки типа ```python ... ``` до или после кода."
        "Просто верни весь код файла с исправлением."
        "Не удаляй комментарии."
    )

    # Ограничиваем логи для экономии токенов, но включаем важные части
    log_lines = logs.splitlines()
    context_logs = "\n".join(log_lines[:30] + [line for line in log_lines if "Traceback" in line or "Error" in line][:20])


    user_message = (
        f"Обнаружена ошибка в программе.\n\n"
        f"Логи:\n```\n{context_logs}\n```\n\n"
        f"Файл с ошибкой: {error_file_path}\n"
        f"Номер строки с ошибкой (примерно): {error_line_number}\n"
        f"Сообщение об ошибке: {error_message}\n\n"
        f"Содержимое файла '{error_file_path}':\n"
        f"```python\n{error_file_content}\n```\n\n"
        f"Пожалуйста, предоставь ПОЛНОСТЬЮ ИСПРАВЛЕННЫЙ КОД для файла '{error_file_path}', "
        f"исправив ТОЛЬКО указанную ошибку '{error_message}' в районе строки {error_line_number}. "
        f"Сохрани остальную часть кода и комментарии БЕЗ ИЗМЕНЕНИЙ, насколько это возможно. "
        f"Твой ответ должен содержать ТОЛЬКО код этого файла с исправлением, без каких-либо дополнительных текстов или разметки."
    )

    return {'system': system_message, 'user': user_message}


def ask_agent(prompt: dict) -> str:
    """Отправляет запрос к ProxyAPI"""
    try:
        resp = requests.post(
            PROXY_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {PROXY_API_KEY}"
            },
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": prompt['system']},
                    {"role": "user", "content": prompt['user']}
                ],
                "temperature": 1,
                "max_completion_tokens": 3000 # Увеличим, если файлы большие
            }
        )

        if resp.status_code != 200:
            raise ValueError(f"API error: {resp.status_code} - {resp.text}")

        data = resp.json()
        if not data.get('choices') or not data['choices'][0].get('message') or not data['choices'][0]['message'].get('content'):
            raise ValueError(f"API_ответ не содержит ожидаемого контента: {data}")
        
        # Удаляем возможные ```python и ``` обертки
        content = data['choices'][0]['message']['content']
        content = content.strip()
        if content.startswith("```python"):
            content = content[len("```python"):].strip()
        if content.startswith("```"): # На случай если просто ```
             content = content[len("```"):].strip()
        if content.endswith("```"):
            content = content[:-len("```")].strip()
        return content

    except Exception as e:
        raise ValueError(f"Ошибка при запросе к ProxyAPI: {str(e)}")

def display_diff(original_content: str, new_content: str, file_path: str):
    """Отображает разницу между оригинальным и новым содержимым файла."""
    print(f"\n--- Предлагаемые изменения для файла: {file_path} ---")
    diff = difflib.unified_diff(
        original_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile='original',
        tofile='corrected',
        lineterm=''
    )
    has_changes = False
    for line in diff:
        print(line, end="")
        has_changes = True
    if not has_changes:
        print("Изменений не предложено или код идентичен.")
    print("--- Конец изменений ---")
    return has_changes


def apply_code_changes(file_path: str, new_content: str):
    """Записывает новое содержимое в файл."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Изменения успешно применены к файлу: {file_path}")
    except Exception as e:
        print(f"Ошибка при записи изменений в файл {file_path}: {e}")


def analyze_and_suggest_fix(log_path: str = "debugger/logs/app.log"):
    """Основная функция анализа и предложения исправления."""
    if not os.path.exists(log_path):
        print(f"Файл логов не найден: {log_path}")
        return

    logs = load_logs(log_path)
    if not logs.strip():
        print(f"Файл логов {log_path} пуст.")
        return

    error_file_path, error_line_number, error_message = extract_error_details_from_logs(logs)

    if not error_file_path:
        print("Не удалось извлечь информацию об ошибке из логов.")
        return

    print(f"Обнаружена ошибка в файле: {error_file_path}, строка: {error_line_number}")
    print(f"Сообщение: {error_message}")

    if not os.path.isabs(error_file_path):
        pass # Путь из лога используется "как есть"

    original_file_content = load_specific_source_file(error_file_path)
    if original_file_content is None:
        return # Ошибка уже выведена функцией load_specific_source_file

    prompt = build_prompt_for_fix(logs, error_file_path, original_file_content, error_line_number, error_message)
    
    try:
        print("\nЗапрашиваю исправление у AI...")
        suggested_full_code = ask_agent(prompt)
    except ValueError as e:
        print(f"Ошибка при получении исправления от AI: {e}")
        return

    if not suggested_full_code.strip():
        print("AI не вернул никакого кода для исправления.")
        return

    has_changes = display_diff(original_file_content, suggested_full_code, error_file_path)

    if not has_changes:
        print("AI не предложил изменений или предложенный код идентичен оригиналу.")
        return

    while True:
        choice = input("Применить предложенные изменения? (y/n): ").lower()
        if choice == 'y':
            apply_code_changes(error_file_path, suggested_full_code)
            break
        elif choice == 'n':
            print("Изменения не применены.")
            break
        else:
            print("Пожалуйста, введите 'y' или 'n'.")


if __name__ == "__main__":
    if not check_api_available():
        print("Проверьте настройки API и подключение.")
    else:

        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        default_log_path = os.path.join(current_script_dir, "logs", "app.log")

        log_file_path = "debugger/logs/app.log"

        try:
            analyze_and_suggest_fix(log_path=log_file_path)
        except Exception as e:
            print(f"Непредвиденная ошибка в процессе анализа: {str(e)}")