import csv
import os
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from langchain.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ollama import pull
from ollama._types import ResponseError
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────────
# Конфигурация
# ────────────────────────────────────────────────────────────────────────────────

EMBED_MODELS = [
    "nomic-embed-text",
    "multilingual-e5-small",
    "mxbai-embed-large",
]
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
GEN_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama2")
TOP_K = 1

DOCS_PATH = Path(os.getenv("DOCS_PATH", "docs"))
QUERIES_CSV = Path(os.getenv("EVAL_PATH", "queries.csv"))


LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / "rag_experiment.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def ensure_model(name: str) -> bool:
    current_digest, bars = "", {}
    try:
        for progress in pull(name, stream=True):
            digest = progress.get("digest", "")
            if digest != current_digest and current_digest in bars:
                bars[current_digest].close()

            if not digest:
                logger.info(progress.get("status"))
                continue

            if digest not in bars and (
                total := progress.get("total")
            ):
                bars[digest] = tqdm(
                    total=total,
                    desc=f"pulling {digest[7:19]}",
                    unit="B",
                    unit_scale=True,
                )

            if completed := progress.get("completed"):
                bars[digest].update(completed - bars[digest].n)

            current_digest = digest
    except ResponseError as err:
        msg = str(err)
        logger.error(
            f"❌  Нет возможности скачать модель '{name}': {msg}"
        )
        return False
    except Exception as err:
        logger.error(
            f"❌  Ошибка при скачивании модели '{name}': {err}"
        )
        return False

    logger.info("✅  Модель скачана.")
    return True


# ────────────────────────────────────────────────────────────────────────────────
# Загрузка документов и вопросов
# ────────────────────────────────────────────────────────────────────────────────


def load_documents() -> List[str]:
    """Читает все *.txt файлы из DOCS_PATH (рекурсивно)."""
    txt_files = sorted(DOCS_PATH.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"В каталоге '{DOCS_PATH}' не найдено документов .txt"
        )
    docs = []
    for path in txt_files:
        with open(path, "r", encoding="utf-8") as f:
            docs.append(f.read())
    logger.info(f"📚 Загружено документов: {len(docs)}")
    return docs


def load_queries() -> List[Dict[str, str]]:
    """Читает CSV-файл queries.csv с колонками question,answer."""
    if not QUERIES_CSV.exists():
        raise FileNotFoundError(
            f"Файл с вопросами '{QUERIES_CSV}' не найден."
        )
    with open(QUERIES_CSV, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        missing = {"question", "answer"} - set(
            reader.fieldnames or []
        )
        if missing:
            raise ValueError(
                f"В queries.csv отсутствуют обязательные колонки: {missing}"
            )
        queries = [
            {"question": row["question"], "answer": row["answer"]}
            for row in reader
        ]
    logger.info(f"❓ Загружено вопросов: {len(queries)}")
    return queries


# ────────────────────────────────────────────────────────────────────────────────
# Подготовка данных
# ────────────────────────────────────────────────────────────────────────────────

documents: List[str] = load_documents()
queries: List[Dict[str, str]] = load_queries()

# Инициализация LLM (через Ollama)
llm = ChatOllama(model=GEN_MODEL, base_url=OLLAMA_BASE_URL)

# ────────────────────────────────────────────────────────────────────────────────
# Функции поиска, генерации и оценки
# ────────────────────────────────────────────────────────────────────────────────


def embed_and_store_docs(model_name: str):
    """Создает векторное хранилище Chroma и сохраняет эмбеддинги документов."""
    try:
        if not ensure_model(model_name):
            logger.info(f"⏭️  Модель '{model_name}' пропущена.")
            return None, None
    except RuntimeError as err:
        logger.info(f"⚠️  Пропускаем модель '{model_name}': {err}")
        return None, None

    embedding = OllamaEmbeddings(model=model_name)
    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embedding,
        metadatas=[{"id": idx} for idx, _ in enumerate(documents)],
        collection_name=f"docs_{model_name}",
    )
    return vectorstore, embedding


def answer_question(query: str, vectorstore: Chroma) -> str:
    """Выполняет поиск и генерирует ответ."""
    docs_and_scores = vectorstore.similarity_search_with_score(
        query, k=TOP_K
    )
    retrieved_docs = []
    for doc, score in docs_and_scores:
        meta_id = doc.metadata.get("id", "?")
        snippet = doc.page_content[:100] + (
            "..." if len(doc.page_content) > 100 else ""
        )
        logger.info(
            f'🔎 Найден документ {meta_id} (score={score:.3f}): "{snippet}"'
        )
        retrieved_docs.append(doc.page_content)
    context = " ".join(retrieved_docs)
    prompt = f"Используя следующие данные: {context}\nОтветь на вопрос: {query}"
    ai_msg = llm.invoke(prompt)
    # ChatOllama возвращает объект AIMessage; берем его содержимое
    return ai_msg.content.strip()


def evaluate_models():
    """Оценивает каждую embedding‑модель по точности и cos similarity."""
    for model in EMBED_MODELS:
        logger.info(
            f"\n=== Тестирование embedding‑модели: {model} ==="
        )
        vectorstore, embedding = embed_and_store_docs(model)
        if vectorstore is None and embedding is None:
            continue
        total_queries = len(queries)
        correct_count = 0
        cos_sim_sum = 0.0
        for i, qa in enumerate(queries):
            question = qa["question"]
            true_answer = qa.get("answer", "")
            logger.info(f"\nВопрос {i+1}: {question}")
            generated_answer = answer_question(question, vectorstore)
            logger.info(f"💬 Ответ модели: {generated_answer}")
            # оценка точности
            if true_answer:
                normalized_answer = generated_answer.lower()
                normalized_true = true_answer.lower()
                is_correct = (
                    normalized_true in normalized_answer
                ) or (normalized_answer in normalized_true)
                correct_count += int(is_correct)
                logger.info(
                    f"✔ Эталон: {true_answer} – {'совпало' if is_correct else 'не совпало'}"
                )
                # cos similarity
                try:
                    answer_vec = embedding.embed_query(
                        generated_answer
                    )
                    true_vec = embedding.embed_query(true_answer)
                    dot = np.dot(answer_vec, true_vec)
                    norm = np.linalg.norm(
                        answer_vec
                    ) * np.linalg.norm(true_vec)
                    cosine_sim = float(dot / norm) if norm else 0.0
                except Exception as e:
                    cosine_sim = 0.0
                    logger.warning(
                        f"[WARN] Не удалось вычислить cos similarity: {e}"
                    )
                cos_sim_sum += cosine_sim
                logger.info(
                    f"🔄 Косинусная близость: {cosine_sim:.3f}"
                )
        # сводные метрики
        accuracy = (
            correct_count / total_queries if total_queries else 0.0
        )
        avg_cossim = (
            cos_sim_sum / total_queries if total_queries else 0.0
        )
        logger.info(
            f"\nРезультаты для модели {model}: accuracy = {accuracy:.2%}, средняя cos similarity = {avg_cossim:.3f}"
        )
        logger.info("=" * 40)


# Запуск
if __name__ == "__main__":
    evaluate_models()
