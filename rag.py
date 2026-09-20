"""Retrieval and grounded answer generation for the FMR chatbot."""

from __future__ import annotations

import os
import re
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv

from embeddings import get_embedding_function

BASE_DIR = Path(__file__).resolve().parent
COLLECTION_NAME = "fmr_documents"

NOT_FOUND_MESSAGE = "I could not find that in the provided FMR reports."
OFF_TOPIC_MESSAGE = (
    "I can only answer questions about the NBP Fund Manager Reports. "
    "Could you please ask about a fund's risk profile, performance, asset allocation, or comparison?"
)
ABUSE_REDIRECT_MESSAGE = (
    "Let's keep this conversation respectful. Please ask a question about a fund's "
    "risk profile, performance, asset allocation, or comparison, and I will be glad to help."
)

GREETING_PATTERN = re.compile(
    r"^\s*(hi+|hello+|hey+|hy+|salam|assalam.?o.?alaikum|assalamualaikum|"
    r"good\s+(morning|afternoon|evening))[\s!.,]*$",
    re.IGNORECASE,
)
ABUSE_KEYWORDS = (
    "stupid",
    "idiot",
    "dumb",
    "useless",
    "shut up",
    "nonsense",
    "garbage",
)

SYSTEM_PROMPT = """You answer questions about NBP Fund Management Limited Fund Manager Reports.
Use only the supplied context. Never guess, calculate missing values, or use outside knowledge.
When the context is insufficient, say: 'I could not find that in the provided FMR reports.'
Distinguish Conventional and Islamic funds carefully. Do not write page numbers in the answer
prose because a report may have a printed page number different from its PDF page number. The
application separately displays the exact source_file and PDF source_page metadata. Never invent
financial facts. This is informational content, not financial advice.

If the user's message is not a question about the NBP Fund Manager Reports (for example personal
questions, general knowledge, or casual conversation unrelated to funds), do not answer using
outside knowledge. Reply with exactly: "I can only answer questions about the NBP Fund Manager
Reports. Could you please ask about a fund's risk profile, performance, asset allocation, or
comparison?"

If the user's message is offensive, rude, or abusive, never respond in kind and never insult the
user. Reply calmly with exactly: "Let's keep this conversation respectful. Please ask a question
about a fund's risk profile, performance, asset allocation, or comparison, and I will be glad to
help."""


def matched_section(question: str) -> str | None:
    """Return the report section most closely described by the user's question."""
    lowered = question.lower()
    for keywords, label in (
        (("risk", "dangerous", "principal erosion"), "Risk Profile"),
        (("performance", "return", "outperformance", "benchmark"), "Performance"),
        (("allocation", "equities", "stocks", "cash"), "Asset Allocation"),
        (("holding", "portfolio", "investment"), "Top Holdings"),
        (("manager", "commentary", "objective"), "Fund Manager Commentary"),
    ):
        if any(keyword in lowered for keyword in keywords):
            return label
    return None


def collection():
    """Open the persistent ChromaDB collection used by the chatbot."""
    load_dotenv(BASE_DIR / ".env")
    client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
    return client.get_or_create_collection(COLLECTION_NAME, embedding_function=get_embedding_function())


def retrieve(question: str, count: int = 5) -> list[dict]:
    """Return relevant chunks, prioritizing an explicitly named NBP fund."""
    candidate_count = max(count * 4, 12)
    result = collection().query(query_texts=[question], n_results=candidate_count)
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    section = matched_section(question)
    matches = [
        {
            "text": text,
            "metadata": {
                **metadata,
                **({"matched_section": section} if section else {}),
            },
            "distance": distance,
        }
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]
    fund_match = re.search(r"\b(NBP(?:\s+[A-Za-z-]+){1,6}\s+FUND)\b", question, re.IGNORECASE)
    if fund_match:
        fund_phrase = re.sub(r"\s+", " ", fund_match.group(1).lower()).strip()
        for item in matches:
            item["fund_match"] = fund_phrase in re.sub(r"\s+", " ", item["text"].lower())
        exact_matches = [item for item in matches if item["fund_match"]]
        if exact_matches:
            return exact_matches[:count]
        matches.sort(key=lambda item: (not item["fund_match"], item["distance"]))
    return matches[:count]


def answer(question: str, count: int = 5) -> dict:
    """Retrieve evidence and generate a grounded answer with the configured AI model."""
    if GREETING_PATTERN.match(question):
        return {
            "answer": (
                "Hello! I can help you explore the NBP Fund Manager Reports. "
                "Ask me about a fund's risk profile, performance, asset allocation, or comparison."
            ),
            "sources": [],
        }
    lowered_question = question.lower()
    if any(keyword in lowered_question for keyword in ABUSE_KEYWORDS):
        return {"answer": ABUSE_REDIRECT_MESSAGE, "sources": []}

    sources = retrieve(question, count=count)
    if not sources:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    context = "\n\n".join(
        f"[Source {index}] {item['metadata']}\n{item['text']}"
        for index, item in enumerate(sources, start=1)
    )
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        ollama_response = requests.post(
            f"{os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')}/api/chat",
            json={
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                "options": {
                    "temperature": 0,
                    "num_predict": int(os.getenv("OLLAMA_MAX_TOKENS", "300")),
                    "num_ctx": int(os.getenv("OLLAMA_CONTEXT_SIZE", "8192")),
                },
            },
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "300")),
        )
        if ollama_response.status_code == 404:
            raise RuntimeError(f"Ollama model was not found. Run: ollama pull {model}")
        ollama_response.raise_for_status()
        response = ollama_response.json().get("message", {}).get("content", "").strip()
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Add a valid key to .env.")

        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        response = completion.choices[0].message.content or "I could not find that in the provided FMR reports."
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Add a valid key to .env.")

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        completion = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=f"Context:\n{context}\n\nQuestion: {question}",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0,
                max_output_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "800")),
            ),
        )
        try:
            response = (completion.text or "").strip()
        except (ValueError, AttributeError):
            response = ""
    else:
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER: {provider}. Use 'ollama', 'openai', or 'gemini'."
        )

    if not response:
        return {
            "answer": (
                "I wasn't able to generate a complete answer for that question. "
                "Please try rephrasing it or asking again."
            ),
            "sources": [],
        }

    if response in (NOT_FOUND_MESSAGE, OFF_TOPIC_MESSAGE, ABUSE_REDIRECT_MESSAGE):
        return {"answer": response, "sources": []}

    return {"answer": response, "sources": sources}
