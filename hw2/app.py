"""Homework 2 Chainlit application using the course RAG database."""

import os
from pathlib import Path

import chainlit as cl
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAIEmbeddings


# These values may be changed in the environment without editing this file.
COURSE_RAG_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "02_LangChain" / "07_RAG" / "rag_data" / ".chromadb"
)
RAG_DATABASE_PATH = os.getenv("RAG_DATABASE_PATH", str(COURSE_RAG_DIRECTORY))
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-west1")
RETRIEVER_K = int(os.getenv("RAG_RETRIEVER_K", "4"))
MIN_RELEVANCE = float(os.getenv("RAG_MIN_RELEVANCE", "0.35"))
HISTORY_EXCHANGES = int(os.getenv("RAG_HISTORY_EXCHANGES", "3"))
FALLBACK_RESPONSE = (
    "I don't have enough information in the retrieved course documents to answer that question."
)


def create_vectorstore() -> Chroma:
    """Open the persisted Chroma database with the course Vertex AI embeddings."""
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        raise RuntimeError("Set GOOGLE_CLOUD_PROJECT before starting the application.")

    if not os.getenv("GOOGLE_MODEL"):
        raise RuntimeError("Set GOOGLE_MODEL before starting the application.")

    if not Path(RAG_DATABASE_PATH).is_dir():
        raise RuntimeError(
            f"RAG database was not found at {RAG_DATABASE_PATH}. "
            "Set RAG_DATABASE_PATH to a persisted Chroma database."
        )

    embeddings = VertexAIEmbeddings(
        model_name="gemini-embedding-001",
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=GOOGLE_CLOUD_LOCATION,
    )
    return Chroma(
        persist_directory=RAG_DATABASE_PATH,
        embedding_function=embeddings,
    )


def format_documents(documents: list) -> str:
    """Combine retrieved document text into the context sent to Gemini."""
    return "\n\n".join(document.page_content for document in documents)


def retrieve_documents(query: str) -> list[tuple]:
    """Retrieve document-score pairs that meet the configured relevance threshold."""
    results = vectorstore.similarity_search_with_relevance_scores(query, k=RETRIEVER_K)
    relevant_results = [
        (document, score) for document, score in results if score >= MIN_RELEVANCE
    ]
    return relevant_results


def format_retrieval_results(retrieval_results: list[tuple]) -> str:
    """Format each retrieved document's source metadata and relevance score."""
    return "\n".join(
        "- "
        f"{document.metadata.get('source', 'Unknown source')} "
        f"(relevance: {score:.3f})"
        for document, score in retrieval_results
    )


def response_to_text(response: object) -> str:
    """Extract text from Gemini responses, returning an empty string if unsupported."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content

    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
            elif isinstance(getattr(block, "text", None), str):
                text_parts.append(block.text)
        return "\n".join(text_parts)

    return ""


def format_conversation_history(history: list[dict[str, str]]) -> str:
    """Format recent question-and-answer exchanges for the Gemini prompt."""
    if not history:
        return "No previous conversation."

    return "\n\n".join(
        f"User: {exchange['question']}\nAssistant: {exchange['answer']}"
        for exchange in history
    )


def save_exchange(question: str, answer: str) -> None:
    """Save one exchange in Chainlit session state and keep only recent history."""
    history = cl.user_session.get("conversation_history", [])
    history.append({"question": question, "answer": answer})
    cl.user_session.set("conversation_history", history[-HISTORY_EXCHANGES:])


prompt = ChatPromptTemplate.from_template(
    """You answer questions using the retrieved context below as the primary factual source.
Conversation history may help interpret follow-up questions, but do not use it as factual
evidence when the retrieved context does not support an answer.
If the context does not contain enough information, respond exactly with:
{fallback_response}
Keep a supported answer concise, with no more than three sentences.

Recent conversation history:
{history}

Question: {question}

Retrieved context:
{context}

Answer:"""
)

vectorstore = create_vectorstore()
llm = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL"))
answer_chain = prompt | llm


@cl.on_chat_start
async def on_chat_start() -> None:
    """Create empty conversation history and welcome the user to the RAG chat."""
    cl.user_session.set("conversation_history", [])
    await cl.Message(
        content=(
            "**Homework 2 RAG Chatbot**\n\n"
            "Ask a question about the documents in the course RAG database."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Answer using RAG plus recent session history and display retrieval scores."""
    question = message.content.strip()
    if not question:
        await cl.Message(content="Please enter a question.").send()
        return

    retrieval_results = retrieve_documents(question)
    if not retrieval_results:
        save_exchange(question, FALLBACK_RESPONSE)
        await cl.Message(
            content=(
                f"{FALLBACK_RESPONSE}\n\n"
                "Retrieved documents:\n- No documents met the relevance threshold."
            )
        ).send()
        return

    documents = [document for document, _ in retrieval_results]
    history = cl.user_session.get("conversation_history", [])
    gemini_response = answer_chain.invoke(
        {
            "question": question,
            "context": format_documents(documents),
            "history": format_conversation_history(history),
            "fallback_response": FALLBACK_RESPONSE,
        }
    )
    answer = response_to_text(gemini_response)
    if not isinstance(answer, str) or not answer.strip():
        answer = FALLBACK_RESPONSE
    else:
        answer = answer.strip()

    save_exchange(question, answer)
    retrieved_documents = format_retrieval_results(retrieval_results)
    response_content = f"{answer}\n\nRetrieved documents:\n{retrieved_documents}"
    await cl.Message(
        content=response_content
    ).send()


if __name__ == "__main__":
    cl.run()
