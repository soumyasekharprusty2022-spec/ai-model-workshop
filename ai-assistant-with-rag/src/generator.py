from langchain_ollama import ChatOllama  # Connect LangChain to the local Ollama model.

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL  # Read the local model configuration.

_llm = None  # Cache the chat model so every query does not reload it.

# Message shown for any question the documents can't answer -- greetings, small talk,
# or genuinely out-of-scope questions.
OUT_OF_SCOPE_MESSAGE = (  # Stable user-facing response for unsupported questions.
    "I am an Ayushman Bharat health adviser. Please ask me a question related to "
    "Ayushman Bharat / PM-JAY."
)

# The model is told to reply with this exact sentinel when it can't answer from context,
# rather than trusting it to phrase "I don't know" consistently -- the sentinel is then
# swapped for OUT_OF_SCOPE_MESSAGE in code below, so the wording shown to the user is
# always exactly what you configured, not whatever the LLM feels like generating.
OUT_OF_SCOPE_SENTINEL = "OUT_OF_SCOPE"  # Exact model output used to signal missing context.

PROMPT_TEMPLATE = f"""You are a helpful assistant answering questions using ONLY the context below.
If the answer is not contained in the context, reply with exactly this single token and
nothing else: {OUT_OF_SCOPE_SENTINEL}
Do not make anything up.Always give information from the provided document only. if no such information found then politely deny the user. Answer in 1-3 concise sentences, no extra elaboration.

Context:
{{context}}

Question: {{question}}

Answer:"""


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    return _llm


def generate(question: str, chunks) -> str:
    if not chunks:  # Retrieval found no evidence to give the model.
        return OUT_OF_SCOPE_MESSAGE  # Refuse instead of generating an unsupported answer.

    context = "\n\n".join(doc.page_content for doc in chunks)  # Combine retrieved text.
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)  # Fill the RAG prompt.

    llm = _get_llm()  # Obtain the cached Ollama client.
    response = llm.invoke(prompt)  # Ask the local model to answer from the context.
    answer = response.content.strip()  # Remove whitespace around the model response.

    if OUT_OF_SCOPE_SENTINEL in answer:  # Convert the model's internal signal to user wording.
        return OUT_OF_SCOPE_MESSAGE  # Prevent the sentinel from appearing in the CLI.
    return answer  # Return the grounded answer.


CONDENSE_PROMPT_TEMPLATE = """Given the conversation history and a follow-up question, rewrite
the follow-up question as a standalone question that includes any context needed from the
history (e.g. resolve "it", "that", "the scheme" to what they actually refer to).
If the follow-up question is already standalone, return it unchanged.
Output ONLY the rewritten question, nothing else -- no preamble, no quotes.

Chat History:
{history}

Follow-up question: {question}

Standalone question:"""


def condense_question(question: str, history: list[dict]) -> str:
    """Rewrite a follow-up question into a standalone one using prior turns.

    history is a list of {"question": ..., "answer": ...} dicts, oldest first.
    Returns `question` unchanged if there's no history -- no LLM call needed for the
    first turn of a conversation.
    """
    if not history:  # A first-turn question needs no rewriting.
        return question  # Avoid an unnecessary Ollama request.

    history_text = "\n".join(  # Serialize prior turns into prompt-readable text.
        f"User: {turn['question']}\nAssistant: {turn['answer']}" for turn in history
    )
    prompt = CONDENSE_PROMPT_TEMPLATE.format(history=history_text, question=question)  # Add context.

    llm = _get_llm()  # Reuse the same local model instance.
    response = llm.invoke(prompt)  # Rewrite a follow-up as a standalone question.
    return response.content.strip()  # Return only the normalized question text.
