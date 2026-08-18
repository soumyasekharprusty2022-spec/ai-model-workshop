"""CLI chat loop.

Run directly from the repo root (uses absolute imports, so this is safe
unlike src/ingest.py):
    python app.py
"""
from src.config import MAX_HISTORY_TURNS  # Limit how many previous turns are retained.
from src.generator import OUT_OF_SCOPE_MESSAGE  # Identify answers that should not enter memory.
from src.pipeline import run  # Send each user question through the RAG pipeline.


def main():
    print("Local RAG chat -- type 'exit' to quit.\n")  # Tell the user how to use the CLI.
    history = []  # Store prior question/answer pairs, oldest first, for follow-up questions.

    while True:
        question = input("You: ").strip()  # Read the next question and remove extra whitespace.
        if question.lower() in {"exit", "quit"}:  # Accept either command to end the session.
            break  # Leave the loop and terminate the application.
        if not question:  # Ignore empty input instead of querying the models.
            continue  # Return to the prompt.

        result = run(question, history=history)  # Retrieve context and generate an answer.
        print(f"\nAssistant: {result['answer']}")  # Display the generated or fallback answer.
        if result["sources"]:  # Only print a source line when documents supported the answer.
            print(f"Sources: {', '.join(result['sources'])}")  # Display source file paths.
        print()  # Add spacing before the next prompt.

        # Don't pollute history with out-of-scope turns -- a "hi" in the middle of a
        # conversation shouldn't get condensed into later follow-up questions.
        if result["answer"] != OUT_OF_SCOPE_MESSAGE:  # Keep only useful domain answers in memory.
            history.append({"question": question, "answer": result["answer"]})  # Save this turn.
            history = history[-MAX_HISTORY_TURNS:]  # Keep the configured number of recent turns.


if __name__ == "__main__":  # Run the CLI only when this file is executed directly.
    main()  # Start the interactive chat session.
