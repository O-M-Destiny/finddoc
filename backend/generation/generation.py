import os
import re

from ..retriever.retriever_pipeline import RAGRetriever

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory


class AnswerGenerator:
    def __init__(self, retriever: RAGRetriever, llm_model: str, history_window: int = 4):
        self.retriever = retriever
        self.llm = ChatGroq(model=llm_model)
        self.history_window = history_window
        self.redis_url = os.environ["REDIS_URL"]

        # ---- prompt file paths (script dir + prompts folder) ----
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        answer_prompt_path = os.path.join(base_dir, "prompts", "answer_prompt.txt")
        contextualize_prompt_path = os.path.join(base_dir, "prompts", "contextualize_prompt.txt")

        # ---- open each prompt file once, at init ----
        with open(answer_prompt_path, "r") as f:
            self.answer_prompt_template = f.read()

        with open(contextualize_prompt_path, "r") as f:
            self.contextualize_prompt_template = f.read()

    # ---- _get_history ----
    def _get_history(self, session_id: str) -> RedisChatMessageHistory:
        return RedisChatMessageHistory(session_id=session_id, url=self.redis_url, ttl=3600)

    # ---- _format_history ----
    def _format_history(self, history: BaseChatMessageHistory) -> str:
        recent = history.messages[-(self.history_window * 2):]
        lines = [f"{'Q' if msg.type == 'human' else 'A'}: {msg.content}" for msg in recent]
        return "\n".join(lines)

    # ---- _handle_casual_query ----
    def _handle_casual_query(self, question: str) -> str | None:
        normalized = question.strip().lower().strip("!?.")

        greetings = {
            "hi", "hello", "hey", "hi there", "hello there", "yo",
            "good morning", "good afternoon", "good evening",
            "how are you", "how's it going", "what's up", "whats up",
        }

        identity_questions = {
            "what is your name", "who are you", "what's your name",
            "what is your purpose", "what can you do", "what do you do",
        }

        if normalized in greetings:
            return "Hi! I'm here to answer questions about NVIDIA's 2025 Annual Report — ask me anything about their financials, executives, or corporate governance."

        if normalized in identity_questions:
            return "I'm an AI assistant built to answer questions about NVIDIA's 2025 Annual Report using retrieval-augmented generation over the actual document text. What would you like to know?"

        return None

    # ---- _contextualize_question ----
    def _contextualize_question(self, question: str, history: BaseChatMessageHistory) -> str:
        if not history.messages:
            return question
            
        formatted_history = self._format_history(history)
        prompt = self.contextualize_prompt_template.format(history=formatted_history, question=question)

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"Error contextualizing question: {e}")
            return question  # fail safe: fall back to raw question

    def _extract_context_lightweight(self, doc: Document) -> str:
        """
        Builds the context representation for a lightweight, text-only,
        limited-context-window LLM (e.g. gpt-oss-120b via Groq).

        Per-type extraction rules and reasoning:

        - Text chunks: page_content only. It's already the real, precise
        source text — ai_summary would be redundant and would just add
        extra tokens for no new information.

        - Table chunks: page_content AND ai_summary. Raw table content
        (HTML/pdfplumber-extracted text) preserves exact figures but can
        be structurally hard for an LLM to parse cleanly on its own.
        ai_summary adds interpretive framing alongside the raw data,
        helping the model reason about what the table actually shows
        without losing the precise numbers.

        - Image chunks: ai_summary only, as plain text. This model is
        text-only and cannot interpret images_base64 directly, so the
        AI-generated description from ChunkEnricher is the only usable
        representation of an image chunk here. images_base64 is never
        sent in this path, regardless of chunk type, to avoid wasting
        the limited context window on data this model can't use anyway.
        """
        page = doc.metadata.get("page_number")
        citation_tag = f"[Page {page}] " if page else ""

        types = doc.metadata.get("types_found", ["text"])
        if isinstance(types, str):
            types = [types]

        if "image" in types:
            return citation_tag + doc.metadata.get("ai_summary", "")

        if "table" in types:
            ai_summary = doc.metadata.get("ai_summary", "")
            if ai_summary:
                return f"{citation_tag}{doc.page_content}\n\nSummary: {ai_summary}"
            return citation_tag + doc.page_content

        return citation_tag + doc.page_content

    def _build_lightweight_context(self, documents: list[Document], max_chunks: int = 5) -> str:
        blocks = [self._extract_context_lightweight(doc) for doc in documents[:max_chunks]]
        return "\n\n---\n\n".join(blocks)

    def _build_prompt(self, question: str, context: str) -> str:
        return self.answer_prompt_template.format(context=context, question=question)

    def answer(self, question: str, session_id: str, n: int = 3, k: int = 3) -> str:
        history = self._get_history(session_id)

        # skip retrieval entirely for greetings/identity questions
        casual_response = self._handle_casual_query(question)
        if casual_response is not None:
            history.add_user_message(question)
            history.add_ai_message(casual_response)
            return casual_response

        standalone_question = self._contextualize_question(question, history)

        documents = self.retriever.hybrid_retriever(question=standalone_question, n=n, k=k)
        context = self._build_lightweight_context(documents)
        prompt = self._build_prompt(question=standalone_question, context=context)

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer_text = response.content
        except Exception as e:
            print(f"Error generating answer: {e}")
            answer_text = "Something went wrong generating the answer."

        history.add_user_message(question)
        history.add_ai_message(answer_text)

        return answer_text

    def answer_stream(self, question: str, session_id: str, n: int = 3, k: int = 3):
        history = self._get_history(session_id)

        casual_response = self._handle_casual_query(question)
        if casual_response is not None:
            history.add_user_message(question)
            history.add_ai_message(casual_response)
            yield {"type": "answer_token", "content": casual_response}
            return

        standalone_question = self._contextualize_question(question, history)

        queries = self.retriever.generate_query_variations(standalone_question, n=n)
        yield {"type": "queries", "content": queries}

        documents = self.retriever.hybrid_retriever_with_queries(queries=queries, k=k)
        context = self._build_lightweight_context(documents)
        prompt = self._build_prompt(question=standalone_question, context=context)

        full_answer = ""
        try:
            for chunk in self.llm.stream([HumanMessage(content=prompt)]):
                token = chunk.content

                if not token:
                    continue

                full_answer += token

                yield {
                    "type": "answer_token",
                    "content": token
                }
        except Exception as e:
            print(f"Error generating answer: {e}")
            full_answer = "Something went wrong generating the answer."
            yield {"type": "answer_token", "content": full_answer}

        history.add_user_message(question)
        history.add_ai_message(full_answer)


if __name__ == "__main__":
    retriever = RAGRetriever(llm_model="openai/gpt-oss-120b")
    generator = AnswerGenerator(retriever=retriever, llm_model="openai/gpt-oss-120b")

    session_id, test_query = "test-stream-1", "When was Nvidia Founded ?"

    for event in generator.answer_stream(question=test_query, session_id=session_id):
        print(event)


#Tackle Citation, irrelevant queries, Add rate limiting, A re-ranker, RedisMemory