import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ..retriever.retriever_pipeline import RAGRetriever
from ..generation.generation import AnswerGenerator
from .schemas import QuestionRequest, AnswerResponse

app = FastAPI(title="My Rag App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


retriever = RAGRetriever(llm_model="openai/gpt-oss-120b")
generator = AnswerGenerator(retriever=retriever, llm_model="openai/gpt-oss-120b")

@app.post("/chat/stream")
def chat_stream(request: QuestionRequest):
    def event_generator():
        try:
            for event in generator.answer_stream(question=request.question, session_id=request.session_id):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"Error during streaming: {e}")
            error_event = {"type": "error", "content": "Something went wrong generating the answer."}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok"}