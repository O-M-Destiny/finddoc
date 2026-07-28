from pathlib import Path
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "multi_query_prompt.txt"


class QueryVariation(BaseModel):
    query: list[str] = Field(
        min_length=3,
        description="Alternative phrasings of the user's question for document retrieval"
    )


def load_prompt_template() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def generate_queries(question: str, llm: ChatGroq, n: int = 3) -> list[str]:
    try:
        structured_llm = llm.with_structured_output(QueryVariation, method='json_mode')

        template: str = load_prompt_template()
        prompt = template.format(question=question, num_queries=n)

        result: QueryVariation = structured_llm.invoke(prompt)

        if not result.query:
            return []

        return result.query

    except Exception as e:
        print(f"Error generating query variations: {e}")
        return []


# llm = ChatGroq(model="openai/gpt-oss-120b")
# question = "Who is the director of the company ?"

# output = generate_queries(question=question, llm=llm)
# print(output)