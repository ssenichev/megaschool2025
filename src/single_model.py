import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from duckduckgo_search import DDGS
import json
import asyncio
import re
from typing import TypedDict
from config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_json(text: str) -> str:
    """
    Clean and parse JSON from text that might contain markdown code blocks or other artifacts.

    Args:
        text (str): Input text that might contain JSON with markdown code blocks

    Returns:
        str: Clean JSON string
    """
    text = re.sub(r'^```json\s*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n```$', '', text, flags=re.MULTILINE)

    text = text.strip()

    try:
        parsed = json.loads(text)
        return json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError as e:
        json_pattern = r'(\{[^}]+\})'
        match = re.search(json_pattern, text)
        if match:
            try:
                parsed = json.loads(match.group(1))
                return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                raise ValueError(f"Could not extract valid JSON from text: {text}") from e
        raise ValueError(f"Invalid JSON format: {text}") from e


class Query(BaseModel):
    query: str
    id: int


class Response(BaseModel):
    id: int
    answer: Optional[int]
    reasoning: str
    sources: List[str]


class SearchResult(TypedDict):
    site: str
    content: str


class ITMOAgent:
    def __init__(self, config: dict):
        if not config.get('OPENAI_API_KEY'):
            raise ValueError("OpenAI API key is missing!")

        if not config.get('MODEL_NAME'):
            raise ValueError("Model name is missing!")

        self.client = OpenAI(
            api_key=config['OPENAI_API_KEY'],
            base_url="https://api.aitunnel.ru/v1/",
        )

        logger.info("ITMOAgent initialized successfully")

    async def search_duckduckgo(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """
        Search DuckDuckGo for information and return URLs with content
        """
        logger.debug(f"Starting DuckDuckGo search for query: {query}")
        search_results = []

        try:
            with DDGS() as ddgs:
                try:
                    news_results = list(ddgs.text(
                        keywords=query,
                        max_results=num_results,
                        region="ru-ru"
                    ))
                    for result in news_results:
                        search_results.append({
                            "site": result['href'],
                            "content": f"Title: {result['title']}\nDescription: {result['body']}"
                        })
                except Exception as e:
                    logger.warning(f"News search failed: {e}")

            if not search_results:
                logger.warning("Unable to find relevant information")
                return []

            logger.info(f"Found {len(search_results)} search results")
            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def get_answer_from_gpt(self, query: str, search_results: List[SearchResult], query_id: int) -> Response:
        """
        Get answer from GPT-4 API with provided search results
        """
        logger.debug(f"Starting GPT processing for query ID: {query_id}")
        try:
            system_prompt = """You are an expert on ITMO University. Your task is to answer questions based on the provided search results.  
            You must respond in a specific JSON format:
            {
                "id": int - The query ID provided
                "answer": int - Number (1-10) of the correct option if the question has numbered options, or null if no options
                "reasoning": str - Your explanation based on the search results
                "sources": list - Array of URLs only from the search results that you actually used to form your answer. Provide NO MORE than 3 links.
            }

            If you cannot find the information in the search results, explain that in the reasoning.
            Never make up information or use prior knowledge - only use the provided search results.
            In case you don't have enough information provided answer by yourself. 
            Ensure you only return numeric values (not strings) for the answer field when it's not null.
            """

            search_context = json.dumps(search_results, ensure_ascii=False, indent=2)

            logger.debug("Making API call to OpenAI")
            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                response_format={"type": "json_object"},
                model=config['MODEL_NAME'],
                max_tokens=512,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""Question (ID: {query_id}): {query}

Search Results:
{search_context}

Provide your answer in the required JSON format."""}
                ]
            )

            response_text = completion.choices[0].message.content
            logger.info(f"Raw GPT response: {response_text}")

            fixed_json = ensure_json(response_text)
            logger.info(f"Fixed GPT response: {fixed_json}")

            response_data = json.loads(fixed_json)

            if response_data["answer"] is not None:
                response_data["answer"] = int(response_data["answer"])

            return Response(
                id=response_data["id"],
                answer=response_data["answer"],
                reasoning=f"Used model: {config['MODEL_NAME']}\n" + response_data["reasoning"],
                sources=response_data["sources"][:3]  # Ensure max 3 sources
            )

        except Exception as e:
            logger.error(f"Error in GPT processing: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error processing request: {str(e)}"
            )

    async def process_query(self, query: Query) -> Response:
        """
        Process incoming query and return formatted response
        """
        logger.info(f"Processing query ID {query.id}: {query.query}")
        search_results = await self.search_duckduckgo(query.query)
        return await self.get_answer_from_gpt(query.query, search_results, query.id)


try:
    config = load_config()
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Error loading configuration: {str(e)}", exc_info=True)
    raise

app = FastAPI()
agent = ITMOAgent(config)


@app.post("/api/request")
async def process_request(query: Query) -> Response:
    """
    Process incoming requests and return formatted responses
    """
    logger.info(f"Received request for query ID: {query.id}")
    try:
        return await agent.process_query(query)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")