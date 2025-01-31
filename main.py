import logging
from http.client import responses
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

    async def get_llama_response(self, query: str) -> str:
        """Get response from support model for thoughts and ideas"""
        sp = """You are an expert on ITMO University. Answer questions based on your knowledge.
Act as a knowledgeable AI assistant that integrates web search results with humor and creativity using Perplexity. 
Respond to user queries as if you were a knowledgeable expert in the field, always providing sources for your claims. 
Conduct real-time web searches to gather relevant information and then aggregate the results into concise, structured, and informative answers that address the main points of the question. 
        """
        try:
            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=config['SUPPORT_MODEL_NAME'],
                messages=[
                    {"role": "system", "content": sp},
                    {"role": "user", "content": query}
                ],
                max_tokens=512
            )
            response = completion.choices[0].message.content
            logger.info(f"Completion: {response}")
            return response
        except Exception as e:
            logger.error(f"Llama processing error: {str(e)}")
            return ""

    async def search_duckduckgo(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search DuckDuckGo for information"""
        logger.debug(f"Starting DuckDuckGo search for query: {query}")
        try:
            search_results = []
            with DDGS() as ddgs:
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
            logger.info(f"DuckDuckGo search results: {search_results}")
            return search_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def get_final_answer(self, query: str, llama_response: str, search_results: List[SearchResult],
                               query_id: int) -> Response:
        """Get final answer from bigger model using both support model response and search results"""
        try:
            system_prompt = """You are an expert about ITMO University and Russian Univerity system. 
            Answer the given question backing up your knowledge with another model response and search results to provide a comprehensive answer about ITMO University.
            Return response in JSON format:
            {
                "answer": int - Number (1-10) of correct option if question has numbered options, or null if no options
                "reasoning": str - Your explanation combining given context and information
                "sources": list - Up to 3 most relevant URLs from search results used
            }"""

            context = json.dumps({
                "llama_response": llama_response,
                "search_results": search_results
            }, ensure_ascii=False, indent=2)

            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                response_format={"type": "json_object"},
                model=config['MODEL_NAME'],
                max_tokens=512,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""Question (ID: {query_id}): {query}

Combined Context:
{context}

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
                id=query_id,
                answer=response_data["answer"],
                reasoning=f"Combined response using {config['SUPPORT_MODEL_NAME']} and {config['MODEL_NAME']} models:\n" + response_data["reasoning"],
                sources=response_data["sources"][:3]
            )

        except Exception as e:
            logger.error(f"Error in final processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

    async def process_query(self, query: Query) -> Response:
        """Process query using parallel pipeline"""
        logger.info(f"Processing query ID {query.id}: {query.query}")

        llama_task = asyncio.create_task(self.get_llama_response(query.query))
        search_task = asyncio.create_task(self.search_duckduckgo(query.query))

        llama_response, search_results = await asyncio.gather(llama_task, search_task)

        return await self.get_final_answer(query.query, llama_response, search_results, query.id)


app = FastAPI()
config = load_config()
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