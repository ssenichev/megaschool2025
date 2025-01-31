from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from config import load_config
from langchain.chat_models import ChatOpenAI
from crewai import Agent, Task, Crew
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# FastAPI models
class Query(BaseModel):
    query: str
    id: int


class Response(BaseModel):
    id: int
    answer: Optional[int]
    reasoning: str
    sources: List[str]


# Initialize FastAPI
app = FastAPI()


class ITMOAgent:
    def __init__(self, config: dict):
        if not config.get('OPENAI_API_KEY'):
            raise ValueError("OpenAI API key is missing!")
        if not config.get('MODEL_NAME'):
            raise ValueError("Model name is missing!")

        self.gpt = ChatOpenAI(
            openai_api_base="https://api.aitunnel.ru/v1/",
            openai_api_key=config['OPENAI_API_KEY'],
            model_name=config['MODEL_NAME'],
            max_tokens=None,
        )

        self.general_agent = Agent(
            role="Сотрудник университета ИТМО",
            goal="Отвечать на вопросы в виде JSON, используя свои знания об университете",
            backstory="Ты имеешь многолетний опыт работы в ИТМО и знаешь о всех нюансах университета",
            allow_delegation=False,
            verbose=True,
            llm=self.gpt,
        )

        logger.info("ITMOAgent initialized successfully")

    async def process_query(self, query: Query) -> Response:
        """Process the query using the agent"""
        logger.info(f"Processing query ID {query.id}: {query.query}")

        task = Task(
            description=f"Ответь на вопрос об университете ИТМО: {query.query}",
            expected_output="""
{
    "id": {query.id},
    "answer": str | null,  # Если вопрос требует выбора из предложенных вариантов (например, "1", "2", "3")
    "reasoning": str,  # Подробное и логичное обоснование твоего ответа с интересными фактами
    "sources": List[str]  # Укажи не более ТРЕХ ссылок из поисковой выдачи, которые ты использовал для ответа
}
""",
            agent=self.general_agent
        )

        try:
            crew = Crew(
                agents=[self.general_agent],
                tasks=[task],
                verbose=True
            )

            result = crew.kickoff()
            logger.info(f"Raw agent response: {result}")

            result_json = json.loads(result)

            response = Response(
                id=result_json["id"],
                answer=result_json["answer"],
                reasoning=f"Used model: {config['MODEL_NAME']}\n" + result_json["reasoning"],
                sources=result_json["sources"]
            )
            return response

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to parse agent response"
            )
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing request: {str(e)}"
            )


# Load configuration
try:
    config = load_config()
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Error loading configuration: {str(e)}")
    raise

# Initialize the ITMO agent
itmo_agent = ITMOAgent(config)


@app.post("/api/request")
async def handle_request(query: Query) -> Response:
    """Handle incoming requests"""
    logger.info(f"Received request for query ID: {query.id}")
    return await itmo_agent.process_query(query)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")