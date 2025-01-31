from pathlib import Path
from dotenv import load_dotenv
import os


def load_config():
    env_path = Path('.') / '.env'
    load_dotenv(dotenv_path=env_path)

    required_vars = ['OPENAI_API_KEY', 'MODEL_NAME', 'SUPPORT_MODEL_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    return {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'MODEL_NAME': os.getenv('MODEL_NAME'),
        'SUPPORT_MODEL_NAME': os.getenv('SUPPORT_MODEL_NAME'),
    }