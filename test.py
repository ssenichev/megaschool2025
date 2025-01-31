import aiohttp
import asyncio
import json
import time
import ssl
from typing import List

TEST_QUESTIONS = [
                     "В каком городе находится главный кампус Университета ИТМО?\n1. Москва\n2. Санкт-Петербург\n3. Екатеринбург\n4. Нижний Новгород",
                     "В каком году был основан Университет ИТМО?\n1. 1900\n2. 1930\n3. 1880\n4. 1920",
                     "Кто является ректором Университета ИТМО?\n1. Владимир Васильев\n2. Виктор Садовничий\n3. Николай Кропачев\n4. Дмитрий Ливанов",
                     "Сколько факультетов в Университете ИТМО?\n1. 5\n2. 8\n3. 12\n4. 15",
                     "Какой основной цвет в логотипе ИТМО?\n1. Синий\n2. Красный\n3. Зеленый\n4. Черный"
                 ] * 4

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


async def send_request(session: aiohttp.ClientSession, endpoint: str, question: str, question_id: int) -> dict:
    payload = {
        "query": question,
        "id": question_id
    }

    start_time = time.time()
    try:
        async with session.post(endpoint, json=payload, ssl=ssl_context) as response:
            response_time = time.time() - start_time
            response_json = await response.json()
            return {
                "id": question_id,
                "status": response.status,
                "response_time": response_time,
                "response": response_json
            }
    except Exception as e:
        return {
            "id": question_id,
            "status": "error",
            "response_time": time.time() - start_time,
            "error": str(e)
        }


async def run_tests(endpoint: str, questions: List[str]):
    conn = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = []
        for i, question in enumerate(TEST_QUESTIONS):
            task = send_request(session, endpoint, question, i + 1)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        successful_requests = sum(1 for r in results if r["status"] == 200)
        total_time = max(r["response_time"] for r in results)
        avg_time = sum(r["response_time"] for r in results) / len(results)

        print(f"\nTest Results:")
        print(f"Total requests: {len(results)}")
        print(f"Successful requests: {successful_requests}")
        print(f"Failed requests: {len(results) - successful_requests}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average response time: {avg_time:.2f} seconds")

        with open("test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    for i in range(1):
        asyncio.run(
            run_tests(
                # endpoint="https://itmo-agent-a1b39bda5a86.herokuapp.com/api/request",
                endpoint="http://localhost:8000/api/request",
                questions=TEST_QUESTIONS
            )
        )