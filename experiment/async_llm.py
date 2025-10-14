import asyncio
import aiohttp
from time import sleep


async def create_completion(session, prompt, model, temp, topp):
    API_KEY = 'your api key'
    BASE_URL = "your base url" 
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    retry = 0
    while retry < 5:
        try:
            async with session.post(url=f"{BASE_URL}chat/completions",
                json={
                    "model": model,
                    "max_tokens":4000,
                    "temperature": temp,
                    "top_p": topp,
                    "messages": [{"role": "user", "content": prompt}],
                },
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    # 处理请求结果
                    print(result['choices'][0]['message']['content'])
                    print('-----------------------')
                    return result['choices'][0]['message']['content']
                else:
                    print(f"请求失败，状态码: {response.status}")
        except Exception as e:
            retry += 1
            sleep(1)
            print(f"Error: {e}", flush=True)


async def run_async(prompts, model='gpt-4o-mini', temp=0, topp=1):
    async with aiohttp.ClientSession() as session:
        tasks = [create_completion(session, prompt, model, temp, topp ) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
    return responses

if __name__ == "__main__":
    prompts = ['who are you?','what you like?']
    responses = asyncio.run(run_async(prompts))
    print('--------')
    print(responses)
