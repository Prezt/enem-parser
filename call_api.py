import requests

def call_local_ai(prompt, model="qwen2.5:7b-instruct"):
    url = "http://localhost:11434/api/generate"

    response = requests.post(url, json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 8192,
            "num_predict": 1500,
            "top_p": 0.9
        }
    })

    if response.status_code != 200:
        raise Exception(response.text)

    return response.json()["response"]