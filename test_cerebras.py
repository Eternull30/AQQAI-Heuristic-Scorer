import urllib.request, json, os
from dotenv import load_dotenv
load_dotenv()

key = os.getenv('CEREBRAS_API_KEY')

req = urllib.request.Request(
    'https://api.cerebras.ai/v1/models',
    headers={
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json',
        'User-Agent': 'python-requests/2.31.0',
        'Accept': 'application/json',
    }
)
try:
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
        print("Available models:")
        for m in data['data']:
            print(f"  {m['id']}")
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()}")