import urllib.request, json, os
from dotenv import load_dotenv
load_dotenv()

key = os.getenv('GEMINI_API_KEY')

req = urllib.request.Request(
    f'https://generativelanguage.googleapis.com/v1beta/models?key={key}',
    headers={'Content-Type': 'application/json'}
)
try:
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
        for m in data['models']:
            print(m['name'])
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()}")