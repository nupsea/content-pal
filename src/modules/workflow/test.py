import requests
import json
import argparse

parser = argparse.ArgumentParser(description='Test Content-Pal API')
parser.add_argument('--query', '-q', default="Tom Cruise movies", 
                   help='Search query (default: "Tom Cruise movies")')
parser.add_argument('--port', '-p', default="5001", 
                   help='Port number (default: 5001)')
args = parser.parse_args()

query = args.query
print("User Query:", query)

url = f"http://127.0.0.1:{args.port}/recommend"

try:
    print(f"Making request to {url}...")
    
    data = {"query": query}
    response = requests.post(url, json=data, timeout=30)
    
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        try:
            json_response = response.json()
            print("Success! Response:")
            print(json.dumps(json_response, indent=2))
        except json.JSONDecodeError:
            print("Error: Server returned invalid JSON")
            print("Raw response:", response.text[:500])
    else:
        print(f"Error: Server returned HTTP {response.status_code}")
        print("Raw response:", response.text[:500])
        
except requests.exceptions.ConnectionError:
    print("Error: Connection failed. Is the server running?")
    print(f"Make sure the API is running at {url}")
    
except requests.exceptions.Timeout:
    print("Error: Request timed out after 30 seconds")
    print("The server may be processing or indexing data")
    
except requests.exceptions.RequestException as e:
    print(f"Error: Request failed - {e}")
    
except Exception as e:
    print(f"Error: Unexpected error - {e}")

