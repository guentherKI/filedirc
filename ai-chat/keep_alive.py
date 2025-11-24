import requests
import time
import sys

def keep_alive(url, interval=840): # 14 minutes (Render sleeps after 15)
    print(f"💓 Keep-Alive Service started for {url}")
    print(f"⏱️ Pinging every {interval} seconds to prevent sleep...")
    
    while True:
        try:
            # Ping the health endpoint
            response = requests.get(f"{url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ [{time.strftime('%H:%M:%S')}] Ping successful! AI is awake.")
                if 'model_loaded' in data:
                    print(f"   🧠 Model Loaded: {data['model_loaded']}")
            else:
                print(f"⚠️ [{time.strftime('%H:%M:%S')}] Server returned {response.status_code}")
        except Exception as e:
            print(f"❌ [{time.strftime('%H:%M:%S')}] Ping failed: {e}")
            
        time.sleep(interval)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python keep_alive.py <your-render-url>")
        print("Example: python keep_alive.py https://my-ai-chat.onrender.com")
    else:
        keep_alive(sys.argv[1])
