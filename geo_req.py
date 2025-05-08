import pandas as pd
import httpx
import time
from tqdm import tqdm

# Загружаем события
df = pd.read_csv("events.csv", header=None, names=["event_text"])
df = df[df["event_text"] != "text"].dropna()

# Инструкция
PROMPT = """You are an assistant that extracts the location and its coordinates from historical event descriptions.

Instructions:
- If the description clearly mentions a place (country, city, region, island, etc.), return its name and approximate coordinates (latitude and longitude).
- If no place is mentioned or it is unclear, return "None" for both fields.
- Format:
Location: [Place or None]
Coordinates: [lat, lon] or None

Examples:

Event: "U.S. President Grover Cleveland is operated on in secret."
Location: United States
Coordinates: [38.9, -77.0]

Event: "The Carlsten fortress in Sweden surrenders to a Danish and Norwegian force after a siege of seven days. Colonel Henrich Danckwardt, who surrendered the fortress to Peter Tordenskjold after being away from it while it was still defensible, is beheaded on September 16."
Location: Sweden
Coordinates: [58.0, 11.4]

Event: "An ammunition explosion on troopship Kuang Yuang near Jiujiang, China, kills 1,200."
Location: Jiujiang, China
Coordinates: [29.7, 115.9]

Event: "Jean-Marc Vacheron founds his watch-making company Vacheron Constantin. To this day, Vacheron Constantin is the oldest watchmaker in the world with an uninterrupted watchmaking history since its foundation."
Location: Geneva, Switzerland
Coordinates: [46.2, 6.1]

Event: "Japan allows U.S. Commodore Perry to come ashore and begin negotiations."
Location: Japan
Coordinates: [35.7, 139.7]

Event: "In France, Romanian inventor Traian Vuia becomes the first person to achieve an unassisted takeoff in a heavier-than-air powered monoplane, but it is incapable of sustained flight."
Location: France
Coordinates: [48.9, 2.3]

Event: "The Metropolitan Mojsije Petrović, leader of the Serbian Orthodox Church within the Habsburg monarchy, issues a 57-point decree to purge the church of the Turkish influence."
Location: Habsburg Monarchy (Austria)
Coordinates: [48.2, 16.4]

Event: "World War II: In the Reich Chancellery, Adolf Hitler holds a secret meeting and states his plans for acquiring 'living space' for the German people (recorded in the Hossbach Memorandum)."
Location: Berlin, Germany
Coordinates: [52.5, 13.4]

Event: "The Woolworth Building opens in New York City. Designed by Cass Gilbert, it is the tallest building in the world on this date, and for more than a decade after."
Location: New York City, United States
Coordinates: [40.7, -74.0]

Event: "Irish-born theatrical manager Bram Stoker's contemporary Gothic horror novel Dracula is first published (in London); it will influence the direction of vampire literature for the following century."
Location: London, United Kingdom
Coordinates: [51.5, -0.1]

Event: "A comet is observed in the night sky."
Location: None
Coordinates: None
"""


# Настройки API
api_url = "http://vLLM_SERVER_IP:8000/v1/chat/completions" # Заменить на IP запущенного vLLM сервера.
headers = {"Content-Type": "application/json"}
model = "Qwen/Qwen2.5-7B-Instruct"

# Результаты
results = []

# Основной цикл
for i, row in tqdm(df.iterrows(), total=len(df)):
    event_text = row["event_text"]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f'Event: "{event_text}"'}
        ],
        "temperature": 0.0,
        "max_tokens": 50,
        "stop": ["\nEvent:", "\n\n"]
    }

    try:
        response = httpx.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()

        # Парсинг Location и Coordinates
        location = None
        latitude = None
        longitude = None

        lines = content.splitlines()
        for line in lines:
            if line.lower().startswith("location:"):
                location = line.split(":", 1)[1].strip()
            elif line.lower().startswith("coordinates:"):
                coords = line.split(":", 1)[1].strip()
                if coords.lower() != "none":
                    try:
                        lat_str, lon_str = coords.strip("[]()").split(",")
                        latitude = float(lat_str.strip())
                        longitude = float(lon_str.strip())
                    except:
                        latitude = longitude = "PARSE_ERROR"

    except Exception as e:
        location = latitude = longitude = f"ERROR: {e}"

    results.append({
        "event": event_text,
        "location": location,
        "latitude": latitude,
        "longitude": longitude
    })

    time.sleep(0.2)

# Сохраняем в CSV
pd.DataFrame(results).to_csv("location_results.csv", index=False)
print("✅ Done. Saved to location_results.csv")
