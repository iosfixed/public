import pandas as pd
import httpx
import time
from tqdm import tqdm

# Загружаем данные
df = pd.read_csv("typed_results.csv")
df = df[df["event"].notna()].copy()

# Фильтруем релевантные типы
allowed_types = [
    "none", "revolution or coup", "religious reform", "financial reform"
]
df = df[df["event_type"].isin(allowed_types)].copy()
df["serfdom_related"] = None

# PROMPT
PROMPT_SERFDOM = """You are an assistant that determines whether a historical event is related to the abolition of serfdom or bonded labor.

Instructions:
- If the event clearly refers to the abolition of serfdom, emancipation of peasants or slaves, land reform ending feudal obligations, or the end of forced/bonded labor, return "yes".
- If the event does not relate to these topics, return "no".
- Return only "yes" or "no", no explanation.

Examples:

Event: "Tsar Alexander II issues the Emancipation Manifesto, freeing the Russian serfs."
Answer: yes

Event: "The Edict of Torda proclaims religious freedom in Transylvania."
Answer: no

Event: "Slavery in Portugal is abolished."
Answer: yes

Event: "The French Revolutionary government implements land redistribution and eliminates feudal privileges."
Answer: yes

Event: "The Metropolitan Mojsije Petrović issues a decree to purge the Serbian church of Turkish influence."
Answer: no

Event: "The Polish Sejm abolishes remaining feudal obligations of peasants."
Answer: yes

Event: "Constitutional reforms expand voting rights to middle-class citizens."
Answer: no

Event: "The Ottoman Empire introduces a new tax on landowners."
Answer: no

Event: "Serfdom is formally abolished in the Austrian Empire by Emperor Joseph II."
Answer: yes

Event: "An earthquake destroys Lisbon and kills thousands."
Answer: no
"""

# Параметры API
api_url = "http://192.168.2.87:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
model = "Qwen/Qwen2.5-7B-Instruct"

# Обработка
for idx, row in tqdm(df.iterrows(), total=len(df)):
    event_text = row["event"]

    # Пропускаем уже размеченные
    if pd.notna(row.get("serfdom_related")) and row["serfdom_related"] != "":
        continue

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT_SERFDOM},
            {"role": "user", "content": f'Event: "{event_text}"\nAnswer:'}
        ],
        "temperature": 0.0,
        "max_tokens": 5,
        "stop": ["\n"]
    }

    try:
        response = httpx.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        label = response.json()["choices"][0]["message"]["content"].strip().lower()
        print(f"\n🔹 Event: {event_text}\n🏷️ Serfdom: {label}")
        df.at[idx, "serfdom_related"] = label
    except Exception as e:
        print(f"❌ Error at row {idx}: {e}")
        df.at[idx, "serfdom_related"] = f"ERROR: {e}"

    if idx % 1000 == 0 and idx > 0:
        df.to_csv("serfdom_results_partial.csv", index=False)
        print("💾 Partial save complete.")

    time.sleep(0.2)

# Финальное сохранение
df.to_csv("serfdom_results.csv", index=False)
print("✅ Done. Results saved to serfdom_results.csv")
