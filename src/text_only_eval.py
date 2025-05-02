import pandas as pd
import time
import os
from datetime import datetime
from openai import OpenAI  # LM Studio API

# Setup
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")  # adjust if needed
BATCH_SIZE = 10
INPUT_CSV = "data/questions.csv"
OUTPUT_DIR = "outputs/text_only"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_answer_mistral(question, caption=""):
    prompt = f"""Caption: {caption}

Please provide a brief and accurate final answer to the following question based on the above caption.

Question: {question}
Reasonnin: Reason for the answer
Answer: Final brief answer
"""
    response = client.chat.completions.create(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()

# Load CSV
df = pd.read_csv(INPUT_CSV)

# Process in batches
for i in range(0, len(df), BATCH_SIZE):
    batch = df.iloc[i:i+BATCH_SIZE]
    results = []
    start = time.time()

    for idx, row in batch.iterrows():
        question = row["question"]
        response = generate_answer_mistral(question, caption="")  # text-only
        results.append({
            "index": idx,
            "question": question,
            "response": response
        })

    end = time.time()
    duration = end - start

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/batch_{i}_{timestamp}.csv", index=False)

    with open(f"{OUTPUT_DIR}/timing_log.txt", "a") as log:
        log.write(f"Batch {i}-{i + BATCH_SIZE}: {duration:.2f} sec\n")
