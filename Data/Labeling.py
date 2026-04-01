import requests
import pandas as pd
import json
import re
import time
import os
from tqdm import tqdm
from openai import OpenAI
from google import genai
import concurrent.futures

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

client = genai.Client(api_key="")

df = pd.read_csv("dataset.csv")

df.drop("label", axis=1)

prompt = """
You are an expert linguistic analyst specializing in sarcasm detection in Indian political headlines.

Your task is to classify the given headline as "sarcastic" or "non-sarcastic" using structured reasoning. You will receive multiple headlines.

Follow these steps strictly:

STEP 1: Determine the literal meaning of the headline.
STEP 2: Determine the implied or intended meaning (if different from literal meaning).
STEP 3: Evaluate the headline using the following sarcasm rule categories.

IMPORTANT:
Only classify as "sarcastic" if there is clear and explicit irony, exaggeration, mock praise, absurdity, or contradiction in the wording itself.

Do NOT infer sarcasm from political disagreement, controversy, or potential criticism.

If the headline is neutral news reporting, it must be labeled "non-sarcastic".

When uncertain, choose "non-sarcastic".

---------------------------
SARCASM RULE CATEGORIES
---------------------------

A. Sentiment & Polarity Rules
1. Sentiment–Situation Mismatch:
   Positive tone describing negative events OR negative tone describing positive events.
2. Polarity Reversal:
   Surface sentiment opposite to implied target sentiment.
3. Mock Praise:
   Praise used to imply criticism.

B. Hyperbole & Exaggeration Rules
4. Extreme exaggeration or overgeneralization (e.g., “entire nation”, “everyone agrees”).
5. Absurd or impossible outcomes.
6. Semantic disproportion (small issue framed as national triumph/disaster).

C. Logical & Causal Incongruity Rules
7. Absurd cause–effect relationship.
8. Policy–Outcome inversion (harmful policy framed as beneficial).
9. Statistical manipulation humor (redefining metrics to claim success).

D. Structural & Semantic Contradictions
10. Internal paradox or contradiction (e.g., “transparent corruption”).
11. Unexpected role reversal.
12. Understatement of severe crisis.

E. Linguistic Markers
13. Ironic intensifiers (e.g., “Wow”, “Historic”, “Masterstroke”) in contradictory contexts.
14. Quotation mark skepticism (e.g., “development”, “transparency”).
15. Faux neutral journalism tone masking absurdity.

F. Political & Cultural Context (Indian Politics)
16. Election slogan reframing (e.g., manifesto promises used ironically).
17. Bureaucratic absurdity framed as innovation.
18. Reference to widely debated controversies framed as achievements.

---------------------------
STEP 4:
List all triggered rule numbers.

STEP 5:
Based on rule triggers and strength of incongruity, assign label:
- "sarcastic"
- "non-sarcastic"

STEP 6:
Assign a sarcasm confidence score between 0 and 1.

---------------------------
OUTPUT FORMAT (STRICT JSON ONLY)
Each element must correspond to one headline in the same order.

Format:

[
  {"label": "...", "confidence": 0.0-1.0},
  {"label": "...", "confidence": 0.0-1.0}
]

Do NOT show your reasoning.
Do NOT show step-by-step analysis.
Think internally.
Output ONLY the final JSON object.
No text before or after JSON.

Headlines:
<INSERT_HEADLINES_LIST>
"""

import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------
# Prompt Builder
# ----------------------------
def build_prompt(base_prompt, headline):
    return base_prompt.replace("<INSERT HEADLINE HERE>", headline)

# ----------------------------
# JSON Parser
# ----------------------------
def parse_response(response_text):
    try:
        return json.loads(response_text)
    except:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        json_str = response_text[start:end]
        return json.loads(json_str)

# ----------------------------
# API Call
# ----------------------------
def annotate_headline(headline, base_prompt, model_name="gemini-3-flash-preview"):

    full_prompt = build_prompt(base_prompt, headline)

    response = client.models.generate_content(
        model=model_name,
        contents=full_prompt
    )

    response_text = response.text.strip()

    parsed = parse_response(response_text)

    return {
        "label": parsed.get("label"),
        "confidence": parsed.get("confidence")
    }

# ----------------------------
# Worker Function
# ----------------------------
def worker(idx, row, text_column, base_prompt):
    try:
        headline = row[text_column]
        result = annotate_headline(headline, base_prompt)
        return idx, result

    except Exception as e:
        print(f"Error at index {idx}: {e}")
        return idx, {"label": None, "confidence": None}

# ----------------------------
# Parallel Annotation Function
# ----------------------------
def annotate_dataframe_parallel(df, text_column, base_prompt, max_workers=50):

    results = [None] * len(df)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = [
            executor.submit(worker, idx, row, text_column, base_prompt)
            for idx, row in df.iterrows()
        ]

        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    result_df = pd.DataFrame(results)

    return pd.concat([df.reset_index(drop=True), result_df], axis=1)

# ----------------------------
# Run Annotation
# ----------------------------
annotated_df = annotate_dataframe_parallel(
    df.head(10000),
    text_column="headline",   # change if column name differs
    base_prompt=prompt,
    max_workers=50
)

print("Annotation Complete")
print(annotated_df.head())

annotated_df.to_csv("/content/drive/MyDrive/dataset_labeled2.csv", index=False)