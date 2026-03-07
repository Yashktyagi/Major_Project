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

client = genai.Client(api_key="AIzaSyBUJFE6OuN3yN6IHZ9mjMc9AI7o7m8RHqI")

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

def build_batch_prompt(base_prompt, headlines):
    """
    Insert multiple headlines into prompt
    """
    headline_block = ""

    for i, h in enumerate(headlines):
        headline_block += f"{i+1}. {h}\n"

    return base_prompt.replace("<INSERT_HEADLINES_LIST>", headline_block)

def parse_batch_response(response_text):
    """
    Extract JSON list safely
    """
    try:
        return json.loads(response_text)
    except:
        start = response_text.find("[")
        end = response_text.rfind("]") + 1
        json_str = response_text[start:end]
        return json.loads(json_str)

# def annotate_dataframe_batch(df, text_column, base_prompt, batch_size=10, model_name="gemini-3-flash-preview"):
#
#     results = []
#
#     for start in range(0, len(df), batch_size):
#         end = start + batch_size
#         batch_df = df.iloc[start:end]
#         headlines = batch_df[text_column].tolist()
#
#         try:
#             full_prompt = build_batch_prompt(base_prompt, headlines)
#
#             response = client.models.generate_content(
#                 model=model_name,
#                 contents=full_prompt,
#                 config={
#                     "temperature": 0.0
#                     }
#             )
#
#             response_text = response.text.strip()
#             parsed_outputs = parse_batch_response(response_text)
#
#             for output in parsed_outputs:
#                 results.append({
#                     "label": output.get("label"),
#                     "confidence": output.get("confidence")
#                 })
#
#         except Exception as e:
#             print(f"Batch error at rows {start}-{end}: {e}")
#
#             # Fill batch with None if failure
#             for _ in headlines:
#                 results.append({
#                     "label": None,
#                     "confidence": None
#                 })
#
#     result_df = pd.DataFrame(results)
#     return pd.concat([df.reset_index(drop=True), result_df], axis=1)

def process_batch(batch_df, text_column, base_prompt, model_name):
    headlines = batch_df[text_column].tolist()
    full_prompt = build_batch_prompt(base_prompt, headlines)

    response = client.models.generate_content(
        model=model_name,
        contents=full_prompt,
        config={"temperature": 0.0}
    )

    response_text = response.text.strip()
    parsed_outputs = parse_batch_response(response_text)

    return parsed_outputs

def annotate_dataframe_parallel(df, text_column, base_prompt,
                                batch_size=20,
                                max_workers=20,
                                model_name="gemini-3-flash-preview"):

    batches = []
    for start in range(0, len(df), batch_size):
        batches.append(df.iloc[start:start+batch_size])

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_batch, batch, text_column, base_prompt, model_name)
            for batch in batches
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                batch_results = future.result()
                for output in batch_results:
                    results.append({
                        "label": output.get("label"),
                        "confidence": output.get("confidence")
                    })
            except Exception as e:
                print("Batch failed:", e)

    result_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), result_df], axis=1)

# annotated_df = annotate_dataframe_batch(
#     df,
#     "headline",
#     prompt,
#     batch_size=20
# )

annotated_df = annotate_dataframe_parallel(
    df,
    "headline",
    prompt,
    batch_size=20,
    max_workers=25  # adjust safely
)

annotated_df.to_csv("dataset_labeled.csv", index=False)