"""Debug: check what attributes the OpenRouter response has for reasoning."""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path("/home/riya/neel-projs/cot-comparisons/.env"))

import openai

MODEL = "Qwen/Qwen3-32B"
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

prompt = "What is 2+2? Answer with just the letter: A) 3 B) 4 C) 5 D) 6"

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=8000,
    temperature=0.7,
    extra_body={"reasoning": {"enabled": True}},
    timeout=60,
)
msg = resp.choices[0].message

print(f"type(msg): {type(msg)}")
print(f"msg.content: {repr(msg.content[:200] if msg.content else None)}")
print()

# Check all attributes
for attr in dir(msg):
    if attr.startswith('_'):
        continue
    val = getattr(msg, attr, None)
    if callable(val):
        continue
    if val is not None and val != '' and val != [] and val != {}:
        if isinstance(val, str) and len(val) > 200:
            print(f"msg.{attr}: <{len(val)} chars> {repr(val[:100])}")
        else:
            print(f"msg.{attr}: {repr(val)}")
