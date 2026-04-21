"""
set_population_48.py
Run this once to resize an existing brain_state.json to exactly 48 bots.
Usage:
    cd local-ai-agent
    python set_population_48.py
"""
import copy
import json
import random
import time
from pathlib import Path

TARGET = 48
state = Path("brain_state.json")

if not state.exists():
    raise SystemExit("brain_state.json not found. Run main.py first to create it.")

payload = json.loads(state.read_text(encoding="utf-8"))
population = payload.get("population", [])

print(f"Population before: {len(population)}")

if len(population) == 0:
    raise SystemExit("Population is empty; cannot expand from existing bots. Use /brain init 48 inside main.py.")

# Grow if needed (clone + mutate existing bots)
while len(population) < TARGET:
    parent = random.choice(population)
    child = copy.deepcopy(parent)
    child["id"] = f"dup-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
    child["score"] = 0.0
    population.append(child)

# Trim if over target
population = population[:TARGET]

payload["population"] = population
payload["saved_at"] = time.time()
state.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print(f"Population after:  {len(population)}  ✓")
