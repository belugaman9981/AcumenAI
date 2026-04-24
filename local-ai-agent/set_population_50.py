import copy
import json
import random
import time
from pathlib import Path


state = Path("brain_state.json")
payload = json.loads(state.read_text(encoding="utf-8"))
population = payload.get("population", [])

print("BEFORE_POP", len(population))

target = 50
if len(population) == 0:
	raise SystemExit("Population is empty; cannot expand from existing bots.")

while len(population) < target:
	parent = random.choice(population)
	child = copy.deepcopy(parent)
	child["id"] = f"dup-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
	child["score"] = 0.0
	population.append(child)

if len(population) > target:
	population = population[:target]

payload["population"] = population
payload["saved_at"] = time.time()
state.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print("AFTER_POP", len(population))
