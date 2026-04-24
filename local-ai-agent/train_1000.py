from pathlib import Path
from brain import EvolutionBrain

b = EvolutionBrain(Path('brain_state.json'))
result = b.train(generations=1000)
print("=== Training Complete ===")
for key, val in result.items():
    print(f"{key}: {val}")
