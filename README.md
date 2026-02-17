# Risk
Risk - Data science 

Chi non riska non roska :D

# Installation

### Prerequisites
- Python 3.10+

### Install

```bash
git clone 
cd Risk

python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
pip install -e .
```

### Loading the Map

The board is defined in `data/risk_map.csv` and loaded via `Map.from_csv()`.
The path is resolved relative to the source file so it works regardless of
where you run the project from:

```python
from Risk.map import Map

risk_map = Map.from_csv("data/risk_map.csv")
```

The CSV has one row per country with the following columns:

| Column | Description |
|---|---|
| `country` | Country name |
| `continent` | Continent name |
| `continent_reward` | Bonus armies for controlling the full continent |
| `neighbors` | Semicolon separated list of adjacent country names |

To add or edit countries, open `data/risk_map.csv` directly — no code changes needed.

## See:
  - https://refactoring.guru/design-patterns/state
  - https://youtu.be/5g48vXnnf30?si=dlhIgDl0X901QuQG
