
- Ogni **individuo** = un vettore di pesi per ogni macro in ogni fase
- **Fitness** = win-rate su N partite
- **Selezione** torneo + **crossover** + **mutazione**
- Evolve la popolazione per generazioni, salva il best

---

**Architettura GA:**

Il **genoma** è un dizionario `{fase → {macro → peso_float}}` — niente Q-table, niente aggiornamenti online. Ogni individuo è una strategia fissa.

La **selezione macro** usa softmax sui pesi invece di greedy, quindi rimane stocastica ma proporzionale a ciò che l'evoluzione ha premiato.

**Pipeline per generazione:**
1. Valuta tutti i `POP_SIZE=30` individui (`EVAL_GAMES=20` partite ciascuno)
2. **Élitismo**: i 2 migliori passano intatti
3. **Torneo** (k=3) → **crossover uniforme** (ogni gene preso a caso da uno dei 2 genitori) → **mutazione gaussiana** (15% chance per gene, σ=0.3)
4. Ripeti per `NUM_GENERATIONS=200` generazioni

**Output:** `ga_best.json` si aggiorna ogni volta che si batte il record, `ga_stats.json` tiene la curva fitness completa per plottare l'evoluzione.