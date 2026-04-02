# MENNY, Multiple ENsemble Neural-like strategY

Player per Risk che usa un ensemble di alberi decisionali con pesi aggiornati in modo online, ispirandosi al funzionamento di XGBoost ma adattato a un contesto di gioco sequenziale dove non hai un dataset fisso ma una partita che si evolve turno per turno.

## Come funziona passo per passo

### 1. Cosa "vede" MENNY prima di decidere, le feature

Prima di ogni decisione, MENNY estrae 10 numeri dalla situazione di gioco corrente. Tutti normalizzati tra 0 e 1 per essere confrontabili tra loro:

- **army_ratio**, quante armate hai tu rispetto al totale sul tabellone. Se è 0.5 hai metà delle armate in gioco.
- **country_ratio**, quanti paesi controlli rispetto al totale. Simile al precedente ma conta i territori, non le truppe.
- **border_ratio**, quanti dei tuoi paesi confinano con un nemico, sul totale dei tuoi paesi. Alto = sei molto esposto.
- **avf_border_pressure**, mediamente quanti nemici affacciano su ogni tuo paese di confine. Un paese con 4 nemici vicini è sotto pressione maggiore di uno con 1 solo.
- **continent_progress**, mediamente a che punto sei nel completare i continenti. 1.0 = li controlli tutti, 0.0 = non ne controlli nessuno intero.
- **continentr_bonus**, quante armate bonus ti danno i continenti che controlli già, normalizzato sul massimo plausibile (12).
- **enemy_count_ratio**, quanti avversari sono ancora vivi. Utile perché la strategia cambia molto tra inizio partita (tanti nemici) e fine (uno contro uno).
- **weakest_ratio**, quanto è debole il tuo paese più debole rispetto alla tua media. Se è basso hai un buco nella difesa.
- **strongest_ratio**, quanto è forte il tuo paese più forte rispetto alla tua media. Alto = hai una riserva concentrata da qualche parte.
- **interior_ratio**, quanti dei tuoi paesi non toccano nessun nemico. Alto = hai un retroterra sicuro.

Questi 10 numeri formano il "ritratto" della situazione che ogni albero usa per decidere.

---

### 2. Gli alberi decisionali

Un albero decisionale binario funziona così: parte dalla radice e a ogni nodo interno fa una domanda del tipo *"la feature numero 3 è maggiore di 0.45?"*. Se sì va a destra, se no va a sinistra. Quando arriva a una foglia, lì c'è scritto quale mossa consigliare.

MENNY costruisce 12 di questi alberi, ognuno profondo 4 livelli (quindi con fino a 16 foglie). All'inizio vengono costruiti in modo casuale: ogni nodo interno sceglie una feature a caso e una soglia a caso tra 0.2 e 0.8. Le foglie sono assegnate casualmente tra le mosse disponibili.

Questo può sembrare inutile, un albero casuale non sa niente. Ma è esattamente qui che entra in gioco il sistema dei pesi.

---

### 3. Come decide: voto pesato

Quando è il momento di scegliere una mossa, MENNY fa così:

1. Estrae le 10 feature dalla situazione corrente
2. Fa "scorrere" quelle feature attraverso tutti e 12 gli alberi
3. Ogni albero arriva a una foglia e dice la sua mossa preferita
4. I voti vengono sommati, ma pesati: se un albero ha peso 0.15 e uno ha peso 0.03, il primo conta 5 volte di più
5. Vince la mossa con il punteggio pesato più alto

All'inizio i pesi sono tutti uguali (1/12 ciascuno). Ma cambiano nel tempo.

C'è anche una componente di esplorazione: con probabilità epsilon (parte al 25%, scende fino al 4%) viene scelta una mossa a caso invece di seguire il voto. Questo serve a far sperimentare il sistema anche mosse che normalmente non verrebbero scelte.

---

### 4. Le macro-azioni

MENNY non sceglie mosse singole (attacca il paese X con Y armate) ma macro-azioni, cioè strategie ad alto livello già implementate nel player base:

**Piazzamento armate:**
- `place_contested`, metti le truppe sul confine più conteso
- `place_weakest`, rinforza il paese più debole
- `place_continent`, rinforza dove sei più vicino a completare un continente

**Attacco:**
- `attack_easy`, attacca solo dove sei nettamente superiore
- `attack_fill`, elimina piccole sacche nemiche circondate
- `attack_consolidate`, riduci il numero di fronti aperti
- `attack_split`, espanditi in più direzioni contemporaneamente
- `attack_pass`, non attaccare questo turno

**Spostamento finale:**
- `fortify_border`, sposta truppe dall'interno verso il confine più esposto
- `fortify_pass`, non spostare niente

---

### 5. Come impara: aggiornamento dei pesi

Dopo ogni azione, MENNY calcola un punteggio (reward) che misura se la situazione è migliorata o peggiorata rispetto a prima dell'azione. Il calcolo tiene conto di:

- quanti paesi hai guadagnato o perso (pesano molto, +2.0 per paese)
- quante armate hai guadagnato o perso (pesano meno, +0.3 per armata)
- se stai dominando la mappa (bonus progressivo oltre il 30% dei territori)
- se hai conquistato un continente intero in questo turno (bonus +5.0)
- quanti fronti aperti hai (penalità -0.1 per ogni paese di confine)

Con questo punteggio, i pesi degli alberi vengono aggiornati così:

- gli alberi che hanno votato la mossa scelta vengono **rinforzati**: il loro peso viene moltiplicato per `exp(0.15 × reward)`. Se il reward è positivo il peso cresce, se è negativo scende.
- gli alberi che hanno votato qualcos'altro vengono **penalizzati parzialmente**: il loro peso viene moltiplicato per `exp(-0.15 × |reward| × 0.5)`. La penalità è dimezzata rispetto al rinforzo perché non è detto che la loro scelta sarebbe andata peggio, semplicemente non è stata provata.

Dopo l'update tutti i pesi vengono rinormalizzati in modo che sommino a 1.

Questo meccanismo è simile a quello di AdaBoost/XGBoost: i "classificatori" (alberi) che sbagliano perdono importanza, quelli che indovinano la guadagnano. La differenza è che qui non c'è un dataset fisso ma una partita che continua, e l'aggiornamento avviene dopo ogni singola azione, non solo a fine partita.

A fine partita arriva anche un reward terminale grande: +50 se ha vinto, -25 se ha perso. Questo serve a segnare in modo netto la differenza tra una strategia complessivamente buona e una cattiva.

---

### 6. Pruning e rigenerazione degli alberi

Dopo molte partite alcuni alberi si trovano con un peso bassissimo, sotto 0.01, perché hanno suggerito sistematicamente mosse sbagliate. Tenerli nell'ensemble non serve: pesano pochissimo e occupano solo calcolo.

Ogni 30 partite MENNY controlla i pesi e rimpiazza gli alberi troppo deboli con alberi nuovi. Questi nuovi alberi non sono però completamente casuali come quelli iniziali: le loro foglie vengono distribuite con una probabilità proporzionale a quanto ogni macro-azione ha guadagnato in totale durante la storia della partita. Se `attack_easy` ha accumulato reward positivo e `attack_split` ha accumulato reward negativo, i nuovi alberi tenderanno ad avere più foglie con `attack_easy`.

È l'equivalente di aggiungere un nuovo weak learner che corregge l'errore residuo, come fa XGBoost quando aggiunge un albero alla sequenza.

---

### 7. Epsilon decay

L'esplorazione diminuisce nel tempo. Parte al 25% (una mossa su quattro è casuale) e si moltiplica per 0.992 dopo ogni partita, fino a un minimo del 4%. Questo permette di esplorare molto all'inizio, quando l'ensemble non ha ancora imparato niente di utile, e di sfruttare quello che ha imparato man mano che le partite avanzano.

---

## Parametri principali

| Parametro | Default | Significato |
|---|---|---|
| `n_trees` | 12 | quanti alberi nell'ensemble |
| `depth` | 4 | profondità massima di ogni albero |
| `learning_rate` | 0.15 | quanto velocemente cambiano i pesi |
| `epsilon` | 0.25 | esplorazione iniziale |

Aumentare `n_trees` rende le decisioni più stabili ma rallenta l'apprendimento perché ogni albero riceve meno "credito" individuale. Aumentare `depth` dà agli alberi più capacità espressiva ma li rende più sensibili al rumore. Con i default attuali dovrebbe convergere a qualcosa di ragionevole in 200-300 partite.
