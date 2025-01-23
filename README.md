# GoalGuru: Previsioni dei Risultati della Premier League

## Introduzione
GoalGuru utilizza una rete neurale avanzata per prevedere i risultati delle partite di calcio. Basato su dati storici e tendenze attuali, questo progetto offre previsioni accurate e affidabili, migliorando l'esperienza degli appassionati di calcio e supportandoli nel seguire e analizzare partite e campionati.

## Obiettivo del Progetto
- **Predizione accurata**: Utilizzo di reti neurali per fornire previsioni realistiche sui risultati delle partite.
- **Engagement degli utenti**: Incoraggiare gli utenti a seguire più partite e campionati.
- **Supporto alle decisioni**: Offrire dati analitici utili per scommesse o analisi sportive.
- **Fidelizzazione**: Fornire un servizio di alta qualità basato su previsioni accurate.

---

## Contenuto del Repository
Il repository contiene le seguenti risorse:
- **Codice principale**: Script Python per l'elaborazione, il training e l'inferenza del modello.
- **Dataset**: file per accedere al dataset utilizzato 
- **Risultati**: Report e visualizzazioni delle performance del modello.
- **Notebook**: File `.ipynb` per analisi ed esplorazione interattiva dei dati.
- **Documentazione**: Dettagli sui processi di modellazione e sperimentazione.

---

## Requisiti
- **Linguaggio**: Python 3.9+
- **Librerie Python richieste**:
  - Manipolazione dati: `pandas`, `numpy`
  - Visualizzazione: `matplotlib`, `seaborn`
  - Modellazione: `scikit-learn`, `tensorflow`
- **Dataset**: /GoalGuru/DATASET

---

## Installazione
1. Clona il repository:
   ```bash
   git clone https://github.com/tuo-username/GoalGuru.git

2. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   
3. /GoalGuru/DATASET scaricare il dataset necessario
   

## Come Riprodurre il Lavoro

### 1. Esplorazione dei dati
Apri /GoalGuru/DATASET per analizzare il dataset in modo preliminare. Questo ti aiuterà a comprendere la struttura e le caratteristiche principali dei dati.

### 2. Pre-elaborazione dei dati
Esegui lo script `preprocessing.py` per pulire e trasformare i dati. Questo passaggio include:
- Rimozione dei valori nulli e delle colonne irrilevanti.
- Codifica delle variabili categoriali.
- Scaling delle variabili numeriche.
  
  Esegui il comando:
  ```bash
  python preprocessing.py

### 3. Addestramento del modello
Addestra il modello utilizzando lo script `training.py`. Puoi configurare i parametri del modello modificando le variabili all'inizio del file.

  Esegui il comando:
  ```bash
  python training.py
  ```

### 4. Predizioni
Genera previsioni sui nuovi dati utilizzando lo script `finaltest.py`. Fornisci un input personalizzato per ottenere le probabilità predette per vittoria, pareggio o sconfitta.

  Esegui il comando:
  ```bash
  python finaltest.py
  ```
























 
   



   
