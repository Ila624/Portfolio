# EcoBot — Agente AI per l'analisi energetica residenziale

Agente conversazionale che analizza i dati di produzione fotovoltaica di un'azienda locale
e di consumo di una famiglia, rispondendo a domande in linguaggio naturale e generando
grafici.

## Architettura scelta

| Componente | Scelta | Motivazione |
|---|---|---|
| Framework agente | **LangChain**  | Framework per agenti tool-calling; sufficiente per un single-agent con reasoning loop, senza la complessità aggiuntiva di un'architettura multi-agente. |
| Ambiente di sviluppo | **Jupyter locale** (Anaconda) | Maggiore controllo sull'ambiente rispetto a Colab, nessuna dipendenza da tunnel esterni per l'avvio di Streamlit. |
| LLM | **Groq** (`openai/gpt-oss-120b`) | Si è scelto Groq, che offre un free tier funzionante senza richiedere setup di billing. Il modello inizialmente scelto (llama-3.3-70b-versatile) è stato in seguito deprecato da Groq; si è passati al modello ufficialmente consigliato in sostituzione, openai/gpt-oss-120b. |
| Memoria conversazionale | `MemorySaver` (checkpointer di LangGraph) | Integrato nativamente in `create_agent`; gestisce lo storico dei messaggi per thread senza bisogno di ricostruire manualmente il contesto ad ogni turno. |
| RAG (base di conoscenza) | **FAISS** + embeddings locali `sentence-transformers/all-MiniLM-L6-v2`, con **MultiQueryRetriever** (riformulazioni via LLM) e **reranking** via cross-encoder locale (`cross-encoder/ms-marco-MiniLM-L-6-v2`) | Permette all'agente di rispondere a domande concettuali (non solo sui dati numerici) appoggiandosi a documenti scritti da EcoGrid, senza inventare risposte. Pipeline a tre stadi (riformulazioni della domanda → retrieval per similarità → reranking sui top 5) per gestire sia il vocabolario semantico sia la pertinenza logica effettiva. Tutto locale tranne le riformulazioni (usano l'LLM Groq già configurato), nessuna nuova API key. |

## Struttura della cartella

```
├── agent.ipynb              # Notebook di sviluppo: costruzione e test dell'agente (Sezione 1)
├── app.py                   # Applicazione Streamlit (Sezione 2 — file separato, vedi nota sotto)
├── requirements.txt
├── .env                      # Chiavi API (non incluso nel repository/zip pubblico)
├── knowledge_base/          # Documenti sorgente del RAG (vedi sezione RAG più sotto)
│   ├── guida_risparmio_energetico.md
│   └── faq_tariffe_cer.md
├── solar_production_raw.csv       # Generato dal notebook
├── household_consumption.csv      # Generato dal notebook
├── charts/                  # Generata dal notebook, contiene i grafici dall'agente
└── streamlit_screenshot.pdf    # Stampa della pagina con la conversazione di test

```

### Nota su `agent.ipynb` e `app.py`

Il file `app.py` è tenuto **separato** dal notebook invece di essere incorporato tramite
`%%writefile`, per evitare la duplicazione dei tool. Il notebook contiene comunque tutta la logica dell'agente
validata passo per passo nella Sezione 1; `app.py` porta quella stessa logica in
un'interfaccia Streamlit, come descritto nella Sezione 2 del notebook.

### Nota su `.env`

Il file delle chiavi API si chiama `.env`, secondo la convenzione standard. Contiene:
```
GROQ_API_KEY=...
```
Nota per chi lavora su Jupyter/Anaconda: il file browser di Jupyter nasconde di default i file
che iniziano con il punto, quindi `.env` non comparirà nella sidebar — non è un errore. Per
crearlo/rinominarlo, è più semplice farlo da una cella (`os.rename("file.env", ".env")`) invece
che dal file browser, che spesso non permette di digitare un nome che inizia con il punto.

## Setup

1. Installare le dipendenze:
   ```
   pip install -r requirements.txt
   ```
2. Creare un file `.env` nella cartella del progetto con la propria chiave Groq (gratuita
   su [console.groq.com](https://console.groq.com)):
   ```
   GROQ_API_KEY=la-tua-chiave
   ```
3. Alla prima esecuzione, il tool RAG scaricherà automaticamente due modelli da Hugging Face:
   il modello di embeddings `sentence-transformers/all-MiniLM-L6-v2` (~90MB) e il modello di
   reranking `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90MB) — serve connessione internet solo
   la prima volta, poi restano in cache locale e le esecuzioni successive non riscaricano nulla.

## Come eseguire

### Parte 1 (Notebook) — sviluppo e test dell'agente

Aprire `agent.ipynb` ed eseguire le celle in ordine. La prima
sezione genera i dataset, costruisce i tool, assembla l'agente e lo testa con alcune
domande di esempio, mostrando i passaggi di reasoning (tool chiamati, argomenti,
risultati intermedi).

### Parte 2 - Applicazione Streamlit 

Da terminale, nella cartella del progetto:
```
streamlit run app.py
```
Si aprirà una pagina su `http://localhost:8501`. Dalla sidebar, si devono i due file CSV
generati dal notebook (`solar_production_raw.csv` e `household_consumption.csv`), poi si può usare la chat per fare domande.

## Domande di esempio testate

### Nel notebook (Sezione 1 — validazione dell'agente)
- Test A — reasoning multi-tool in un singolo turno: "Confronta produzione e consumo delle ultime due settimane, segnalami eventuali anomalie e mostrami un grafico dell'andamento" (verifica che l'agente incateni correttamente compare_production_consumption, detect_anomalies e generate_chart)
- Test B — memoria conversazionale multi-turno (stesso thread_id, con checkpointer):
  1. "Qual è stata la produzione della scorsa settimana?"
  2. "E quella precedente?" (verifica che l'agente usi il contesto del turno precedente per interpretare correttamente "quella")


### Nell'app Streamlit (Sezione 2)

- "Mostrami un grafico dell'andamento settimanale di produzione e consumo"
- "In quale settimana la produzione è stata più alta?"
- "E in che giorno il deficit è stato più grande?"
- "Fammi vedere il grafico dell'andamento giornaliero, non settimanale"
- "Grazie, e la settimana precedente rispetto a quella che mi hai mostrato prima?" (follow-up, verifica della memoria conversazionale e della corretta generazione di un grafico per un periodo diverso da quello del turno precedente).
- "Conviene installare una batteria di accumulo?" (Per testare la base di conoscenza).
- "Posso guadagnare qualcosa vendendo l'elettricità che i miei vicini non usano?" (domanda che non usa nessuna delle parole chiave presenti nei documenti).
  
## RAG - base di conoscenza

Oltre ai tool sui dati numerici, l'agente dispone di un tool `search_knowledge_base` che
recupera informazioni da documenti testuali per rispondere a domande concettuali (es. "perché
la bolletta è alta anche con i pannelli solari?", "cos'è una comunità energetica?"), invece di
affidarsi solo alla conoscenza generale del modello.

Funzionamento: i file `.md` in `knowledge_base/` vengono divisi in chunk (chunk_size=1200,
chunk_overlap=150) e trasformati in embedding con il modello locale
`sentence-transformers/all-MiniLM-L6-v2`, indicizzati con **FAISS**. Il retrieval è a **tre
stadi**:
1. `MultiQueryRetriever` usa l'LLM (Groq) per generare **3 riformulazioni** della domanda
   dell'utente (sinonimi, termini tecnici diversi, altri modi di dire la stessa cosa) — risolve
   il problema del "vocabolario semantico": se l'utente scrive *"conviene una batteria?"* ma il
   documento parla di *"sistemi di accumulo"*, più riformulazioni aumentano le probabilità di
   recuperare il passaggio giusto. I risultati delle diverse riformulazioni vengono uniti
   eliminando i duplicati (anche un piccolo risparmio di token più avanti nella pipeline).
2. FAISS recupera, per ciascuna riformulazione, gli **8 chunk più simili** per similarità tra
   embedding (veloce, ma approssimativa).
3. Un **cross-encoder** locale (`cross-encoder/ms-marco-MiniLM-L-6-v2`, via
   `CrossEncoderReranker`) rilegge l'unione dei chunk recuperati insieme alla domanda
   **originale** (non alle riformulazioni) e li riordina per pertinenza logica effettiva. Solo i
   **5 migliori** (top_n=5) vengono passati all'agente.

In LangChain questa pipeline si costruisce componendo `MultiQueryRetriever` (che avvolge il
retriever FAISS) dentro un `ContextualCompressionRetriever` (che applica il reranking sul
risultato). Embeddings e cross-encoder girano in locale, nessuna nuova API key; le
riformulazioni della domanda usano invece l'LLM Groq già configurato per l'agente — quindi ogni
domanda concettuale genera una chiamata extra a Groq (gestita dal retry automatico già presente
per il rate limit del free tier).

**Nota sui documenti di esempio:** i due file in `knowledge_base/`
(`guida_risparmio_energetico.md` e `faq_tariffe_cer.md`) sono stati generati da Claude
(Anthropic) come contenuto dimostrativo, non essendo stati forniti documenti reali di EcoGrid
nel materiale del progetto. In un caso d'uso reale andrebbero sostituiti con la documentazione
ufficiale dell'azienda (guide, FAQ, condizioni contrattuali, ecc.).

## Tool implementati

- `load_and_clean_data` — pulizia dei CSV grezzi (interpolazione dei NaN, clipping dei
  valori negativi impossibili)
- `calculate_production` / `calculate_consumption` — aggregazione per ora/giorno/
  settimana/totale
- `compare_production_consumption` — confronto produzione/consumo con saldo
  surplus/deficit, con supporto a totali su più periodi (`last_n`)
- `detect_anomalies` — individuazione di giorni con consumo anomalo o surplus non
  sfruttato, escludendo i giorni parziali agli estremi del dataset
- `generate_chart` — generazione di grafici PNG (produzione, consumo o confronto),
  con supporto a settimane precedenti (`weeks_back`) per richieste di follow-up
- `search_knowledge_base` — RAG su documenti testuali (guide, FAQ) per rispondere a domande
  concettuali non coperte dai dati numerici (vedi sezione "RAG - base di conoscenza")

## Nota su caching e lettura dei dati

Nella prima versione, ogni tool rileggeva da zero i CSV puliti da disco (`pd.read_csv`) ad ogni
singola chiamata, anche più volte nello stesso turno di conversazione. Ora i dati puliti vengono
letti una sola volta e tenuti in cache: in `app.py` tramite `@st.cache_data` di Streamlit
(invalidata solo se l'utente carica CSV diversi dalla sidebar), nel notebook tramite due
variabili globali (`df_prod_clean`, `df_cons_clean`) popolate da `load_and_clean_data` e lette
dagli altri tool con `get_clean_data()`.

## Nota sul rate limit del free tier Groq

Il free tier di Groq per openai/gpt-oss-120b ha un limite di token al minuto (TPM). Eseguendo molte celle di test in rapida successione (es. "Restart & Run All" sull'intero notebook), è possibile incontrare un errore 429 rate_limit_exceeded. Non è un problema di codice: basta attendere pochi secondi e rieseguire la cella. In un utilizzo normale (domande scritte una alla volta, come nell'app Streamlit) il limite non viene raggiunto.

## Nota sulla riproducibilità dei dati

Il dataset è generato con `np.random.seed(42)` per rendere riproducibile il *pattern*
delle anomalie sintetiche (NaN, valori negativi). Il range temporale resta ancorato a
`datetime.today()` (come da script fornito), quindi la finestra di 30 giorni si
aggiorna ad ogni esecuzione — i valori numerici esatti riportati in questo README
(es. "191.86 kWh") sono quelli osservati durante lo sviluppo e potrebbero non
coincidere esattamente rieseguendo il progetto in un giorno diverso.