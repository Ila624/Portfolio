import os
import time
from pathlib import Path
from typing import Literal

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# --- RAG: retrieval su documenti testuali (guide, FAQ) ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate

# =========================================================
# Configurazione pagina
# =========================================================
st.set_page_config(page_title="EcoBot Dashboard", page_icon="🌞", layout="centered")
st.title("🌞 EcoBot - Analisi Energetica")
st.caption("Chiedi informazioni sulla produzione fotovoltaica e sui consumi domestici.")

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# =========================================================
# Sidebar - Upload dei CSV
# =========================================================
st.sidebar.header("📁 Carica i dati")
uploaded_prod = st.sidebar.file_uploader("Produzione solare (CSV)", type="csv")
uploaded_cons = st.sidebar.file_uploader("Consumo domestico (CSV)", type="csv")

if uploaded_prod and uploaded_cons:
    # NB: usiamo variabili globali Python (sotto), non st.session_state, per
    # tenere i bytes caricati. LangGraph può eseguire i tool in thread separati
    # dal thread principale di Streamlit, e st.session_state non è accessibile
    # da quei thread (darebbe KeyError anche se il valore era stato impostato
    # correttamente qui sopra). Le variabili globali di modulo, invece, sono
    # condivise da tutti i thread dello stesso processo e non hanno questo problema.
    _prod_bytes = uploaded_prod.getvalue()
    _cons_bytes = uploaded_cons.getvalue()
    st.sidebar.success("File caricati correttamente.")

# =========================================================
# Caricamento dati con caching
# =========================================================
# Prima ogni tool rileggeva i CSV da disco a ogni chiamata (anche più volte
# nello stesso turno). Ora i dati puliti vengono calcolati una sola volta per
# ogni coppia di CSV caricati e tenuti in una cache in memoria (dizionario
# _clean_cache), evitando letture e parsing ripetuti e inutili. Usiamo una
# cache "manuale" con un dizionario invece di @st.cache_data proprio perché
# deve restare leggibile anche dai thread dei tool (vedi nota sopra).

_clean_cache: dict = {}


def _load_and_clean(prod_bytes: bytes, cons_bytes: bytes):
    """
    Pulisce i dati sporchi (valori NaN da sensori rotti, valori negativi
    impossibili) a partire dai bytes grezzi dei due CSV caricati, e restituisce
    i DataFrame puliti insieme a un report testuale delle correzioni applicate.
    """
    import io

    report = []

    df_prod = pd.read_csv(io.BytesIO(prod_bytes), parse_dates=["timestamp"])
    df_prod = df_prod.sort_values("timestamp").reset_index(drop=True)

    n_negativi = (df_prod["kwh_produced"] < 0).sum()
    df_prod["kwh_produced"] = df_prod["kwh_produced"].clip(lower=0)

    n_nan = df_prod["kwh_produced"].isna().sum()
    df_prod["kwh_produced"] = df_prod["kwh_produced"].interpolate(method="linear")
    df_prod["kwh_produced"] = df_prod["kwh_produced"].fillna(0)

    report.append(
        f"Produzione: corretti {n_negativi} valori negativi impossibili, "
        f"interpolati {n_nan} valori mancanti (sensore guasto)."
    )

    df_cons = pd.read_csv(io.BytesIO(cons_bytes), parse_dates=["timestamp"])
    df_cons = df_cons.sort_values("timestamp").reset_index(drop=True)

    n_negativi_cons = (df_cons["kwh_consumed"] < 0).sum()
    df_cons["kwh_consumed"] = df_cons["kwh_consumed"].clip(lower=0)

    n_nan_cons = df_cons["kwh_consumed"].isna().sum()
    df_cons["kwh_consumed"] = df_cons["kwh_consumed"].interpolate(method="linear").fillna(0)

    report.append(
        f"Consumo: corretti {n_negativi_cons} valori negativi, "
        f"interpolati {n_nan_cons} valori mancanti."
    )

    return df_prod, df_cons, " ".join(report)


def get_clean_data():
    """
    Ritorna (df_prod, df_cons) puliti, leggendoli dalla cache in memoria
    invece che da disco. Va chiamata da ogni tool al posto di pd.read_csv
    diretto sui file *_clean.csv. Se i CSV non sono ancora stati caricati,
    solleva un errore chiaro invece di un KeyError poco esplicativo.
    """
    if "_prod_bytes" not in globals() or "_cons_bytes" not in globals():
        raise RuntimeError(
            "Nessun CSV caricato: carica entrambi i file dalla sidebar prima di fare domande."
        )

    cache_key = (hash(_prod_bytes), hash(_cons_bytes))
    if cache_key not in _clean_cache:
        _clean_cache.clear()  # un solo set di CSV alla volta: svuotiamo la cache precedente
        _clean_cache[cache_key] = _load_and_clean(_prod_bytes, _cons_bytes)

    df_prod, df_cons, _ = _clean_cache[cache_key]
    return df_prod, df_cons


# =========================================================
# Tool (identici a quelli validati nel notebook, ma ora usano get_clean_data())
# =========================================================


@tool
def load_and_clean_data() -> str:
    """
    Pulisce i dati sporchi di produzione solare e consumo domestico (valori
    NaN dovuti a sensori rotti, valori negativi impossibili). I dati puliti
    vengono tenuti in cache e riusati da tutti gli altri tool, senza bisogno
    di rileggerli da disco a ogni chiamata.

    Usa questo tool come primo passo prima di qualsiasi analisi sui dati.

    Returns:
        Un riepilogo testuale delle anomalie trovate e corrette.
    """
    if "_prod_bytes" not in globals() or "_cons_bytes" not in globals():
        raise RuntimeError(
            "Nessun CSV caricato: carica entrambi i file dalla sidebar prima di fare domande."
        )
    _, _, report = _load_and_clean(_prod_bytes, _cons_bytes)
    return report


@tool
def calculate_production(period: Literal["hour", "day", "week", "total"] = "total") -> str:
    """
    Calcola l'energia prodotta dai pannelli solari dell'azienda, aggregata
    per il periodo richiesto. Usa i dati puliti (esegui load_and_clean_data
    prima se non l'hai già fatto).

    Args:
        period: livello di aggregazione.
            "total" = somma totale sull'intero dataset (ultimi 30 giorni)
            "day"   = somma per ogni giorno (mostra andamento giornaliero)
            "week"  = somma per settimana
            "hour"  = media di produzione per ogni ora del giorno (0-23)

    Returns:
        Un riepilogo testuale dei kWh prodotti secondo l'aggregazione scelta.
    """
    df, _ = get_clean_data()

    if period == "total":
        totale = df["kwh_produced"].sum()
        return f"Produzione totale negli ultimi 30 giorni: {totale:.2f} kWh."
    elif period == "day":
        daily = df.set_index("timestamp").resample("D")["kwh_produced"].sum()
        righe = [f"{d.strftime('%d/%m')}: {v:.2f} kWh" for d, v in daily.items()]
        return "Produzione giornaliera:\n" + "\n".join(righe)
    elif period == "week":
        weekly = df.set_index("timestamp").resample("W")["kwh_produced"].sum()
        righe = [f"Settimana del {d.strftime('%d/%m')}: {v:.2f} kWh" for d, v in weekly.items()]
        return "Produzione settimanale:\n" + "\n".join(righe)
    elif period == "hour":
        hourly = df.groupby(df["timestamp"].dt.hour)["kwh_produced"].mean()
        righe = [f"ore {h}:00 -> {v:.2f} kWh medi" for h, v in hourly.items()]
        return "Produzione media per ora del giorno:\n" + "\n".join(righe)


@tool
def calculate_consumption(period: Literal["hour", "day", "week", "total"] = "total") -> str:
    """
    Calcola l'energia consumata dalla famiglia, aggregata per il periodo richiesto.
    Usa i dati puliti (esegui load_and_clean_data prima se non l'hai già fatto).

    Args:
        period: livello di aggregazione ("total", "day", "week", "hour" -
            stesso significato di calculate_production).

    Returns:
        Un riepilogo testuale dei kWh consumati secondo l'aggregazione scelta.
    """
    _, df = get_clean_data()

    if period == "total":
        totale = df["kwh_consumed"].sum()
        return f"Consumo totale negli ultimi 30 giorni: {totale:.2f} kWh."
    elif period == "day":
        daily = df.set_index("timestamp").resample("D")["kwh_consumed"].sum()
        righe = [f"{d.strftime('%d/%m')}: {v:.2f} kWh" for d, v in daily.items()]
        return "Consumo giornaliero:\n" + "\n".join(righe)
    elif period == "week":
        weekly = df.set_index("timestamp").resample("W")["kwh_consumed"].sum()
        righe = [f"Settimana del {d.strftime('%d/%m')}: {v:.2f} kWh" for d, v in weekly.items()]
        return "Consumo settimanale:\n" + "\n".join(righe)
    elif period == "hour":
        hourly = df.groupby(df["timestamp"].dt.hour)["kwh_consumed"].mean()
        righe = [f"ore {h}:00 -> {v:.2f} kWh medi" for h, v in hourly.items()]
        return "Consumo medio per ora del giorno:\n" + "\n".join(righe)


@tool
def compare_production_consumption(period: Literal["day", "week"] = "day", last_n: int = None) -> str:
    """
    Confronta produzione ed energia consumata nello stesso periodo, evidenziando
    surplus (energia prodotta e non usata) o deficit (energia da acquistare
    dalla rete). Usa i dati puliti.

    Args:
        period: "day" per confronto giornaliero, "week" per confronto settimanale.
        last_n: se specificato, limita l'output agli ultimi N periodi (es. last_n=2
            con period="week" per le ultime 2 settimane) e aggiunge un totale
            aggregato già calcolato, così non serve sommare i valori a mano.

    Returns:
        Un riepilogo testuale con produzione, consumo e saldo per ogni periodo,
        più un totale aggregato se last_n è specificato.
    """
    df_prod, df_cons = get_clean_data()

    freq = "D" if period == "day" else "W"
    prod_agg = df_prod.set_index("timestamp").resample(freq)["kwh_produced"].sum()
    cons_agg = df_cons.set_index("timestamp").resample(freq)["kwh_consumed"].sum()

    df = pd.DataFrame({"prodotto": prod_agg, "consumato": cons_agg})
    df["saldo"] = df["prodotto"] - df["consumato"]

    if last_n:
        df = df.tail(last_n)

    label = "Giorno" if period == "day" else "Settimana del"
    righe = []
    for ts, row in df.iterrows():
        stato = "surplus" if row["saldo"] >= 0 else "deficit"
        righe.append(
            f"{label} {ts.strftime('%d/%m')}: prodotti {row['prodotto']:.2f} kWh, "
            f"consumati {row['consumato']:.2f} kWh, saldo {row['saldo']:+.2f} kWh ({stato})"
        )
    output = "\n".join(righe)

    if last_n and last_n > 1:
        tot_prod, tot_cons = df["prodotto"].sum(), df["consumato"].sum()
        tot_saldo = tot_prod - tot_cons
        stato = "surplus" if tot_saldo >= 0 else "deficit"
        output += (f"\n\nTotale sulle ultime {last_n} {period}e: prodotti {tot_prod:.2f} kWh, "
                   f"consumati {tot_cons:.2f} kWh, saldo {tot_saldo:+.2f} kWh ({stato})")

    return output


@tool
def detect_anomalies() -> str:
    """
    Individua situazioni particolari nei dati puliti, utili per generare consigli:
    giorni con surplus elevato di energia prodotta e non consumata (occasione
    sprecata) e giorni con consumo anomalmente alto rispetto alla media.
    I giorni parziali (inizio/fine dataset con meno di 24 ore registrate)
    vengono esclusi per evitare falsi positivi. Usa i dati puliti.

    Returns:
        Un riepilogo testuale delle situazioni degne di nota trovate.
    """
    df_prod, df_cons = get_clean_data()

    prod_daily = df_prod.set_index("timestamp").resample("D")["kwh_produced"].sum()
    cons_daily = df_cons.set_index("timestamp").resample("D")["kwh_consumed"].sum()

    ore_per_giorno = df_cons.set_index("timestamp").resample("D")["kwh_consumed"].count()
    giorni_completi = ore_per_giorno[ore_per_giorno == 24].index

    prod_daily = prod_daily.loc[giorni_completi]
    cons_daily = cons_daily.loc[giorni_completi]

    saldo = prod_daily - cons_daily
    soglia_surplus = saldo.mean() + saldo.std()
    media_cons = cons_daily.mean()
    soglia_consumo_alto = media_cons + cons_daily.std()

    note = []
    for ts in saldo.index:
        if saldo[ts] > 0 and saldo[ts] > soglia_surplus:
            note.append(
                f"{ts.strftime('%d/%m')}: surplus elevato di {saldo[ts]:.2f} kWh "
                f"non sfruttato (energia prodotta ma non consumata)."
            )
        if cons_daily[ts] > soglia_consumo_alto:
            note.append(
                f"{ts.strftime('%d/%m')}: consumo insolitamente alto "
                f"({cons_daily[ts]:.2f} kWh, media {media_cons:.2f} kWh)."
            )

    if not note:
        return "Nessuna anomalia particolare rilevata nei dati puliti."
    return "Situazioni rilevanti trovate:\n" + "\n".join(note)


@tool
def generate_chart(
    metric: Literal["production", "consumption", "comparison"] = "comparison",
    period: Literal["day", "week"] = "day",
    weeks_back: int = 0,
) -> str:
    """
    Genera un grafico e lo salva come file PNG nella cartella 'charts/'.
    Usa questo tool quando l'utente chiede esplicitamente di vedere un grafico,
    un andamento, una curva o un confronto visivo. Usa i dati puliti.

    Args:
        metric: cosa visualizzare ("production", "consumption", "comparison").
        period: "day" per andamento a 24h (media oraria), "week" per andamento
            settimanale (7 giorni, aggregazione giornaliera).
        weeks_back: solo se period="week". Quante settimane indietro rispetto
            alla più recente disponibile: 0 = ultima settimana, 1 = quella
            precedente, ecc. Usa questo per richieste tipo "la settimana prima"
            o "due settimane fa".

    Returns:
        Il percorso del file PNG generato, da usare per mostrare l'immagine.
    """
    os.makedirs("charts", exist_ok=True)

    df_prod, df_cons = get_clean_data()

    fig, ax = plt.subplots(figsize=(8, 4))

    if period == "day":
        prod_plot = df_prod.groupby(df_prod["timestamp"].dt.hour)["kwh_produced"].mean()
        cons_plot = df_cons.groupby(df_cons["timestamp"].dt.hour)["kwh_consumed"].mean()
        xlabel = "Ora del giorno"
    else:
        prod_daily = df_prod.set_index("timestamp").resample("D")["kwh_produced"].sum()
        cons_daily = df_cons.set_index("timestamp").resample("D")["kwh_consumed"].sum()

        end = len(prod_daily) - weeks_back * 7
        start = max(0, end - 7)
        prod_plot = prod_daily.iloc[start:end]
        cons_plot = cons_daily.iloc[start:end]
        prod_plot.index = prod_plot.index.strftime("%d/%m")
        cons_plot.index = cons_plot.index.strftime("%d/%m")
        xlabel = "Giorno"

    if metric in ("production", "comparison"):
        ax.plot(prod_plot.index, prod_plot.values, label="Produzione (kWh)", color="#f4a261", marker="o")
    if metric in ("consumption", "comparison"):
        ax.plot(cons_plot.index, cons_plot.values, label="Consumo (kWh)", color="#2a9d8f", marker="o")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("kWh")
    ax.set_title(f"{metric.capitalize()} - andamento per {period}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = f"charts/{metric}_{period}_back{weeks_back}.png"
    fig.savefig(filename)
    plt.close(fig)

    return filename


# =========================================================
# RAG - Base di conoscenza (guide, FAQ) oltre ai dati numerici
# =========================================================
# I tool sopra rispondono a domande sui NUMERI (quanto ho prodotto/consumato).
# Questo tool copre invece le domande "concettuali" (perché la bolletta è
# alta, cos'è una CER, quando conviene una batteria...) grazie a un indice
# vettoriale costruito sui documenti presenti in knowledge_base/.
#
# Gli embeddings sono calcolati in locale con sentence-transformers (nessuna
# API key aggiuntiva richiesta), il vector store è FAISS (in memoria).
#
# Pipeline di retrieval a tre stadi (revisione post-consegna):
# 1) MultiQueryRetriever usa l'LLM (Groq) per generare 3 riformulazioni della
#    domanda dell'utente (sinonimi, termini tecnici diversi, altri modi di
#    dire la stessa cosa) e recupera i chunk più simili per ciascuna, poi
#    unisce i risultati eliminando i duplicati. Risolve il problema del
#    "vocabolario semantico": se l'utente scrive "conviene una batteria?" ma
#    il documento parla di "sistemi di accumulo", la sola similarità tra
#    embedding potrebbe non far combaciare bene i due termini; con più
#    riformulazioni della domanda aumentano le probabilità di intercettare
#    il passaggio giusto. Riduce anche i token inviati più avanti nella
#    pipeline: i duplicati (stesso chunk recuperato da più riformulazioni)
#    vengono contati una sola volta.
# 2) FAISS recupera, per ciascuna riformulazione, gli 8 chunk più simili per
#    similarità tra embedding (veloce, ma approssimativa).
# 3) Un cross-encoder (CrossEncoderReranker) rilegge l'unione dei chunk
#    recuperati al passo 1-2 insieme alla domanda ORIGINALE (non alle
#    riformulazioni) e li riordina in base alla pertinenza logica effettiva.
#    Solo i 5 migliori (top_n) vengono passati all'agente.
# Imbeddings e cross-encoder girano in locale, senza nuove API key. Le
# riformulazioni della domanda usano invece l'LLM Groq già configurato per
# l'agente: ogni domanda concettuale genera quindi una chiamata extra a Groq
# (oltre a quella per la risposta finale) — da tenere presente sul fronte
# del rate limit del free tier (già gestito da invoke_agent_with_retry).

KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"

# Prompt italiano per la generazione delle riformulazioni: il prompt di
# default di MultiQueryRetriever è in inglese, ma sia le domande degli
# utenti sia i documenti della knowledge base sono in italiano.
MULTI_QUERY_PROMPT_IT = PromptTemplate(
    input_variables=["question"],
    template="""Sei un assistente che aiuta a migliorare la ricerca in una base di conoscenza
sull'energia domestica e il fotovoltaico. Genera 3 riformulazioni diverse della domanda
dell'utente qui sotto, usando sinonimi, termini tecnici alternativi o modi di dire diversi
per esprimere lo stesso concetto, così da recuperare documenti pertinenti anche se la domanda
originale usa un vocabolario diverso da quello dei documenti. Scrivi le 3 riformulazioni in
italiano, una per riga, senza numerarle e senza aggiungere altro testo.

Domanda originale: {question}""",
)


@st.cache_resource(show_spinner="Indicizzazione della base di conoscenza...")
def build_knowledge_base(kb_dir: str):
    """
    Carica tutti i file .md/.txt in kb_dir, li divide in chunk e costruisce
    un retriever a tre stadi: MultiQueryRetriever (riformulazioni della
    domanda via LLM) + FAISS (recupero per similarità) + cross-encoder
    reranker (riordino e selezione dei 5 passaggi più pertinenti). Viene
    eseguito una sola volta per sessione (cache_resource), non ad ogni
    domanda dell'utente.
    """
    loader = DirectoryLoader(kb_dir, glob="**/*.md", loader_cls=TextLoader,
                              loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()

    # chunk_size aumentato (da 800 a 1200): con chunk troppo piccoli, concetti
    # articolati su più frasi (es. il ragionamento completo sul perché una
    # batteria conviene o meno) rischiano di essere spezzati a metà, perdendo
    # il contesto necessario al cross-encoder per valutarli correttamente.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Stadio 1: recupero per similarità (k=8 per ciascuna riformulazione;
    # con 3 riformulazioni + domanda originale il totale prima dei duplicati
    # può arrivare fino a 4x8=32 candidati, poi ridotti dalla deduplica).
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    # Stadio 2: MultiQueryRetriever genera le riformulazioni con l'LLM Groq
    # e unisce i risultati (deduplicati) delle ricerche su tutte le varianti.
    query_llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=query_llm,
        prompt=MULTI_QUERY_PROMPT_IT,
        include_original=True,
    )

    # Stadio 3: reranking con cross-encoder locale sull'unione dei candidati,
    # tiene solo i 5 migliori rispetto alla domanda originale dell'utente.
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=multi_query_retriever)


@tool
def search_knowledge_base(query: str) -> str:
    """
    Cerca nella base di conoscenza di EcoGrid (guide sul risparmio energetico,
    FAQ su tariffe e comunità energetiche rinnovabili) informazioni utili a
    rispondere a domande concettuali dell'utente, quando non si tratta di una
    domanda sui suoi dati numerici (produzione/consumo/saldo).

    Esempi di domande adatte a questo tool: "Perché la bolletta è alta anche
    se ho i pannelli solari?", "Cos'è una comunità energetica rinnovabile?",
    "Conviene installare una batteria?".

    Args:
        query: la domanda o l'argomento da cercare nella base di conoscenza.

    Returns:
        I 5 passaggi più rilevanti (dopo reranking) trovati nei documenti,
        da usare come base per la risposta.
    """
    retriever = build_knowledge_base(str(KNOWLEDGE_BASE_DIR))
    risultati = retriever.invoke(query)

    if not risultati:
        return "Nessuna informazione pertinente trovata nella base di conoscenza."

    passaggi = [f"[Fonte: {Path(r.metadata.get('source', '?')).name}]\n{r.page_content}" for r in risultati]
    return "\n\n---\n\n".join(passaggi)


TOOLS = [
    load_and_clean_data,
    calculate_production,
    calculate_consumption,
    compare_production_consumption,
    detect_anomalies,
    generate_chart,
    search_knowledge_base,
]

SYSTEM_PROMPT = """Sei EcoBot, un assistente virtuale specializzato in analisi energetica residenziale.
Aiuti l'utente a capire i dati di produzione fotovoltaica e di consumo domestico della sua abitazione.

Il tuo tono è professionale ma cordiale: chiaro, preciso, senza gergo tecnico non necessario.
Ti rivolgi all'utente con "tu", in modo diretto ma sempre rispettoso.

Regole importanti:
- Usa SEMPRE i tool a disposizione per rispondere a domande su dati, numeri o statistiche.
  Non inventare mai valori: se non hai un tool adatto per rispondere con certezza, dillo esplicitamente.
- Se i dati non sono ancora stati puliti (prima interazione), usa load_and_clean_data come primo passo.
- Se l'utente chiede un grafico, un andamento, una curva o un confronto visivo, usa generate_chart
  e menziona nella risposta che il grafico è disponibile.
- Se l'utente fa una domanda di follow-up che si riferisce a un grafico mostrato in un turno
  precedente (es. "e la settimana prima?", "quella prima ancora?", "confrontala con l'altra"),
  continua a generare il grafico corrispondente con generate_chart, anche se il follow-up non
  ripete esplicitamente parole come "grafico" o "andamento".
- IMPORTANTE: non scrivere mai frasi come "ecco il grafico" o "il grafico è disponibile" se non hai
  effettivamente chiamato generate_chart in questo stesso turno. Se non hai generato un grafico,
  non parlarne come se esistesse.
- Non citare mai all'utente i nomi tecnici dei tool (es. "detect_anomalies", "search_knowledge_base",
  "generate_chart"): sono strumenti interni che tu usi per rispondere, non qualcosa che l'utente può
  eseguire. Se vuoi proporre un approfondimento, descrivilo in linguaggio naturale
  (es. "posso controllare se ci sono giorni con surplus significativo" invece di
  "posso eseguire detect_anomalies").
- Se descrivi i colori di un grafico generato con generate_chart, usa sempre e solo questi (sono fissi
  nel codice, non indovinarli): produzione = arancione, consumo = verde acqua (teal). Non usare mai
  altri colori (es. blu, rosso) per descriverli.
- Quando riporti numeri, arrotonda a 2 decimali e specifica sempre l'unità (kWh).
- Se noti surplus o deficit energetici rilevanti, offri un breve consiglio pratico
  (es. spostare l'uso di elettrodomestici nelle ore di massima produzione).
- Per domande concettuali (es. "perché la bolletta è alta anche con i pannelli solari?",
  "cos'è una comunità energetica?", "conviene una batteria?") usa il tool
  search_knowledge_base invece di rispondere solo dalla tua conoscenza generale.
- Rispondi sempre in italiano, in modo sintetico ma completo.
"""

# =========================================================
# Gestione errori: retry automatico su rate limit (429)
# =========================================================
# Il free tier di Groq ha un limite di token al minuto (TPM). Quando viene
# superato, l'API risponde con un errore 429 che indica anche dopo quanti
# secondi/millisecondi riprovare (di solito meno di un minuto). Invece di far
# fallire la conversazione con uno stack trace per l'utente, ritentiamo
# automaticamente con un breve backoff esponenziale.


def invoke_agent_with_retry(agent, user_message: str, config: dict, max_retries: int = 3, base_delay_seconds: float = 3):
    """
    Chiama l'agente gestendo automaticamente gli errori di rate limit (429).

    Se la chiamata fallisce per un motivo diverso dal rate limit, l'eccezione
    viene rilanciata subito, senza ritentare inutilmente.

    Args:
        agent: l'agente LangGraph da invocare.
        user_message: il messaggio dell'utente per questo turno.
        config: la configurazione (thread_id) da passare ad agent.invoke.
        max_retries: numero massimo di tentativi aggiuntivi dopo il primo.
        base_delay_seconds: attesa base per il backoff esponenziale (raddoppia
            ad ogni tentativo: 3s, 6s, 12s, ...).

    Returns:
        Il risultato di agent.invoke, se uno dei tentativi va a buon fine.

    Raises:
        L'ultima eccezione incontrata, se anche l'ultimo tentativo fallisce.
    """
    for tentativo in range(max_retries + 1):
        try:
            return agent.invoke({"messages": [{"role": "user", "content": user_message}]}, config=config)
        except Exception as e:
            is_rate_limit = "rate_limit" in str(e).lower() or "429" in str(e)
            if not is_rate_limit or tentativo == max_retries:
                raise
            attesa = base_delay_seconds * (2 ** tentativo)
            st.warning(
                f"⏳ Limite di richieste raggiunto, riprovo tra {attesa:.0f} secondi "
                f"(tentativo {tentativo + 1}/{max_retries})..."
            )
            time.sleep(attesa)


# =========================================================
# Logica dell'agente (attiva solo dopo l'upload dei CSV)
# =========================================================
if uploaded_prod and uploaded_cons:

    # L'agente e la memoria vengono creati una sola volta per sessione
    if "agent" not in st.session_state:
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.3,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        checkpointer = MemorySaver()
        st.session_state.agent = create_agent(
            model=llm,
            tools=TOOLS,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )
        # thread_id fisso per sessione utente: è la "chiave" che LangGraph usa
        # per recuperare lo storico conversazionale tra un invoke e l'altro
        st.session_state.thread_id = "sessione-1"
        st.session_state.chat_display = []  # solo per mostrare i messaggi in UI
        st.session_state.processed_count = 0  # quanti messaggi del thread sono già stati processati

    agent = st.session_state.agent

    # --- Render della cronologia già presente ---
    for msg in st.session_state.chat_display:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("chart"):
                st.image(msg["chart"])

    # --- Input utente ---
    user_input = st.chat_input("Fai una domanda su produzione e consumi...")

    if user_input:
        st.session_state.chat_display.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        try:
            result = invoke_agent_with_retry(agent, user_input, config)
        except Exception:
            # Anche dopo i tentativi di retry non è stato possibile ottenere
            # una risposta (rate limit persistente o altro errore imprevisto):
            # mostriamo un messaggio chiaro invece di far crashare l'app.
            errore_msg = (
                "⚠️ Non sono riuscito a rispondere a causa di un problema tecnico "
                "(probabilmente il limite di richieste è ancora saturo). "
                "Aspetta qualche secondo e riprova."
            )
            with st.chat_message("assistant"):
                st.error(errore_msg)
            st.session_state.chat_display.append({"role": "assistant", "content": errore_msg, "chart": None})
            st.stop()

        risposta = result["messages"][-1].content

        # Il checkpointer accumula TUTTA la cronologia in result["messages"], quindi
        # cerchiamo il grafico solo tra i messaggi nuovi generati in QUESTO turno,
        # altrimenti riemergerebbe un grafico di un turno precedente non pertinente.
        nuovi_messaggi = result["messages"][st.session_state.processed_count:]
        st.session_state.processed_count = len(result["messages"])

        chart_path = None
        for m in nuovi_messaggi:
            if type(m).__name__ == "ToolMessage" and m.name == "generate_chart":
                chart_path = m.content

        with st.chat_message("assistant"):
            st.markdown(risposta)
            if chart_path:
                st.image(chart_path)

        st.session_state.chat_display.append(
            {"role": "assistant", "content": risposta, "chart": chart_path}
        )

else:
    st.info("⬅️ Carica entrambi i file CSV dalla sidebar per iniziare la chat con EcoBot.")