# Soddisfazione dei passeggeri aerei: EDA e classificazione

Progetto di data science che analizza i dati di un questionario post-volo per capire quali fattori determinano la soddisfazione dei passeggeri di una compagnia aerea, e per prevedere se un passeggero è **soddisfatto** oppure **neutrale/insoddisfatto** (classificazione binaria).

Il lavoro segue un percorso completo: analisi esplorativa, selezione delle feature, scelta motivata della metrica, confronto tra modelli con baseline di riferimento, ottimizzazione degli iperparametri e valutazione finale su un test set mai usato prima.

## Presentazione

La presentazione del progetto è disponibile su Canva: [Apri la presentazione](https://canva.link/lop6erevifwl455)

## Dataset

**Airline Passenger Satisfaction**, dataset pubblico disponibile su [Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction).

- 129.880 passeggeri, già divisi in `train.csv` (103.904 righe) e `test.csv` (25.976 righe)
- 22 feature:
  - profilo del passeggero e del viaggio: genere, età, tipo di cliente, tipo di viaggio, classe, distanza del volo
  - 14 valutazioni di servizio su scala 0-5 (prima del volo, comfort a bordo, personale)
  - ritardi alla partenza e all'arrivo, in minuti
- Target: `satisfaction` (`satisfied` = 1, `neutral or dissatisfied` = 0), con uno sbilanciamento lieve (56.7% / 43.3%)

I file CSV vanno posizionati nella stessa cartella del notebook.

## Contenuto del notebook

1. **Analisi esplorativa**: valori mancanti (imputati sfruttando la correlazione 0.965 tra ritardo in partenza e in arrivo), variabili categoriche, outlier (metodo IQR, nessuna riga rimossa), correlazioni e multicollinearità.
2. **Selezione delle feature**: confronto tra Chi-square, Mutual Information e T-test, con un approfondimento sull'effetto "a soglia" del ritardo.
3. **Standardizzazione**: StandardScaler e RobustScaler (per i ritardi), con fit solo sul train.
4. **Scelta della metrica**: accuracy come metrica principale, con balanced accuracy, F1-score, ROC-AUC e MCC come controllo; tre baseline di riferimento (classificatore banale, casuale e regola su una sola feature).
5. **Modellazione**: spot check di RandomForest, AdaBoost e Logistic Regression con k-fold cross validation stratificata, su due configurazioni di feature (tutte le 22, oppure le 4 più rilevanti nel T-test).
6. **Tuning**: RandomizedSearchCV con un protocollo fissato in anticipo (20 combinazioni per modello, stessa CV, seed fisso).
7. **Valutazione finale** sul test set e discussione dei risultati e dei limiti.

Il test set viene usato solo nella valutazione finale: tutte le decisioni di preprocessing, selezione e tuning si basano esclusivamente sul train.

## Risultati principali

| Strada | Modello | Accuracy CV (dopo tuning) | Accuracy test | Balanced accuracy test |
|---|---|---|---|---|
| 22 feature | **RandomForest** | 96.37% | **96.41%** | 0.962 |
| 22 feature | AdaBoost | 92.92% | 92.84% | 0.927 |
| 4 feature | RandomForest | 86.97% | 86.85% | 0.863 |
| 4 feature | AdaBoost | 85.72% | 85.66% | 0.854 |
| Baseline | Regola su Online boarding | 78.77%* | 78.34% | 0.788 |
| Baseline | Classificatore banale | 56.67%* | 56.10% | 0.500 |

\* Accuracy in CV: le baseline non hanno iperparametri da ottimizzare.

- Il modello migliore è un **RandomForest** (200 alberi, `max_depth=20`, `max_features=0.5`, `min_samples_leaf=4`) addestrato su tutte le feature: supera di circa 18 punti la regola basata sulla sola valutazione dell'online boarding.
- La classifica dei modelli è la stessa con tutte le metriche considerate.
- Le feature più influenti riguardano l'esperienza digitale e il comfort a bordo (online boarding, intrattenimento, classe di viaggio) più che la puntualità; il ritardo ha un effetto "a soglia" (in orario vs. in ritardo) più che proporzionale ai minuti.

## Limiti

- L'origine del dataset non è documentata (compagnia, periodo e modalità di raccolta non sono noti).
- Le valutazioni dei servizi e il giudizio complessivo provengono dallo stesso questionario: il modello è quindi più **esplicativo** (quali servizi pesano sulla soddisfazione) che **predittivo**, e questo contribuisce all'accuracy elevata.
- Il voto 0 nelle valutazioni potrebbe indicare un servizio non utilizzato ed è stato trattato come un normale valore numerico.
- La ricerca degli iperparametri è limitata a 20 combinazioni per modello.

## Come eseguire il notebook

Il notebook è stato sviluppato con Python 3.12. Le librerie necessarie, con le versioni usate, sono elencate in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Poi, con `train.csv` e `test.csv` nella stessa cartella:

```bash
jupyter notebook Eda_p2.ipynb
```

ed eseguire tutte le celle in ordine (*Restart & Run All*). L'esecuzione completa richiede circa 45 minuti, quasi tutti dedicati alla ricerca degli iperparametri. I `random_state` sono fissati, quindi i risultati sono riproducibili; versioni diverse di scikit-learn possono produrre piccole differenze nei decimali.

## Struttura della cartella

```
├── Eda_p2.ipynb             # notebook con analisi, modelli e commenti
├── train.csv                # dati di training
├── test.csv                 # dati di test
├── chi2_top_features.png    # grafico generato dal notebook
├── requirements.txt         # librerie e versioni usate
└── README.md
```
