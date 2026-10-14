# Scheduling delle proiezioni FFN indipendenti

Questa prova cambia lo scheduling, mantenendo tutti gli shader originali.
Il gate numerico sul 27B passa: 66 vocabolari completi finiti, token e
stato canonico coincidono bit per bit. Il throughput non è ancora
misurato; la produzione rimane invariata e il vantaggio non è dimostrato.

Nel checkout llama locale la concorrenza è abilitata per default, salvo
`GGML_METAL_CONCURRENCY_DISABLE`, con gestione delle dipendenze tra
operazioni. Il decoding qw3 apre invece un encoder seriale. Il confronto
dei sorgenti individua un candidato preciso: quando gate e up non usano
già il matvec paired IQ3_S/IQ4_XS, entrambe le proiezioni leggono x1 e
scrivono regioni disgiunte di scratch, a offset 0 e n_ff. Non dipendono
l'una dal risultato dell'altra. Questo è un audit dei sorgenti, non una
misura del vantaggio di concorrenza in llama o del suo utilizzo effettivo
per ciascun nodo del grafo.

Il core privato aggiunge soltanto due hook attorno a quel ramo del
decoding. Chiude l'encoder seriale dopo la norm, apre una finestra
concorrente per i due matvec e torna a un encoder seriale prima di SwiGLU.
Mantiene lo stesso command buffer, tutti i kernel e relativi argomenti,
layout, pesi e buffer dati originali. Il generatore e la verifica
ricostruiscono l'intero qw3.c originale rimuovendo i due hook. Prefill e
matvec già paired restano originali. Non si introducono copie dei pesi,
buffer di sessione, shader nuovi o nuove pipeline. Aumentano gli oggetti
encoder; la RAM totale e di picco non sono misurate.

## Gate numerico

Build senza warning, Metal API Validation attiva, una sola sessione
ctx256 riutilizzata e oracolo temporaneo su disco. Prefisso sintetico 128
seed1, seguito greedy32, KV F16, 64 layer/head GPU. Baseline e candidato
confrontano l'intero vocabolario finito dopo il prefisso e ciascun passo,
token, history, posizione e stato canonico prima/dopo: KV GQA vivo,
DeltaNet e convoluzione; tail temporaneo Flash Attention escluso.

Baseline zero finestre; candidato 992 durante il seguito, 31×32.
Il contatore e il ripristino della modalità seriale sono verificati.
Sorgenti, snapshot, binario, stat del GGUF, binari di produzione e store
personale restano invariati durante il gate. La prova non esegue ancora
gli otto prefissi, checkpoint o regressioni dell'agente: restano necessari
prima di un'eventuale adozione confermata dai tempi.

Il [confronto di riferimento](m5_q4_uint_reference_2026-10-13.md) richiede
ancora +2,383% di throughput nel decoding, circa 2,55 ms risparmiati per
token. La concorrenza potrebbe anche aggiungere overhead: la misura
ordinaria OFF/ON/OFF/ON/OFF, tre ripetizioni e 60 secondi prima di ogni
processo, deve stabilirne l'effetto senza scartare campioni.

[Manifest numerico](measurements/decode_ffn_concurrent_2026-10-14/manifest.json),
[verifica](measurements/decode_ffn_concurrent_2026-10-14/verification.json) e
[audit dei sorgenti e comandi di build](measurements/decode_ffn_concurrent_2026-10-14/source_audit.json).
Non si archiviano binari, modelli o checkpoint.
