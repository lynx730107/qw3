# GQA: suddivisione degli head query indipendenti

La variante con tre head per threadgroup passa 432 fixture e il gate sul
27B, bit per bit. Quella con un head fallisce e viene esclusa. La velocità
della variante selezionata non è ancora misurata; nessuna adozione.

Nel 27B ci sono 24 head query, quattro head KV e dimensione 256. Il
kernel originale assegna tutti i sei head query associati a un head KV
allo stesso threadgroup: quattro gruppi totali per l'attenzione. Il
candidato li divide in due gruppi di tre, otto gruppi totali. I tre head
di ogni gruppo conservano l'ordine dei token di contesto, delle operazioni
FP32 e delle riduzioni SIMD a 32 lane. Si modifica soltanto la mappatura
dei gruppi/head, la capacità degli array privati e il limite dei cicli
sugli head. Restano tutte e tre le barriere originali per token di contesto.

Il kernel ricava head KV e partizione dal gruppo; il setter privato
moltiplica coerentemente la griglia. I gruppi aggiuntivi oltre il limite
ritornano senza leggere o scrivere dati. Non si creano copie dei pesi o
buffer dati di sessione. Restano zero byte statici di threadgroup e 256
byte dinamici per gruppo, SIMD32 e fino a 256 thread nel modello. La
maggiore griglia cambia la distribuzione del lavoro e i rileggimenti KV;
non dimostra da sola un guadagno. Oggetti di pipeline e temporanei di
compilazione sono presenti; la RAM totale e di picco non sono misurate.

## Numerica e fallimento conservato

Build senza warning, Metal API Validation attiva. Il candidato a un head
passa i primi sei casi, poi differisce a gruppo1/dim32/ctx2/KV F32/input
misto: `38da8c97` contro `38da8cb0`, 25 ULP. Nessun test modello o timing
viene eseguito per quella variante. L'iniziale gate comune si ferma,
conservando entrambi i risultati delle matrici.

Il candidato a tre head passa tutti i 432 casi: gruppi query/KV 1/6/7/8,
dimensioni 32/128/256, contesti 1/2/31/128/129/159, KV F32/F16 e tre
pattern (misto, zeri con segno e scale variate). Output finiti identici
byte per byte, guardie, input immutati e gruppi extra oltre il limite;
432 selezioni, zero fallback. Si usano piccoli buffer temporanei
riutilizzati, senza caricare i pesi del modello.

Un runner dedicato confronta poi solo baseline e variante2 sul GGUF,
con una sola sessione ctx256 riutilizzata e oracolo temporaneo su disco:
prefisso128 seed1, seguito greedy32, KV F16 e 64 layer/head GPU. Coincidono
66 vocabolari completi finiti, token, history, posizione e stato canonico
prima/dopo: KV GQA vivo, DeltaNet e convoluzione. Il tail temporaneo Flash
Attention è escluso. La variante seleziona 512 dispatch, zero fallback.
Otto prefissi, checkpoint e regressioni dell'agente restano da eseguire.

Il primo controller dedicato aveva mantenuto scope e messaggio esterno
della prova DeltaNet copiata. I marker interni GQA e la selezione della
variante2 erano corretti. Si conserva integralmente quel tentativo e si
riesegue lo stesso binario, senza modifiche ai kernel, in una directory
nuova con scope e messaggio corretti. Anche la seconda esecuzione passa;
la correzione e gli hash sono in `selected_scope_correction.json`.

[Gate iniziale](measurements/gqa_groups_2026-10-14/manifest.json),
[gate corretto della sola variante2](measurements/gqa_groups_2026-10-14/selected_v2_corrected/manifest.json),
[verifica](measurements/gqa_groups_2026-10-14/verification.json) e
[metadati del modello](measurements/gqa_groups_2026-10-14/model_shape.json).
Sorgenti, snapshot, binari, stat del GGUF e store personale restano
invariati nelle rispettive prove. Non si archiviano binari, GGUF o checkpoint.

Lo screening selezionato usa OFF/variante2/OFF/variante2/OFF, tg32 depth128
ctx256, tre ripetizioni, warmup e 60 secondi prima di ogni processo,
API Validation disattiva per i tempi. Tutti i campioni saranno conservati.
Non c'è ancora un nuovo confronto con llama.
