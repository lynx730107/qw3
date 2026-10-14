# GQA: suddivisione degli head query indipendenti

La variante con tre head per threadgroup passa 432 fixture e il gate sul
27B, bit per bit. Quella con un head fallisce e viene esclusa. Lo screening
ordinario dà +2,622%, positivo in entrambi i blocchi; serve una replica
indipendente prima di un'eventuale adozione.

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

## Screening ordinario della variante2

Tutti i 15 campioni dei cinque processi sono conservati. Medie in ms:
3571,070 /3525,142 /3637,410 /3530,468 /3635,344. Primo ON +2,244% sugli
OFF interpolati, +1,303%/+3,185% sui singoli vicini. Secondo ON +3,000%,
+3,029%/+2,971% sui singoli vicini. L'aggregato è +2,622%; i controlli
sono condivisi, quindi non si calcola un intervallo paired indipendente
da questi due effetti. Spread max/min OFF 1,858%, entro il limite del 5%.
Il segnale è coerente nei due blocchi, ma la dispersione richiede replica.

Ogni ON conta 1552 selezioni, 97 forward×16, zero fallback; OFF zero.
Le righe CSV mantengono 176.761.680 byte di sessione. Swap 933,31 MiB
prima/dopo ciascun processo; sorgenti, snapshot, binari e stat del GGUF
invariati. Il manifest dei tempi eredita il sottoinsieme dei sorgenti
del modello selezionato. Il verificatore lo unisce all'intero envelope
di compilazione iniziale, verificando che il binario nativo sia identico
a quello già congelato e che tutti i sorgenti corrispondano agli snapshot.
Non si riscrivono i manifest o i dati grezzi originari.

[Analisi](measurements/gqa_groups_2026-10-14/selected_native/analysis.json)
e [verifica](measurements/gqa_groups_2026-10-14/selected_native/verification.json)
rileggono comandi, log, CSV, contatori e campioni. Non si confrontano i
tassi assoluti con quelli di llama ottenuti in una campagna diversa e
non si dichiara il raggiungimento dell'obiettivo da questo incremento.
La replica usa sei coppie AB/BA tra processi indipendenti,
stesso binario e workload, conservando ogni campione.

## Replica indipendente

La replica conclusa conserva 36 campioni di dodici processi. La media
OFF è 3599,788 ms, quella ON 3557,449 ms: **+1,190%**, inferiore al primo
screening. Tutte le sei coppie sono positive; il risparmio medio è
42,338 ms, intervallo paired t95% [10,819; 73,858] ms. L'unità statistica
è la media del processo, non ciascuna delle tre ripetizioni. Le coppie
sono trattate come indipendenti; correlazioni termiche/temporali possono
indebolire questa ipotesi. I due ordini risparmiano 42,004 e 42,673 ms.
Gli spread OFF/ON sono 1,271%/1,676%, entro il limite del 5%.

Il verificatore ricontrolla tutti i log, i 1552 dispatch per ON, i CSV,
33 dipendenze congelate, binari e stat del modello. Byte di sessione
176.761.680; swap 933,31 MiB invariato prima/dopo ciascun processo.
Non si misura la RAM totale o di picco. Nessun campione viene scartato.

[Manifest](measurements/gqa_groups_2026-10-14/replica_v2/manifest.json),
[analisi](measurements/gqa_groups_2026-10-14/replica_v2/analysis.json) e
[verifica](measurements/gqa_groups_2026-10-14/replica_v2/verification.json).
Il segnale supera il gate di prestazione privato; da solo non dimostra
di aver raggiunto llama. Non c'è ancora un confronto diretto aggiornato
della versione pubblica.

## Gate esteso della variante privata

Build senza warning e Metal API Validation attiva. Passano i prefissi
1/15/16/31/127/128/129/159, estensione 128→159, 32 token greedy per fase
e sedici roundtrip di checkpoint. Vocabolari completi finiti, token,
history, posizione e stato canonico coincidono bit per bit. Una sola
sessione ctx256 viene riutilizzata con oracoli temporanei su disco;
ogni seguito ON conta 512 dispatch effettivi e zero fallback.

[Manifest](measurements/gqa_groups_2026-10-14/quality_v2/manifest.json) e
[verifica](measurements/gqa_groups_2026-10-14/quality_v2/verification.json).
Questo gate riguarda ancora il candidato privato; integrazione pubblica,
regressioni dell'agente e confronto aggiornato con llama sono separati.

## Integrazione pubblica verificata

Il commit `a7dd19d` introduce il kernel pubblico, identico al candidato
privato a tre head salvo il nome. La selezione richiede il nome dispositivo
Apple M5, KV F16, 24 head query, quattro head KV e dimensione 256. Il
pipeline è opzionale: deve avere SIMD32, almeno 256 thread e lo stesso
scratch statico dell'originale. Altri dispositivi, forme o tipi KV e un
pipeline non disponibile mantengono il percorso originale. La modifica
agisce solo sull'attenzione decode non split; Flash e Q8 restano nei loro
percorsi. Si aggiunge un oggetto di pipeline, senza copie dei pesi o nuovi
buffer dati di sessione. RAM totale e di picco non sono misurate.

I test osservano il pipeline pubblico effettivo, senza sostituire kernel
o correggere le griglie mediante proxy. Passano le 432 matrici, comprese
le forme che devono mantenere il percorso originale: 18 selezioni del
nuovo pipeline. Passano inoltre otto prefissi, estensione 128→159,
continuazioni greedy32, vocabolari finiti completi, stato canonico,
history e sedici checkpoint, bit per bit. Si disabilita temporaneamente
solo il puntatore al pipeline opzionale per il controllo OFF, riusando
una sessione ctx256. Ogni continuazione ON conta 512 selezioni effettive.

Build senza warning e Metal API Validation attiva. Passano bash, edit,
fingerprint del runtime e smoke test dense dell'agente (generazione,
checkpoint salvato/ripreso, prefix cache creata/ripresa e controlli dei
parametri). Gli store sono temporanei. Non è l'intera suite per ogni
modello/backend.

Il primo tentativo di build si ferma per `linenoise.h` assente nel checkout
sparse, prima delle prove GPU. Log, manifest e controller di quel tentativo
restano in `public_integration/`. Dopo aver aggiunto i file mancanti al
checkout, il controller corretto usa una nuova directory senza riscrivere
i risultati precedenti.

[Manifest dell'integrazione](measurements/gqa_groups_2026-10-14/public_integration_v2/manifest.json)
e [verifica](measurements/gqa_groups_2026-10-14/public_integration_v2/verification.json)
congelano 28 sorgenti e i binari. La nuova campagna diretta con llama usa
questi binari pubblici, sei coppie AB/BA per workload, tre ripetizioni e
60 secondi prima di ogni processo, senza compilazioni o altri carichi GPU
durante le misure. Il risultato della replica privata non viene sommato
ad altri esperimenti né convertito in una vittoria su llama.
