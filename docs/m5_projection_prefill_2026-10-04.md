# Proiezioni quantizzate e margine di generazione — MacBook Air M5

Campagna del 4 ottobre 2026, successiva alla [prova GatedDeltaNet](m5_deltanet_prefill_2026-10-04.md). Base del motore: archivio `4f033f6`; branch di lavoro `codex/m5-deltanet-prefill`. Nessuna ripresa dello sviluppo della delega.

## Metodo

Modello locale `Qwen3.8-27B-UD-Q3_K_XL.gguf`, 13.146.393.504 byte, 64 layer interamente su Metal, KV F16, contesto allocato 512, 4 thread ausiliari, seed 1. pp128 puro e tg32 a profondità 128 sono processi distinti; tg32 comprende **32 forward ordinari con logits**, senza campionamento. Warmup e preparazione della profondità sono esclusi. Controlli OFF alternati alla variante, tre ripetizioni per blocco, 40 secondi di pausa, una sola inferenza GPU attiva.

Criterio stabilito prima delle misure: almeno +5% pp128 ripetibile, nessuna regressione tg32 oltre il 2%, dispersione dei controlli OFF entro il 5%. I microbenchmark non sostituiscono questo criterio.

## Misurare senza spezzare il batching

Il dispositivo espone il set `timestamp`, supporta `AtStageBoundary` e non supporta `AtDispatchBoundary` o `AtBlitBoundary`. `xctrace` non è disponibile nell'installazione locale dei Command Line Tools. Il prototipo diagnostico inserisce gli attachment dei timestamp negli **encoder che il motore apre già**, conservando il loro dispatch type. Non aggiunge encoder, barriere, commit o attese. La risoluzione dei contatori avviene dopo il timer e dopo le normali attese del motore. Ogni buffer accetta al massimo 4096 timestamp su questa macchina: il diagnostico usa pagine separate preallocate.

Questo segue il meccanismo descritto da [Apple per i contatori Metal](https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers). Si registrano anche nomi e numero dei dispatch e gli intervalli GPU dei command buffer. Il vecchio profiler che sincronizza le singole fasi non viene usato per calcolare percentuali.

Calibrazione del prototipo v1: OFF / profilo / OFF / profilo / OFF, tre ripetizioni, stessi input. pp128: 151,54 → 148,25 token/s, **−2,17%**. tg32: 8,650 → 8,494 token/s, **−1,81%**. Dispersione dei controlli: 2,28% e 0,90%. Quindi il profilo ha un costo: tutte le decisioni di velocità si prendono con contatori disattivati. [Dati e snapshot del prototipo misurato](measurements/projection_profile_2026-10-04/results.json).

In pp128 si osservano 1286 passaggi compute e 68 command buffer per ripetizione. Molti passaggi contengono un solo dispatch; i passaggi misti restano non attribuiti ai singoli kernel. La media dei passaggi singoli dà:

| Kernel | Durata GPU cumulata, ms |
|---|---:|
| IQ4_XS staged128 | 244,81 |
| IQ3_S staged128 | 214,53 |
| Q4_K / Q5_K staged128 | 68,42 |
| IQ3_XXS staged128 | 66,25 |
| Q8_0 batch4 | 47,98 |
| IQ2_S staged128 | 35,35 |

Le durate sono intervalli dei passaggi campionati, **non percentuali esatte della latenza senza profiling**. Sommare intervalli di command buffer/encoder può includere sovrapposizioni e costo dei contatori. Q3_K staged128 ha solo 11 dispatch, contro 156 IQ4_XS e 111 IQ3_S: il nome del GGUF non descrive tutte le quantizzazioni effettive.

In tg32 si osservano 544 passaggi compute e 64 command buffer: i passaggi sono misti, perciò non si assegna un tempo a ciascun matvec. Media: timer CPU 3767,43 ms, intervalli GPU dei command buffer cumulati 3738,07 ms. Questa vicinanza suggerisce che ridurre il solo lavoro CPU non sia il primo intervento, ma non prova che il carico GPU sia interamente limitato dalla memoria. [Dettaglio del profilo](measurements/projection_profile_2026-10-04/stage_summary.json).

## Inventario dei pesi realmente legati al modello

Si contano i tensori usati dai 64 layer del motore, il layer finale e una sola riga dell'embedding per token. Si escludono i tensori del blocco next-N non eseguito. I byte sono calcolati dai blocchi GGUF, senza usare gli spazi di allineamento del file.

- Pesi legati al modello, conteggio degli offset univoci: **12.784.388.096 byte**.
- Pesi attivi per token nel modello denso, assumendo una lettura ciascuno: **12.238.086.296 byte**.
- Proiezioni IQ4_XS: 4.971.724.800 byte; IQ3_S: 3.527.884.800 byte; Q3_K: 322.150.400 byte.

Il conteggio dei pesi legati non è una misura della RAM fisicamente residente. Il conteggio per token non è un contatore DRAM: cache, riletture, stato ricorrente, KV, attivazioni e output introducono differenze. I piccoli vettori possono restare in cache. L'inventario non si generalizza al MoE né a un contesto diverso. [Inventario completo](measurements/projection_cost_2026-10-04/inventory.json).

## Esperimenti sui metadati

Le varianti private annidano otto passi K=32 dentro un blocco K=256. Conservano la geometria 64 colonne × 128 righe, 128 thread, quattro simdgroup, staging dei pesi half e accumulo GEMM nell'ordine originale. Non cambiano GGUF, precisione degli input o dei KV.

- Q3_K: ricostruzione dei quattro word delle scale e lettura del moltiplicatore una volta per blocco.
- IQ3_S / IQ4_XS: moltiplicatore e scale impacchettate caricati una volta per blocco; dequantizzazione e prodotti restano nel passo K=32.

I diagnostici compilano tutte le varianti anche per il controllo OFF. L'attivazione e i contatori privati verificano che il percorso richiesto sia effettivamente eseguito. Il motore ordinario non usa questi shader.

I controlli numerici confrontano l'intero vocabolario con uguaglianza esatta e finitezza, greedy per 32 token, KV canonici, DeltaNet e conv, più salvataggio/ripristino del payload. Prefissi 16, 31, 128 con estensione a 159, 1 e 15. La variante con i contatori viene confrontata separatamente con il percorso senza contatori. [Manifest di qualità](measurements/projection_quality_2026-10-04/manifest.json).

## Risultati di velocità e banda

Confronto completo IQ3_S/IQ4_XS, OFF / variante / OFF / variante / OFF, tre ripetizioni per blocco:

| Misura | OFF | Variante | Variazione |
|---|---:|---:|---:|
| pp128, token/s | 151,764 | 148,749 | −1,986% |
| tg32, token/s | 8,733 | 8,737 | +0,046% |

Controlli stabili: dispersione pp128 1,58%, tg32 0,75%. Le due comparazioni con i controlli immediatamente vicini danno −2,42% e −1,70% pp128. La variante viene scartata. [Risultati completi](measurements/projection_iq_metadata_2026-10-04/results.json).

La prova isolata ha riprodotto tutte le 41 combinazioni effettive di formato e dimensioni delle proiezioni maggiori; otto replay per command buffer, un warmup e cinque campioni, confronto OFF / variante / OFF / variante / OFF per le forme interessate. I risultati dei buffer coincidono esattamente e sono finiti. Pesando i replay per il numero effettivo di matrici della stessa forma: Q3_K −9,22%, IQ3_S +1,99%, IQ4_XS −8,05%. Questi sono risultati diagnostici con input sintetici, non guadagni del modello né tempi additivi delle fasi. Non si dimostra una causa di occupazione/registri: non disponiamo di quei contatori. [Matrici e campioni](measurements/projection_cost_2026-10-04/matrices.json), [riepilogo](measurements/projection_cost_2026-10-04/matrix_summary.json).

Il confronto Q3_K è OFF / variante / OFF, tre ripetizioni per blocco, **un solo blocco candidato**:

| Misura | OFF | Variante | Variazione osservata |
|---|---:|---:|---:|
| pp128, token/s | 152,387 | 154,681 | +1,505% |
| tg32, token/s | 8,663 | 8,893 | +2,661% |

Dispersione dei controlli: pp128 0,56%, tg32 2,72%. Il punto pp128 resta sotto il criterio del 5% e non dimostra un miglioramento ripetibile. Non si attribuisce il risultato tg32 alla variante: il kernel staged128 cambiato lavora nella preparazione della profondità, fuori dal timer, mentre i 32 matvec cronometrati usano il percorso originale. Il disaccordo con i replay isolati è un ulteriore motivo per non promuovere la variante. [Risultati Q3_K](measurements/projection_q3_metadata_2026-10-04/results.json).

### Banda e margine della generazione

Il microbenchmark legge un buffer GPU senza modificarlo, con vettori uint4 e risultati controllati da un oracolo intero CPU su più posizioni. Otto letture ordinate per command buffer, tre warmup e sei campioni cronometrati; il tempo riportato è per lettura e include le barriere del replay. Tre processi separati, con 40 secondi di pausa prima delle due ripetizioni. Nessuna inferenza contemporanea.

| Buffer | GB/s nei tre processi | Dispersione tra processi | Tutti i campioni interni entro 5% |
|---|---|---:|---|
| 128 MiB | 127,27 / 127,43 / 127,57 | 0,24% | No |
| 256 MiB | 127,77 / 129,57 / 129,65 | 1,47% | Sì |
| 512 MiB | 128,74 / 129,85 / 129,18 | 0,86% | Sì |
| 1024 MiB | 113,78 / 128,10 / 126,61 | 12,59% | No |

La prima prova con una sola lettura per comando era troppo variabile; resta registrata in `bandwidth.json`, ma non fonda la stima. Anche il caso 1 GiB delle prove lunghe resta instabile e non viene usato come limite preciso. [Campioni e riepilogo](measurements/projection_cost_2026-10-04/bandwidth_summary.json), [manifest e swap](measurements/projection_cost_2026-10-04/bandwidth_long_manifest.json).

Con tg32 OFF 8,733 token/s, il prodotto `12.238.086.296 byte × token/s` vale **106,87 GB/s di banda equivalente dei pesi**, non banda DRAM osservata. La banda del replay stabile da 256–512 MiB, circa 129 GB/s, divisa per i soli pesi suggerisce un **tetto ideale di circa 10,5–10,6 token/s**, circa il 21% sopra quel controllo. Non è una previsione del motore: trascura dequantizzazione, calcolo, riletture, KV, stato ricorrente, attivazioni, scheduling e comportamento diverso delle cache. Non basta per affermare che quel guadagno sia raggiungibile.

[Apple dichiara 153 GB/s per il MacBook Air M5](https://support.apple.com/en-us/126320): con la stessa assunzione dei soli pesi, il tetto nominale sarebbe 12,50 token/s. Usarlo come prestazione realistica sarebbe ingiustificato. Le misure **non dimostrano assenza di margine nella generazione**, e nemmeno dimostrano un'accelerazione del 21% disponibile con una specifica modifica.

Lo swap era già presente: circa 2,0 GiB usati. I manifest conservano prima/dopo ogni lavoro; nel blocco Q3_K si osserva anche una diminuzione di 16 MiB tra processi. Non si afferma che la macchina fosse priva di swap, né che il file GGUF fosse tutto fisicamente residente nello stesso momento.

## Esito

Nessuna variante raggiunge il criterio di adozione. `qw3.c`, `qw3_metal.m`, `qw3_bench.c` e tutti gli shader di produzione sono byte per byte uguali a `4f033f6`; [verifica registrata](measurements/projection_quality_2026-10-04/production_identity.json). I tre controlli numerici finali passano; i tre eseguibili diagnostici si costruiscono senza warning. Le regressioni di agente/sessioni della campagna precedente restano valide per il percorso ordinario, che questa campagna non ha modificato; non sono state rilanciate senza motivo.

Il risultato utile è l'attribuzione misurata delle proiezioni predominanti e una stima esplicita del vincolo di banda. Riusare questi metadati non produce un vantaggio dimostrato. Una futura prova sul prefill dovrebbe intervenire sul costo dominante di IQ4_XS/IQ3_S con un'idea diversa; una prova sulla generazione dovrebbe misurare un miglioramento dei kernel di decode completi, senza trasformare il tetto di banda in una promessa. Nessun nuovo confronto con llama.cpp è stato eseguito in questa campagna, perciò non viene dichiarato un sorpasso.


## Riproduzione

Gli eseguibili opzionali `qw3-projection-prefill-test`, `qw3-projection-prefill-bench` e `qw3-projection-cost-bench` si costruiscono esplicitamente con Makefile; non entrano nella build ordinaria. Il primo accetta il percorso del modello e `q3_metadata`, `iq_metadata` o `stage_profile`; il terzo accetta `inventory`, `matrices` o `bandwidth`.

`tools/bench_qw3_candidates.py` registra comandi, hash di sorgenti/eseguibile, configurazioni e swap prima/dopo ogni lavoro. Impostazioni private: `QW3_PRIVATE_Q3_METADATA=1`, `QW3_PRIVATE_IQ_METADATA=1`, `QW3_PRIVATE_STAGE_PROFILE=1`, una alla volta. Il benchmark completo va sempre lanciato dal root del repository perché i diagnostici leggono gli shader opzionali da `tests/`.
