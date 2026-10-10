# Microfusioni sul default M5 del 5 ottobre

Questa campagna parte dal default packed8 con scratch diretto già verificato.
I candidati sono isolati in `qw3-micro-test`, `qw3-micro-q8-test` e
`qw3-micro-bench`; non modificano il codice di produzione durante le misure.

## Candidati

- **Alpha/beta prefill:** un solo kernel Q8_0 batch4 con due serie di quattro
  accumulatori indipendenti, input condiviso, identico ordine di accumulo e
  `simd_sum` per ogni output. I 48 output non usavano già la via Tensor/MM,
  che richiede almeno 64 righe. Si modifica soltanto il percorso dense con
  DeltaNet completamente batched; gli altri layer conservano il fallback.
- **Alpha/beta decode:** una sola griglia contiene entrambe le proiezioni.
  Ogni workgroup mantiene l'aritmetica del matvec singolo, lo stesso numero
  di thread e le stesse riduzioni. Il riuso dell'input dentro il medesimo
  accumulatore è stato escluso: una prima variante con due accumulatori
  cambiava gli ultimi bit (71 dei 96 output sintetici del primo layer).
  Disabilitare la riassociazione non ripristinava l'identità (82 differenze).
- **Convoluzione e QK norm:** workgroup per token e head di 128 canali,
  convoluzione e SiLU in registri, stessa riduzione Q/K a 256 thread, scrittura
  del risultato normalizzato. V conserva soltanto la convoluzione/SiLU.
  L'aggiornamento dello stato conv rimane un dispatch separato dopo barriera.

`tests/prepare_micro_candidates.py` genera una copia privata del grafo da
`qw3.c`, sostituendo tre frammenti esatti e richiedendo un'unica corrispondenza
per ciascuno. I rami OFF chiamano le originali funzioni di produzione.
Il default packed8/scratch diretto è attivo su entrambi i lati.

## Identità numerica

Il [manifest della qualità](measurements/micro_pair_quality_2026-10-05/manifest.json)
conserva hash dei binari, snapshot dei sorgenti e conferma che i sorgenti di
produzione coincidono con quelli dell'ultima campagna di regressione.

La prima variante prefill isolata, la variante definitiva alpha/beta
prefill+decode e conv/norm isolata passano gli otto prefissi 1,15,16,31,127,128,129,159. Il caso 128 prosegue
fino a 159 prima del checkpoint. Ogni caso confronta l'intero vocabolario
finito bit per bit, KV F16 vivi, tutti gli stati DeltaNet e conv, token,
salvataggio/caricamento del payload e 32 passi di continuazione con confronto
di tutti i logits a ogni passo. Anche gli stati finali coincidono.

La prova isolata delle proiezioni passa 384 casi: tutte le 48 coppie reali
alpha/beta, sette lunghezze batch (1,15,16,31,127,128,129) e il matvec di decode,
con output e padding confrontati byte per byte e input batch invariato.

## Prestazioni

Cinque blocchi OFF/candidato/OFF/candidato/OFF, tre ripetizioni e 40 secondi
fra blocchi. Una sola inferenza GPU attiva. Stesso GGUF, ctx512, KV F16,
64 layer Metal, seed1 e quattro thread. Compilazione e warmup esclusi.
Prefill e conv/norm misurano pp128; la variante decode misura 32 forward a
profondità 128, compresi i logits, senza attribuirle il prefill di preparazione.

| Candidato | tok/s OFF → candidato | Variazione | Confronti vicini | Dispersione OFF |
|---|---:|---:|---:|---:|
| [Alpha/beta prefill](measurements/micro_prefill_perf_2026-10-05/results.json) | 159.814 → 157.439 | -1.486% | +0.280% / -3.269% | 1.043% |
| [Alpha/beta decode](measurements/micro_decode_perf_2026-10-05/results.json) | 8.995 → 8.837 | -1.752% | -3.658% / -0.151% | 1.213% |
| [Conv + QK norm](measurements/micro_conv_perf_2026-10-05/results.json) | 160.087 → 160.780 | +0.433% | +0.613% / +0.415% | 0.533% |

Alpha/beta prefill e decode non mostrano un vantaggio replicato: restano
privati. Conv/norm è positiva nei due confronti, ma il guadagno aggregato
(+0,433%) è minore della dispersione dei controlli (0,533%).

La [seconda serie conv/norm](measurements/micro_conv_repeat_2026-10-05/results.json)
usa sette blocchi alternati e cinque ripetizioni, con lo stesso binario,
shader, parametri e pausa di 40 secondi. Il dato aggregato è +0,564%
(159,242 → 160,140 tok/s), ma i confronti vicini sono +0,374%, −1,118% e
+1,798%. I controlli variano del 2,898%; l'ultimo OFF rallenta a 819,60 ms,
contro 796,52–800,55 ms negli altri OFF. Questo rende ingiustificato attribuire
il piccolo movimento aggregato alla fusione. Si conservano tutti i campioni,
compresi i blocchi peggiorativi; non si selezionano solo quelli favorevoli.

**Nessuna delle tre varianti viene integrata.** La bozza del percorso nativo
conv/norm è stata preparata separatamente, ma non applicata. Il default
packed8/scratch diretto rimane quello già verificato. I sorgenti di produzione
coincidono ancora con la manifest dell'ultima regressione: non sono state
introdotte modifiche a prefill, decode, logits o checkpoint pubblici.

Il campo storico `stable_baseline_5pct` del runner indica solo il superamento
di quella vecchia soglia e non giustifica una promozione: qui serve distinguere
un effetto di circa 0,4% dalla variabilità osservata. Non si dichiarano intervalli
di confidenza o significatività statistica da questi pochi blocchi.

Le percentuali riguardano esclusivamente la fase misurata e non si sommano
automaticamente a campagne precedenti.

## Riproduzione

```sh
make qw3-micro-test qw3-micro-q8-test qw3-micro-bench
QW3_METAL_KV_F16=1 QW3_METAL_KV_Q8_0=0 caffeinate -disu \
  ./qw3-micro-q8-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf
QW3_METAL_KV_F16=1 QW3_METAL_KV_Q8_0=0 caffeinate -disu \
  ./qw3-micro-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf pair
QW3_METAL_KV_F16=1 QW3_METAL_KV_Q8_0=0 caffeinate -disu \
  ./qw3-micro-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf conv
python3 tools/bench_qw3_candidates.py --output /tmp/qw3-micro-prefill \
  --binary ./qw3-micro-bench --jobs pp128 \
  --order off,candidate,off,candidate,off \
  --candidate-env QW3_PRIVATE_MICRO=prefill --repetitions 3 --cooldown 40
```

Per decode usare `QW3_PRIVATE_MICRO=decode` e `--jobs tg32_d128`.
Per conv/norm usare `QW3_PRIVATE_MICRO=conv` e `--jobs pp128`.

## Follow-up del 6 ottobre: conv/norm non integrata

È stata completata una sola nuova campagna: sette blocchi
OFF/candidato/OFF/candidato/OFF/candidato/OFF, cinque ripetizioni per blocco
e 60 secondi fra blocchi, con una sola inferenza GPU attiva. Si misura
esclusivamente pp128, inclusi i logits, sullo stesso
Qwen3.8-27B-UD-Q3_K_XL GGUF, ctx512, KV F16, 64 layer Metal, quattro thread
e seed1. Entrambi i percorsi conservano packed8 con scratch diretto.

Gli SHA256 di tutti i sorgenti rilevanti e dei tre binari privati coincidono
con il manifest della qualità del 5 ottobre, prima e dopo la campagna; anche
dimensione e mtime del modello coincidono. La verifica bit per bit completa
già archiviata viene quindi riutilizzata: otto prefissi, intero vocabolario,
KV e stati DeltaNet/conv, checkpoint e 32 passi di continuazione. Queste
prove non sono state rieseguite oggi. Il [preflight](measurements/micro_conv_followup_2026-10-06/preflight.json)
e l'[esito](measurements/micro_conv_followup_2026-10-06/outcome.json) documentano
questa decisione; i sorgenti sono archiviati anche nella nuova campagna.

| Blocco | Percorso | Media ms | Deviazione standard ms |
|---|---|---:|---:|
| 0 | OFF | 808,82 | 0,91 |
| 1 | Conv/norm | 811,55 | 6,16 |
| 2 | OFF | 808,72 | 1,34 |
| 3 | Conv/norm | 807,03 | 1,28 |
| 4 | OFF | 808,49 | 1,56 |
| 5 | Conv/norm | 807,25 | 1,44 |
| 6 | OFF | 810,00 | 1,34 |

Il risultato aggregato è **158,219 → 158,296 tok/s (+0,049%)**. La dispersione
fra le medie dei controlli OFF è **0,187%**; i tre confronti con i controlli
immediatamente precedente e successivo sono **−0,343%, +0,195%, +0,247%**.
Su tutti i singoli campioni, OFF ha media 809,008 ms e deviazione standard
1,339 ms; il candidato ha media 808,609 ms e deviazione standard 4,066 ms.
Il primo blocco candidato, sfavorevole e più variabile, è conservato integralmente.

I [risultati](measurements/micro_conv_followup_2026-10-06/perf/results.json),
i [35 campioni](measurements/micro_conv_followup_2026-10-06/samples.json) e
i file stdout/stderr per ciascun blocco conservano tutti i dati: nessun campione
è scartato. I contatori confermano 288 dispatch conv/norm per blocco candidato
(warmup e cinque ripetizioni sui 48 layer), zero nei controlli e zero fusioni
alpha/beta in tutti i blocchi.

**Il vantaggio non è ripetibile e distinguibile dalla variabilità osservata.**
La fusione resta diagnostica; nessuna integrazione nativa o modifica al default
viene applicata. La mascheratura dei soli otto SIMD sum e la replica sul binario
di produzione erano condizionate alla promozione e non vengono eseguite.
I sorgenti di produzione continuano a coincidere con quelli delle regressioni
già superate. I valori assoluti di giorni diversi non vengono usati come
confronto prestazionale. Il follow-up termina dopo questa campagna.
