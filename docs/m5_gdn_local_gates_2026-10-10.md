# M5: gate locali nel decode DeltaNet (2026-10-10)

Due prove private eliminano la prima barriera del kernel cache64, lasciando
invariati cache, ricorrenza, somme FP32 e normalizzazione. La prima calcola
beta e gamma in ciascun thread; la seconda nel lane 0 di ciascun gruppo
SIMD e distribuisce i risultati con `simd_shuffle`. La baseline è la
pipeline cache64 nativa adottata, non il kernel precedente senza cache.

Quattro casi di matrice coprono entrambe le varianti, dimensione 128,
3/7 value heads e 2 query heads. Alpha include −21, +21 e 0,25; beta
−80, +80, 0,7 e −0,35. Stato in-place e output finiti bit per bit, offset
non nulli, guardie e tutti gli input immutati. Il riferimento CPU doppio
per RMSNorm/gate, a partire dallo stato GPU aggiornato, ha errore massimo
2,845e-7; non verifica indipendentemente l'intera ricorrenza.

Due sessioni ctx512, prefisso seed 1 di 128 token, 64 layer Metal,
KV F16 e quattro thread. Pausa di 60 secondi per variante; 72 forward
greedy AB/BA, otto warmup e 64 temporizzati. Vocabolario completo finito
ed esatto dopo ogni forward; history e stato canonico K/V, DeltaNet e
convoluzione finali esatti. 3456 dispatch candidati e altrettanti della
baseline per variante. Tutti i 128 campioni accoppiati sono conservati.

| Variante | Guadagno | Risparmio ms/forward, intervallo descrittivo 95% | Blocchi positivi |
| --- | ---: | ---: | ---: |
| Per thread | +0,016% | 0,018±0,184 | 3/8 |
| Per gruppo SIMD | +0,001% | 0,002±0,165 | 4/8 |

**Nessuna variante viene adottata**: i risultati sono neutri. Gli intervalli
Student sulle otto medie di blocco non eliminano autocorrelazione e deriva.
Questi throughput di due sessioni misurano l'effetto relativo, non il decode
ordinario di una sessione singola. Sorgenti e binari sono invariati.
[Archivio completo](measurements/gdn_local_gates_2026-10-10/results.json).

Il controllo completo su otto prefissi e checkpoint è disponibile
(`MODEL all ID`), ma non viene eseguito per le varianti scartate. Restano
valide le verifiche native della cache64 già adottata.

```sh
make qw3-gdn-local-gates-test qw3-gdn-local-gates-bench
QW3_METAL_KV_F16=1 ./qw3-gdn-local-gates-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf
QW3_METAL_KV_F16=1 caffeinate -disu ./qw3-gdn-local-gates-bench ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 1
# Ripetere con ID 2 dopo la pausa.
```
