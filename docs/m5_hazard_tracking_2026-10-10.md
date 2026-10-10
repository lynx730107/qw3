# M5: tracking delle dipendenze Metal (2026-10-10)

Le prove restano private. Il sorgente di produzione e i kernel non cambiano.
Il requisito confermato dall’utente è mantenere risultati identici bit per bit,
oltre alla corretta generazione e al roundtrip dei checkpoint.
L'SDK locale `MTLResource.h` specifica che i buffer creati dal device sono
tracked per default; una verifica diretta su Apple M5 conferma modalità 2
(tracked) e 1 (untracked) secondo le opzioni richieste.

## Buffer privati di sessione: scartati

Il generatore modifica esclusivamente le allocazioni effettuate da
`qw3_metal_new_private_buffer`. Due sessioni verificano la modalità effettiva
di ciascuna allocazione e del buffer x0. Le barriere e gli encoder rimangono
quelli nativi. Il primo prefisso di un token fallisce sul logit 0:
6,95593739 contro 3,68741369. Nessun prefisso completo, checkpoint o benchmark
è superato per questa variante. Non si attribuisce il risultato a una specifica
barriera mancante senza una prova aggiuntiva.

[Log e manifest](measurements/hazard_tracking_quality_2026-10-10/manifest.json)
con exit code 1, snapshot del sorgente e hash invariati. La compilazione non
produce warning. Il test non viene usato per misurare velocità.

## Sole viste GGUF: controllo numerico superato

La GPU legge i pesi senza scriverli. La prova crea due insiemi di buffer Metal
no-copy sullo stesso mmap in sola lettura: uno tracked e uno untracked.
Prima di scegliere l'insieme, sincronizza la coda e verifica la modalità
reale. Entrambe le sessioni mantengono buffer privati, kernel, FP32,
cache64 e prefill nativi; non cambia il contenuto del modello.

Otto prefissi (1, 15, 16, 31, 127, 128, 129, 159), estensione 128→159,
intero vocabolario finito ed esatto bit per bit, history e stato canonico
K/V F16, DeltaNet e convoluzione esatti. Per ogni prefisso, roundtrip del
payload checkpoint e 32 continuazioni esatte. Una vista di 13.135.413.248
byte per insieme; 266 selezioni tracked e 265 untracked, nessuna vista
GGUF temporanea. Questi contatori verificano selezioni e modalità dei
buffer, non contano singoli dispatch.

[Controllo completo](measurements/weight_tracking_quality_2026-10-10/manifest.json)
con exit code 0, sorgenti e binario invariati. La compilazione non produce warning.

Due repliche private usano due sessioni ctx512, quattro thread, GPU64,
KV F16, prefisso seed1 di 128 token, otto forward greedy di warmup e
64 coppie temporizzate AB/BA, dopo 60 secondi di pausa. Il cambio dei
buffer e il confronto dell'intero vocabolario sono fuori dal timer.
Logits finiti ed esatti dopo ogni forward, stato canonico finale esatto;
tutti i 128 campioni accoppiati e gli otto warmup per replica vengono
eseguiti, i log conservano i campioni temporizzati.

| Replica | Guadagno | Risparmio ms/forward, intervallo descrittivo 95% | Blocchi positivi |
| --- | ---: | ---: | ---: |
| 1 | +3,838% | 5,956±3,641 | 7/8 |
| 2 | +4,362% | 6,734±3,998 | 7/8 |

Gli intervalli Student sulle otto medie di blocco non eliminano deriva o
autocorrelazione. Il throughput di circa 6,2→6,4/6,5 token/s è molto
inferiore al decode ordinario precedente; due sessioni e due viste alias
dei pesi possono modificare il comportamento del sistema. Non si usa
questa prova come confronto con llama o come dimostrazione di un guadagno
nativo. [Serie complete](measurements/weight_tracking_paired_2026-10-10/results.json).

## Controllo ordinario

Il benchmark privato nativo modifica soltanto le opzioni delle due funzioni
che creano viste GGUF no-copy. Ogni processo crea un solo insieme di viste;
nessun cambio di buffer durante il benchmark. Il flag viene letto prima
dell'apertura del modello e la modalità effettiva dei buffer viene verificata.
Sessione singola ctx512, GPU64, KV F16, quattro thread, input fissi seed1,
protocollo stock `--llama-style`, tg32 a profondità128, cinque ripetizioni
per blocco e warmup nativo. Sette blocchi OFF/ON/OFF/ON/OFF/ON/OFF,
pausa di 60 secondi prima di ogni blocco. Tutti i campioni vengono conservati.

La serie completa conserva 35 campioni. Il throughput aggregato è
8,876154 token/s OFF contro 8,876495 ON: **+0,00384%, effetto neutro**.
Le medie OFF sono 3594,36 / 3602,44 / 3617,06 / 3606,80 ms per tg32;
la loro dispersione è 0,632%, sotto la soglia del 5%. Sorgenti e binario
restano invariati. Il swap riportato dal sistema non cambia tra l'inizio
e la fine di ciascun blocco.

Il vantaggio del +3,8/+4,4% osservato nella prova accoppiata **non si
conferma nel decode ordinario**. Non si dimostra quale differenza del
protocollo produca quel segnale. Nessuna delle due modifiche di tracking
viene adottata; nessun guadagno viene attribuito al confronto con llama.
[Misure ordinarie complete](measurements/weight_tracking_native_perf_2026-10-10/results.json).

```sh
make qw3-hazard-tracking-test qw3-weight-tracking-test qw3-weight-tracking-bench qw3-weight-tracking-native-bench
QW3_METAL_KV_F16=1 QW3_METAL_NGL=64 ./qw3-weight-tracking-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf all
python3 tools/bench_weight_tracking.py --binary ./qw3-weight-tracking-bench --output /tmp/weight-paired-new --cooldown 60
python3 tools/bench_qw3_candidates.py --binary ./qw3-weight-tracking-native-bench --output /tmp/weight-native-new --order off,candidate,off,candidate,off,candidate,off --jobs tg32_d128 --candidate-env QW3_PRIVATE_WEIGHT_UNTRACKED=1 --cooldown 60 --repetitions 5
```
