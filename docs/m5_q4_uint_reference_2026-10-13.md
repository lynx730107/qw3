# Confronto pubblico Q4_K uint con llama.cpp su M5

La versione integrata supera di poco llama.cpp nel prefill pp128 in questa
serie, ma resta indietro nel decoding ordinario tg32 a profondità 128.
L'obiettivo complessivo rimane aperto: serve ancora +2,383% di throughput
nel decoding rispetto al qw3 misurato qui.

| Lavoro | qw3, token/s | llama.cpp, token/s | Variazione qw3 |
| --- | ---: | ---: | ---: |
| pp128 | 162,56927 | 162,30751 | +0,161% |
| tg32, depth128 | 9,13452 | 9,35218 | −2,327% |

Si usano lo stesso GGUF Qwen3.8-27B UD-Q3_K_XL, KV F16, ctx256 e offload
completo di trunk/head. I flussi sintetici nativi dei due motori sono
diversi; non si misura sampling né MTP. Sei coppie AB/BA per lavoro,
cinque ripetizioni per processo, warmup e 60 secondi di pausa prima di
ciascun processo. Un solo processo modello alla volta. Tutti i 24 processi
terminano con successo: 120 campioni conservati, zero scarti.

Il tempo risparmiato da qw3 sui 128 token di prefill è 1,270 ms, con
intervallo paired al 95% [0,435; 2,105] ms. Nel decoding perde 81,533 ms
sui 32 token, intervallo [62,809; 100,257] ms a favore di llama. Le unità
sperimentali sono sei medie di coppia per lavoro; gli intervalli sono
condizionati all'indipendenza delle coppie separate nel tempo. Il segno
medio resta lo stesso nei gruppi qw3-primo e qw3-secondo per entrambi i
lavori. Il prefill ha una coppia negativa: il piccolo vantaggio aggregato
non implica che qw3 vinca ogni esecuzione.

Gli spread max/min delle medie dei processi sono 0,518%/0,462% per pp128
e 0,677%/1,013% per tg32, qw3/llama: tutti sotto il limite del 5%.
Non si confrontano tassi assoluti con altre campagne per attribuire
all'ottimizzazione una percentuale di miglioramento. Il suo beneficio
contro il kernel byte è la distinta replica privata
[Q4_K word](m5_q4_words_2026-10-13.md), +0,934%.

Le righe qw3 riportano 174.664.400 byte di sessione in pp128 e 176.761.680
in tg32. La modifica non introduce buffer di dati della sessione né copie
dei pesi; aggiunge un oggetto di pipeline. La RAM totale e il picco non
sono misurati. Lo swap osservato ai confini passa da 1990,25 a 1862,25 MiB
e cambia anche durante alcuni processi: non è dichiarato costante.

Il confronto usa i binari congelati del worktree isolato al commit
`571bbe1`, con la stessa integrazione ora nel branch (`1dc41c8`). Sorgenti,
snapshot, due eseguibili e otto librerie dirette llama restano invariati;
dimensione e mtime del GGUF coincidono. Il verificatore rilegge CSV/JSON,
singole ripetizioni, configurazioni, offload, comandi e ricalcola le
statistiche. Le prove numeriche bit per bit, sedici checkpoint e quattro
controlli dell'agente sono nell'[archivio dell'integrazione](m5_q4_uint_integration_2026-10-13.md).

Il primo launcher si fermava prima di caricare il modello o misurare
tempi, perché il checkout llama relativo non esisteva dal worktree in
`/tmp`. È stato corretto con `--llama-checkout` esplicito e il tentativo
fallito resta nell'archivio separato
`measurements/q4_uint_reference_2026-10-13_initial_launcher_failure/`.

[Dati e configurazione](measurements/q4_uint_reference_2026-10-13/results.json),
[analisi ricalcolata](measurements/q4_uint_reference_2026-10-13/analysis.json)
e `verify_campaign.py` conservano l'evidenza completa. L'archivio comprende
log e snapshot delle sorgenti; non contiene binari, GGUF o checkpoint.
