# Dequantizzazione IQ4_XS nei registri M5 — 6 ottobre 2026

**Le due implementazioni provate sono corrette nei confronti isolati ma nettamente più lente del kernel nativo. Nessuna è integrata.** Il default di inferenza resta packed8 con scratch diretto. Sono conservati tutti i **60 campioni in 12 blocchi** e sono passati **32 casi numerici isolati**.

## Implementazione e verifica

La proposta usa i tensori cooperativi di input di TensorOps per consegnare direttamente i pesi dequantizzati nei registri. Apple descrive questa possibilità nella [sessione WWDC26](https://developer.apple.com/videos/play/wwdc2026/330/). L'API compila e funziona sul sistema effettivamente provato: MacBook Air Apple M5, 24 GB, macOS 26.6.2, SDK 26.5. Questa verifica non richiede di supporre un aggiornamento del sistema operativo.

- **v2, diretta:** stesso descrittore del nativo IQ4_XS staged128, tile di 128 token × 64 output, quattro gruppi SIMD, K=32, input FP32 e pesi F16, accumulatore FP32 e stesso ordine K. Riempie il tensore cooperativo tramite `qw3_iq4xs_dequant_k`, eliminando staging threadgroup e le due barriere per K.
- **v3, indici precalcolati:** stessa operazione, ma coordinate e maschere preparate fuori dal ciclo K; riempimento esplicitamente srotolato su 128 slot massimi. Gli slot inattivi restano mascherati; una capacità maggiore viene rifiutata. È una seconda prova limitata a cinque blocchi, decisa prima della sua misura.
- **v1:** prima compilazione Metal fallita perché l'alias del tensore threadgroup ometteva `tensor_inline`. La correzione è inclusa in v2; errore e sorgenti originari restano [archiviati](measurements/register_iq4_2026-10-06/v1/manifest.json). Nessun campione prestazionale v1.

Ogni variante finale passa 16 confronti contro il kernel nativo: pesi GGUF reali down 17408→5120 e gate 5120→17408; 128/256 token; 1, 33, 65 e tutte le righe di output. Tutti gli output sono finiti e identici bit per bit. Offset e stride non compatti, guardie di output e input immutati; layout invalidi e numero di token non supportato rifiutati. Durante le misure l'output viene nuovamente confrontato dopo ogni command buffer.

## Misure isolate

Stesso `Qwen3.8-27B-UD-Q3_K_XL.gguf`, prima matrice down IQ4_XS `blk.2.ffn_down.weight`, 128 token e input FP32 pseudocasuale seed1. Motore configurato ctx512, KV F16, 64 layer Metal e quattro thread; questi parametri non trasformano il microbenchmark in una misura del modello completo.

Una sola inferenza GPU attiva; pipeline compilate prima delle misure, un caricamento del GGUF, 60 secondi prima di ogni blocco, un command buffer di warmup escluso, cinque ripetizioni per blocco, otto dispatch identici per command buffer. Tempo GPU del lavoro completato, diviso per otto. Ambiente ripulito dalle opzioni QW3_/GGML_ ereditate. Il calcolo dei logits, gli stati e i checkpoint non sono eseguiti in questo test.

Ordine v2: `OFF/v2/OFF/v2/OFF/v2/OFF`; ordine v3: `OFF/v3/OFF/v3/OFF`.

I valori ± sono deviazioni standard campionarie, non intervalli di confidenza. La differenza di velocità è `100 × (tempo OFF / tempo candidato − 1)`. Ogni confronto vicino usa entrambi gli OFF adiacenti. Le varianti v2/v3 vanno confrontate con i propri controlli, non direttamente fra campagne.

| Variante | Campioni | OFF, ms ± SD | Candidato, ms ± SD | Tempo candidato/OFF | Differenza velocità | Confronti vicini |
|---|---:|---:|---:|---:|---:|---|
| v2 | 35 | 2,330 ± 0,072 | 12,614 ± 0,218 | 5,414× | -81,528% | -81,636%; -81,228%; -81,491% |
| v3 | 25 | 2,493 ± 0,226 | 16,812 ± 0,120 | 6,744× | -85,173% | -84,996%; -85,678% |

Dispersione delle medie dei blocchi OFF, massimo/minimo meno uno: **5,078%** in v2 e **12,300%** in v3. Il primo OFF di v3 include il campione 3,219 ms; resta incluso. Tutti i confronti vicini sono fortemente negativi, quindi la variabilità dei controlli non cambia la decisione.

| Campagna | Blocco | Modalità | ms ± SD, cinque campioni |
|---|---:|---|---:|
| v2 | 0 | off | 2,283 ± 0,059 |
| v2 | 1 | register | 12,745 ± 0,358 |
| v2 | 2 | off | 2,399 ± 0,052 |
| v2 | 3 | register | 12,565 ± 0,012 |
| v2 | 4 | off | 2,319 ± 0,090 |
| v2 | 5 | register | 12,531 ± 0,073 |
| v2 | 6 | off | 2,320 ± 0,036 |
| v3 | 0 | off | 2,676 ± 0,324 |
| v3 | 1 | register | 16,858 ± 0,115 |
| v3 | 2 | off | 2,383 ± 0,096 |
| v3 | 3 | register | 16,767 ± 0,119 |
| v3 | 4 | off | 2,420 ± 0,035 |

## Interpretazione e decisione

Eliminare le barriere non basta a ridurre il lavoro totale. Il kernel nativo dequantizza in forma vettoriale un tile condiviso dai quattro gruppi SIMD. Questi prototipi preparano invece un input cooperativo separato in ciascun gruppo, con dequantizzazione scalare. È una differenza strutturale che può contribuire al costo; non è una diagnosi misurata di register spill o occupancy. Non sono stati raccolti contatori hardware per attribuire causalmente il rallentamento.

L'esito respinge **queste due implementazioni e questa geometria**; non dimostra che ogni uso futuro dei tensori cooperativi sia più lento. Precalcolare gli indici non recupera prestazioni in questo caso. Non estendo il medesimo schema a IQ3_S e non lo porto all'inferenza completa. Non dichiaro guadagni di pp128 o generazione, né identità dell'intero modello sulla base di soli test di matrice.

I dieci sorgenti di inferenza confrontati con la precedente regressione nativa conservano gli stessi SHA-256; preservate le altre modifiche già presenti nella checkout. I diagnostici sono target opzionali e non cambiano il percorso del prodotto.

## Archivi e riproduzione

- [v2: manifest e identità](measurements/register_iq4_2026-10-06/v2/manifest.json), [16 casi di qualità](measurements/register_iq4_2026-10-06/v2/quality.stdout), [35 campioni e analisi](measurements/register_iq4_2026-10-06/v2/benchmark.json).
- [v3: manifest e identità](measurements/register_iq4_2026-10-06/v3/manifest.json), [16 casi di qualità](measurements/register_iq4_2026-10-06/v3/quality.stdout), [25 campioni e analisi](measurements/register_iq4_2026-10-06/v3/benchmark.json).
- Ogni directory conserva build, stdout/stderr grezzi e snapshot dei sorgenti; [esito e verifica produzione](measurements/register_iq4_2026-10-06/outcome.json).

Build: `make qw3-register-iq4-test qw3-register-iq4-cached-test`. I binari accettano `MODEL quality` e `MODEL bench 60`; avviarli dalla root del progetto, uno alla volta, con ambiente QW3_/GGML_ ripulito, KV F16 abilitato e KV Q8 disabilitato. La qualità deve passare prima delle misure. Gli hash dei binari effettivamente usati sono nei manifest.
