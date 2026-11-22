# Inkling 2 Mini in qw3

Stato al 21 novembre 2026. Le prove sono state eseguite su un MacBook Air M5 con 24 GB di memoria unificata. Il GGUF validato in `qw3` è l’originale Unsloth `Inkling-2-Mini-UD-Q3_K_XL.gguf` da 13,63 GB.

## Stato attuale

`qw3-cli` esegue la generazione testuale a turno singolo con l’intero grafo del modello su Metal e **cache K/V FP16**. La GPU calcola embedding, normalizzazioni, residui, controlli del loop, attenzione, convoluzioni, routing, proiezioni degli esperti e logits. Le attivazioni e lo stato persistente risiedono in buffer Metal privati. In generazione, la CPU prepara i token, coordina gli invii, legge gli otto indici per indirizzare i pesi degli esperti e campiona dai logits finali. Il prefill seriale è il default. Il percorso sperimentale a batch su Metal 4 si attiva con `QW3_INKLING_PREFILL_BATCH=128`.

Il modello esegue **48 visite su 24 strati fisici**. I primi sei e gli ultimi sei strati vengono visitati una volta. Il blocco centrale di dodici strati viene percorso tre volte e condivide i pesi, con stato K/V e cronologia delle convoluzioni distinti per visita. Otto visite usano attenzione globale. Le altre quaranta usano un buffer circolare da 1.024 token.

L’attenzione Metal applica il bias relativo direttamente durante il calcolo dei punteggi. Ogni partizione copre fino a 128 posizioni. Il merge usa accumulatori FP32 e include l’attention sink nella normalizzazione. `--cpu` conserva il percorso di riferimento con cache K/V FP32. [Grafo e stato](../qw3_inkling_cpu.inc) · [Sessione Metal](../qw3_metal.m) · [Kernel di attenzione](../metal/qw3_core_sequence.metal)

## Lavoro svolto

1. **Riferimento con llama.cpp.** Avviata la build Unsloth `b12588-mix-c0c3de8` con i GGUF `UD-Q4_K_XL` e `UD-Q3_K_XL`. A 32.768 di capacità con KV F16, il Q4 ha esaurito la memoria Metal con tutti gli esperti sulla GPU. Spostando gli esperti dei primi quattro strati su CPU, ha completato un prompt da 31.814 token. Il Q3 ha completato le prove lunghe a 32k e 64k con tutti gli esperti su Metal. I launcher `start-inkling-32k.sh` e `start-inkling-64k.sh` usano il server locale sulla porta 8080 e stampano i log diagnostici all’avvio.
2. **Inferenza nativa su CPU.** Aggiunti riconoscimento del profilo `inkling2`, ispezione e validazione di metadati e tensori, dequantizzazione NVFP4, tokenizer BPE con pre-tokenizzazione GPT-4o tramite ICU di macOS, formato chat Inkling e gestione dei token di controllo. Implementate le 48 visite con le rispettive cache e il replay dopo rewind. [Loader e tokenizer](../qw3.c) · [CLI](../qw3_cli.c)
3. **Primo offload Metal.** Trasferite le grandi proiezioni e gli esperti. Nella successiva versione ibrida, accorpare i controlli del loop e il calcolo degli esperti e spostare il piccolo router F32 su CPU ha ridotto gli invii sincroni da 337 a 193 per token. Sullo stesso prompt breve, la media di due misure per versione in ordine alternato è passata da 7,29 a 9,42 tok/s in generazione (+29%).
4. **Grafo Metal completo e cache FP16.** Trasferite attenzione, convoluzioni, residui, normalizzazioni, embedding e routing. Ogni visita ha una propria cache privata Metal. I pesi rimangono nel formato quantizzato del GGUF. In Metal FP16 è il formato KV predefinito ed è selezionabile esplicitamente con `--kv-f16`.

## Verifica e misure native

- I test CPU e Metal verificano il tokenizer, la dimensione della cache, il top 5 dei logits rispetto a llama.cpp su tre posizioni e il replay esatto dopo rewind. Nella prova breve full Metal, lo scarto assoluto massimo rispetto a llama.cpp sui 200.058 logits per posizione è circa 0,00155. Passati anche il controllo del fingerprint del runtime e la regressione di `qw3-agent` con Qwen3.8-27B `UD-IQ3_XXS`. [Test Inkling](../tests/test_inkling.c) · [Regressione agent](../tests/test_dense_agent.sh)
- Con capacità di **65.536 token**, il runtime ha allocato **2.208 MiB (2,16 GiB)** di sola cache KV. Ha elaborato una sequenza sintetica di **1.032 token**, attraversando il primo riuso degli slot della finestra. Il token più probabile coincide con llama.cpp in tutti gli otto punti campionati. Il riempimento quasi completo a 64k è stato verificato con llama.cpp.
- Nei punti campionati del prompt lungo i logits grezzi a volte divergono. Lo scarto massimo osservato è circa 5,21 alla posizione 1.025. Su un confronto aggiuntivo limitato ai primi 129 token, disattivare Flash Attention in llama.cpp ha ridotto lo scarto massimo alla posizione 127 da 2,03 a 0,12. In quella prova il top 5 ordinato coincide in tutte e sei le posizioni campionate. Le risposte native brevi `OK` e `17 × 6 = 102` con spiegazione sono corrette.

Le righe sotto riportano singole prove da CLI. I token generati includono ragionamento e token di controllo. La velocità di caricamento e la pressione sulla memoria variavano tra le esecuzioni.

| Percorso e prompt | Capacità | Token prefill / generazione | Prefill, tok/s | Generazione, tok/s |
|---|---:|---:|---:|---:|
| CPU FP32, «Rispondi solo OK.» | 128 | 19 / 62 | 1,28 | 1,13 |
| Primo Metal ibrido, stesso prompt | 128 | 19 / 62 | 6,85 | 9,21 |
| Metal completo, stesso prompt | 128 | 19 / 62 | 11,14 | 10,98 |
| Metal completo, spiegazione di `17 × 6` | 256 | 33 / 115 | 9,66 | 11,33 |
| Metal completo, «Rispondi solo OK.» | 65.536 | 19 / 62 | 2,46 | 8,79 |

La cache Metal occupa `4096 × (8 × ctx + 40 × min(ctx, 1024))` byte. Sono 1.184 MiB a 32.768 token e 2.208 MiB a 65.536. Il totale richiesto dal processo comprende anche i pesi, lo stato delle convoluzioni e i buffer di lavoro.

## Benchmark nativo e prefill a batch

`qw3-bench` ora carica Inkling e accetta una lista di frontiere esplicite. Il validatore confronta gli ID esatti dei prefissi prima di interrogare llama.cpp. Il primo archivio ABBA con prefill seriale contiene tre prove complete. Da contesto vuoto misura 16,57 tok/s in prefill e 16,50 tok/s in generazione per `qw3`, contro 217,37 e 22,18 tok/s per llama.cpp. I benchmark sintetici usano le stesse lunghezze, con flussi di token generati dai rispettivi backend. [Risultati ABBA](measurements/inkling-baseline-2026-11-21/abba/results.json) · [Validatore dei contesti](../tools/validate_inkling_long_context.py)

Il prefill a batch usa le visite del grafo seriale, proiezioni a blocchi, routing e indirizzamento degli esperti su GPU. I K/V nuovi rimangono nello scratch fino al termine delle query del blocco, poi vengono scritti nel ring. Il nuovo test Metal verifica l’output delle due query che riusano gli slot e il contenuto finale del ring. Sono state corrette anche la lettura degli indici IQ2_S e la scelta delle proiezioni F32. [Revisione della cache](../metal/qw3_core_sequence.metal) · [Regressione della finestra](../tests/test_inkling_window.m)

In una prova entro lo stesso processo con 32 token, il percorso a batch ha raggiunto 123,66 tok/s contro 14,57 tok/s in seriale. I logits possono divergere e alcuni ranking cambiano, quindi il batch resta opt-in.

Il confronto esplorativo usa gli stessi **8.283 ID di input**, il GGUF Q3, cache FP16 e 9.216 token di capacità. Le due prove llama.cpp sono state eseguite prima delle prove native. Sia batch 32 sia batch 128 hanno recuperato esattamente il codice `MX-40719-QZ`. [Misure e hash del prefisso](measurements/inkling-baseline-2026-11-21/matched-8k.json)

| Percorso | Prefill, tok/s |
|---|---:|
| `qw3`, batch 32 | 84,56 e 60,53 |
| `qw3`, batch 128 | 123,40 |
| llama.cpp, microbatch 128 e Flash Attention | 146,04 e 127,62 |

## Generazione con prefisso breve e a 8k

Sul M5, entrambi i backend hanno elaborato gli stessi prefissi e gli stessi 128 token di continuazione forzata. Il timer parte dopo il prefill e otto passi di riscaldamento. Ogni passo del decode rende disponibili i logits sulla CPU. La capacità è di 9.216 token e la KV è FP16. llama.cpp usa Flash Attention e cache scorrevole compatta, come il server e `llama-bench`. La KV occupa 448 MiB in `qw3` e 488 MiB in llama.cpp, che alloca 1.280 celle per visita scorrevole con questo microbatch. [Protocollo e misure grezze](measurements/inkling-baseline-2026-11-21/matched-decode.json)

| Token di prefisso | `qw3`, tok/s | llama.cpp, tok/s | Scarto di `qw3` |
|---:|---:|---:|---:|
| 128 | 14,31 (12,75–16,32) | 15,68 (14,48–17,11) | −8,7% |
| 8.283 | 13,76 (12,26–15,66) | 13,91 (13,66–14,16) | −1,1% |

Ogni valore aggregato divide i token elaborati per il tempo totale delle due prove. Le parentesi riportano il minimo e il massimo. A 8k la differenza tra i valori aggregati è inferiore alla variabilità tra le prove. Nella generazione libera del test precedente, `qw3` con prefill a batch 128 aveva prodotto 145 token a 14,06 tok/s e restituito il codice esatto.

## Riferimento prestazionale con llama.cpp

Nel confronto con `llama-bench` Unsloth b12588, i due modelli hanno usato Metal, KV F16, Flash Attention e batch e microbatch da 512. Sono state aggregate sei ripetizioni per prova con ordine dei processi Inkling–Qwen–Qwen–Inkling. Il test usa token sintetici e misura il throughput.

| Prova | Inkling `UD-Q3_K_XL`, tok/s | Qwen3.8-27B `UD-IQ3_XXS`, tok/s |
|---|---:|---:|
| Prefill 512 da contesto vuoto | 206,70 | 121,92 |
| Generazione 128 da contesto vuoto | 19,37 | 7,29 |
| Prefill 512 dopo 1.024 token | 165,10 | 106,66 |
| Generazione 128 dopo 1.024 token | 18,58 | 6,26 |

Con lo stesso GGUF Inkling Q3 e prompt reali quasi pieni, il server llama.cpp ha recuperato correttamente un codice vicino all’inizio del prompt.

| Capacità | Token effettivi di prompt | Microbatch | Prefill, tok/s | Generazione, tok/s |
|---:|---:|---:|---:|---:|
| 32.768 | 31.814 | 64 | 91,93 | 15,94 |
| 65.536 | 63.992 | 128 | 76,88 | 12,03 |

Nelle prove di prefill di llama.cpp, il microbatch 256 ha funzionato con la finestra 32k e un prompt breve. A 64k ha esaurito la memoria Metal durante la prova. Il microbatch 128 ha completato il prompt da 63.992 token. I resoconti, i launcher e i dati grezzi di llama.cpp si trovano localmente in `~/Downloads/llama-inkling-b12588/`, in particolare in `contesto.md`, `bench-qwen27b.ef3ZUu/` e `prefill-inkling.XY1dyC/`.

## Cosa manca

| Priorità | Attività | Verifica attesa |
|---:|---|---|
| 1 | **Consolidare la baseline fino a 8k.** Il benchmark nativo e il validatore dei prefissi sono disponibili. | Confrontare `qw3` e llama.cpp sullo stesso GGUF e sugli stessi token. Ripetere le misure a freddo e a caldo e distinguere prefill, generazione e allocazione. |
| 2 | **Convalidare il prefill a batch su Metal.** Il percorso sperimentale è collegato alla CLI e al benchmark. | Misurare il guadagno per microbatch a 8k e verificare qualità e logits rispetto al percorso seriale, anche al cambio di blocco. Il recupero con finestre quasi piene a 32k e 64k resta una verifica di correttezza e memoria da fare dopo il lavoro sul prefill a 8k. |
| 3 | **Ridurre le sincronizzazioni durante la generazione.** Misurare il tempo speso in calcolo, invii e attese del router. Valutare kernel per gli esperti con indirizzamento tramite indici già residenti sulla GPU. | Stessi esperti selezionati e risposte coerenti. Maggior throughput ripetibile con contesti brevi e lunghi, a parità di cache FP16. |
| 4 | **Completare l’integrazione nelle interfacce del progetto.** `qw3-agent` usa il formato chat e le chiamate agli strumenti di Inkling. Restano `qw3-eval`, la chat multi-turno nella CLI, la compattazione, i checkpoint KV e la cache persistente dei prefissi. | In `qw3-agent` sono stati verificati lo strumento `read` e la ripresa di una conversazione con replay dei token. Restano da verificare la ricerca web e gli altri strumenti. I launcher HTTP usano llama.cpp. |
| 5 | **Estendere la compatibilità e valutare l’uso al posto di Qwen.** Convalidare un altro GGUF, a partire da `UD-Q4_K_XL` se serve, e confrontare i due modelli sulle attività reali dell’utente. | Parità di qualità su istruzioni in italiano, codice, strumenti e recupero da contesti lunghi. Per Q4 serve anche una misura della memoria sul M5. |

## Come ripetere le prove native

Eseguire dalla radice di `qw3`.

```sh
model="../../models/Inkling-2-Mini-UD-Q3_K_XL.gguf"
make qw3-cli
./qw3-cli -m "$model" --inspect
./qw3-cli --metal --kv-f16 -m "$model" \
  --ctx 128 --nothink --temp 0 -n 80 -p "Rispondi solo OK."
QW3_INKLING_PREFILL_BATCH=128 ./qw3-cli --metal --kv-f16 -m "$model" \
  --ctx 128 --nothink --temp 0 -n 80 -p "Rispondi solo OK."
QW3_INKLING_MODEL="$model" make test-inkling-cpu test-inkling-metal
make test-inkling-window
make test-runtime-fingerprint
```

Per ripetere il controllo di allocazione e generazione breve a 64k, usare `--ctx 65536`. Il flag `--nothink` imposta lo sforzo di ragionamento a zero. Inkling può comunque consumare parte del budget prima della risposta visibile.

## Prove con qw3-agent

```sh
make qw3-agent
./start-inkling-agent.sh --read-only --compact-tools
```

Il launcher usa Inkling `UD-Q3_K_XL`, Metal con KV FP16, 8.192 token di contesto e microbatch di prefill 128. La cronologia è in `~/.qw3-agent-inkling`. Con `--read-only --compact-tools`, il modello dispone dello strumento `read`.

1. Scrivere `Rispondi solo OK.`.
2. Scrivere `Usa read per leggere la prima riga di README.md. Rispondi solo con il titolo esatto.`.

Per registrare le velocità dei singoli turni, passare `--trace /tmp/inkling-agent.jsonl` e leggere `prefill_tps` e `generation_tps` negli eventi `inference`.
