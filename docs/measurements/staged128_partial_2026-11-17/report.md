# Prefill parziale su Apple M5, 17 novembre 2026

Il kernel `staged128` sui batch da 96 a 127 token ha ridotto la latenza di prefill del **30,94% a 96 token** e del **26,30% a 127 token** nella replica. Il candidato usa lo stesso GGUF e le dimensioni logiche del prompt. L'esperimento è isolato dalla checkout sul branch `codex/m5-deltanet-prefill`.

| Token | Percorso attuale, ms | Candidato, ms | Riduzione latenza | Risparmio nelle coppie, ms |
|---:|---:|---:|---:|---|
| 96 | 1.027,52 | 709,62 | 30,94% | 333,48 e 302,33 |
| 127 | 1.281,30 | 944,28 | 26,30% | 477,91, 259,62, 295,63 e 314,94 |
| 128 | 857,50 | 841,11 | 1,91% | 24,56 e 8,24 |

A 128 token le modalità scelgono lo stesso percorso. Il risparmio stimato del controllo ha un intervallo t al 95% di −87,29 a +120,08 ms. Le quattro coppie a 127 token danno un intervallo descrittivo di +183,18 a +490,86 ms. Le poche coppie e la variazione nel corso della sessione limitano la precisione di queste stime.

La misura iniziale sul binario di controllo, prima della replica, ha dato **986, 1.086, 809, 909 e 1.292 ms** rispettivamente per 96, 127, 128, 129 e 159 token. I contatori mostrano un blocco singolo a 127 con 156 proiezioni dirette IQ4_XS e 73 dirette IQ3_S. A 128 ci sono 156 staged IQ4_XS e 111 staged IQ3_S, perché il motore esegue gate e up separatamente. A 129 il motore completa un blocco da 128 e un token con il percorso di generazione. A 159 completa 128+31. Il candidato attiva il percorso parziale per chunk effettivi da 96 a 127.

La prima serie ha confrontato due varianti. La prima ha sostituito le proiezioni dirette IQ3_S e IQ4_XS. La seconda ha eseguito anche gate e up in due proiezioni staged, conservando l'arrotondamento F16. A 127 token i tempi medi sono stati 1.089,42 ms per il controllo, 953,99 ms per la prima variante e 909,50 ms per la seconda. A 96 token sono stati 986,71, 819,19 e 793,83 ms. La terza variante ha aggiunto Q3_K, Q4_K, Q5_K, IQ2_S e IQ3_XXS. Il confronto incrementale è risultato favorevole in quattro coppie su quattro, con variazioni marcate tra i processi. La replica ha perciò confrontato la terza variante con il controllo nello stesso binario, alternando l'ordine.

Tutti i gruppi SIMD continuano a partecipare alle barriere per preparare il tile di pesi. I gruppi che cadono oltre l'ultimo token saltano moltiplicazione e scrittura. Il gruppo con l'ultima riga parziale usa una vista con estensione dinamica limitata alla lunghezza del prompt. [Guida Metal Performance Primitives](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)

Le misure includono i logits del prefill, con GGUF `Qwen3.8-27B-UD-Q3_K_XL.gguf`, ctx 256, KV F16, 64 layer Metal, quattro thread, seed 1. Ogni processo ha tre campioni dopo il warmup. La replica ha 16 processi e 48 campioni. Sono conservati tutti i 138 campioni delle quattro serie. Il payload di sessione coincide fra le modalità a parità di lunghezza. Il candidato non alloca nuovi buffer di sessione o copie dei pesi. La RAM totale di picco non è stata misurata.

[Identità dei sorgenti](identity.json) · [Misure iniziali](baseline.json) · [Prima serie](screen.json) · [Formati aggiuntivi](formats.json) · [Replica](replica.json) · [Analisi](analysis.json)

I controlli numerici su 96, 97, 127, 128, 129, 159 e 224 token hanno confrontato **231 vettori completi di logits**, inclusi 32 token di continuazione per caso. Logits, KV F16 dal modello, DeltaNet, convoluzione e history coincidono bit per bit. Il checkpoint della stessa modalità conserva esattamente logits e stato. I primi test a 96 e 127 token passano anche con Metal GPU Validation. [Continuazioni e checkpoint](continuation-v3.log)

Nel confronto con llama.cpp a 127 token la classifica dei primi otto logits e gli otto token greedy coincidono. Controllo e candidato danno gli stessi risultati rispetto a llama.cpp, RMSE 0,002268 e massimo errore assoluto 0,009989. Entrambi superano le soglie del progetto di 0,001 e 0,005. Si tratta di uno scarto già presente nel controllo. [Controllo contro llama.cpp](frontier-v0.log) · [Candidato contro llama.cpp](frontier-v3.log) · [Criteri di accettazione](../../optimization_acceptance.md)

Il test dell'agente dense passa in entrambe le modalità, con checkpoint e cache del prefisso. La variante ha effettivamente eseguito 267 proiezioni staged parziali. Il test di pressione passa in entrambe le modalità. Il candidato esegue 801 proiezioni staged parziali e riprende correttamente un prompt da 224 token interrotto a 128, con stato e 32 passi successivi identici alla propria esecuzione diretta. [Test agente](agent.json) · [Log interruzione e ripresa](agent-pressure-v3.log)

Il candidato resta nel diagnostico. Il test rispetto a llama.cpp supera le soglie in entrambe le modalità e le verifiche dell'agente qui coprono il test dense e la pressione sul contesto. I diff e il generatore del diagnostico sono nell'archivio. [Generatore](prepare_partial.py) · [Dispatch host](formats-host.diff) · [Kernel IQ3_S e IQ4_XS](formats-moe.diff) · [Altri kernel](formats-linear.diff)
