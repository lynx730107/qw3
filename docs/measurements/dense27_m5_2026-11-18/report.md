# Qwen3.8-27B denso su Apple M5, 18–19 novembre 2026

GGUF `Qwen3.8-27B-UD-Q3_K_XL`, KV F16, 64 layer di qw3 su GPU, quattro thread. I sorgenti del runtime nella checkout corrispondono agli hash dell’esperimento del 17 novembre. Le varianti sono state costruite in una directory temporanea. I processi di inferenza sono stati eseguiti uno alla volta. [Identità dei sorgenti e delle librerie](identity.json) · [Build diagnostiche](diagnostic-hashes.json)

## Prefill parziale

Il candidato estende da 96 a 64 token il limite inferiore della variante 3 del 17 novembre. I contatori a 64 e 95 token confermano l’attivazione delle proiezioni IQ4_XS e IQ3_S staged. A 63 token entrambe le modalità usano le proiezioni dirette. La modalità 0 riproduce il percorso precedente nello stesso eseguibile. [Preparazione](prepare.py) · [Dispatch a 64 token](tail-route-64-v3.log)

### Code isolate

Ogni processo riscalda entrambe le modalità e misura quattro coppie alternate nella stessa sessione. Il timer comprende il prefill e la scelta del primo token greedy. Le medie comprendono tutte le coppie. Un delta positivo indica una riduzione della latenza. [Runner](turn.m) · [Tempi e differenze per coppia](analysis.json)

| Token | Baseline, ms | Candidato, ms | Riduzione | Coppie favorevoli |
|---:|---:|---:|---:|---:|
| 63 | 768,8 | 753,4 | 2,0% | 2/4 |
| 64 | 770,0 | 677,3 | 12,0% | 3/4 |
| 65 | 1.269,6 | 921,6 | 27,4% | 4/4 |
| 69 | 1.276,6 | 854,2 | 33,1% | 4/4 |
| 71 | 1.263,3 | 822,2 | 34,9% | 4/4 |
| 87 | 1.285,9 | 862,4 | 32,9% | 4/4 |
| 95 | 1.302,6 | 868,8 | 33,3% | 4/4 |
| 96 | 1.304,4 | 896,3 | 31,3% | 4/4 |
| 128 | 1.065,6 | 1.146,8 | -7,6% | 1/4 |

I controlli da 63 e 128 token instradano le due modalità sugli stessi kernel. I loro scarti danno la scala della variabilità della sessione. A 64 token la direzione si inverte in una coppia. Per tutte e cinque le code da 65 a 95 token il candidato migliora in ogni coppia.

### Turni con prefisso in cache

Abbiamo riprodotto le cinque estensioni della traccia agente che terminano con una coda da 64 a 95 token. Lunghezze e posizioni corrispondono alla traccia. I token sono sintetici, con seed 1. Prima di ogni campione viene ripristinato lo stesso checkpoint. Il timer parte dopo il ripristino e termina alla scelta greedy. Tokenizzazione, interfaccia, tool e caricamento del modello sono fuori dal timer. In ogni caso il primo token è identico fra le otto prove cronometrate. [Traccia originale](../agent_context_pressure_2026-10-11/gadda_tail_v7/gadda.trace.jsonl)

| Token in cache | Token aggiunti | Coda | Baseline, ms | Candidato, ms | Risparmio, ms | Riduzione |
|---:|---:|---:|---:|---:|---:|---:|
| 2404 | 727 | 87 | 7.599,4 | 6.908,6 | 690,8 | 9,1% |
| 3250 | 709 | 69 | 7.441,4 | 6.952,4 | 489,0 | 6,6% |
| 4101 | 705 | 65 | 7.588,7 | 7.143,0 | 445,7 | 5,9% |
| 3976 | 991 | 95 | 10.381,2 | 9.890,8 | 490,4 | 4,7% |
| 5230 | 967 | 71 | 10.870,9 | 10.212,2 | 658,6 | 6,1% |

### Correttezza

Su 12 lunghezze di prefisso da 63 a 223 token, il controllo comprende otto passi successivi per prefisso. Sono stati verificati altri 32 passi sia a 65 sia a 95 token. Complessivamente 174 confronti di vocabolari completi, con zero differenze di bit. Stato canonico e roundtrip dei checkpoint coincidono. Le 12 lunghezze sono state eseguite con Metal API Validation. [Suite dei prefissi](tail-quality.log) · [Continuazione a 65](tail-checkpoint-65.log) · [Continuazione a 95](tail-checkpoint-95.log)

### Origine dello scarto sul prefisso da 127

Abbiamo letto il vettore residuo di 5.120 componenti dell’ultimo token dopo l’attenzione e dopo la FFN di ciascuno dei 64 strati, oltre all’embedding. Abbiamo ripetuto la lettura su qw3 e llama.cpp in seriale e in batch. L’embedding coincide fra tutte le esecuzioni. Dopo l’attenzione dello strato 0 la RMSE fra i due motori in batch è 0,00000334, dopo l’attenzione dello strato 1 è 0,000113. All’uscita dello strato 63 la RMSE assoluta sale a 0,02015, con errore RMS relativo dello 0,134%. La crescita assoluta finale accompagna la crescita di scala del residuo. Le letture ai confini dei blocchi localizzano la divergenza senza assegnarla a una singola istruzione o proiezione.

Con il readback attivo, la RMSE dei logits finali riproduce il confronto ordinario sullo stesso prefisso. In seriale è 0,00008372 con massimo 0,0004573. In batch è 0,0022676 con massimo 0,0099885. Le soglie di progetto sono rispettivamente 0,001 e 0,005. La variante del 17 novembre aveva già riprodotto esattamente il controllo qw3 su questo prefisso. [Tutti gli strati](layers.json) · [Confronto seriale](layers-serial.log) · [Confronto in batch](layers-batch.log) · [Accettazione del progetto](../../optimization_acceptance.md)

Il 19 novembre abbiamo ripetuto il confronto esterno anche su quattro lunghezze più corte. Il controllo qw3 e il candidato hanno ottenuto gli stessi errori. In ogni esecuzione coincidono gli otto elementi del ranking e gli otto token greedy. Tutte e quattro le lunghezze superano le soglie sull’intero vocabolario. La selezione pubblica resta quindi quella precedente. [Otto confronti con llama.cpp](tail-frontier.json)

| Token | RMSE | Errore massimo |
|---:|---:|---:|
| 64 | 0,001885 | 0,008746 |
| 65 | 0,003827 | 0,016378 |
| 87 | 0,002058 | 0,009509 |
| 95 | 0,003901 | 0,019975 |

## Generazione ordinaria del 27B

Due coppie di processi in ordine inverso per profondità, 32 forward sintetici ciascuno. Il prefill è escluso dal timer. Entrambi i motori usano KV F16, batch di prefill da 128 e pieno offload. I generatori dei token sintetici differiscono. Il benchmark ordinario qw3 prepara l’intera profondità due volte. Il benchmark locale llama.cpp, build 11960, riscalda un token a contesto vuoto prima di preparare la profondità. [Comandi e campioni](depth.json) · [Modifica del solo warmup per la replica](warmup.diff)

| Profondità | qw3, tok/s nelle due coppie | llama.cpp, tok/s nelle due coppie |
|---:|---:|---:|
| 1.024 | 9,32 / 9,33 | 9,11 / 9,32 |
| 4.096 | 8,19 / 7,14 | 8,22 / 8,18 |
| 8.192 | 6,25 / 5,72 | 6,10 / 5,94 |

### Costi GPU sotto strumentazione

Media di due passi per profondità. I contatori richiedono encoder separati, quindi la tabella serve a ordinare i costi. Le prove di throughput sopra usano i confini ordinari degli encoder. A ogni profondità 16 vocabolari completi coincidono bit per bit tra l’esecuzione ordinaria e quella strumentata, riusando una sola sessione con checkpoint su disco. A 8K i contatori rilevano la FlashAttention raggruppata e la riduzione dei risultati parziali. [Risultati per singolo kernel e passo](analysis.json)

| Profondità | IQ4_XS, ms/passo | IQ3_S, ms/passo | GQA/Flash, ms/passo |
|---:|---:|---:|---:|
| 1.024 | 50,94 | 41,16 | 1,92 |
| 4.096 | 50,54 | 40,66 | 5,34 |
| 8.192 | 54,77 | 46,57 | 7,03 |

Il confronto tra due e tre teste GQA ha riprodotto esattamente i 33 vocabolari esaminati e lo stato canonico. IQ3_S nibble ha superato il criterio numerico sulle 192 matrici e sul modello. Si tratta di test su una sessione. Le campagne non misurano il picco di RAM né l’affidabilità su tutti i compiti dell’agente. [GQA sul modello](gqa-model.log) · [Matrici IQ3_S](nibble-matrix.log) · [IQ3_S sul modello](nibble-model.log)

### Replica delle code e dei due candidati decode, 19 novembre

Su 24 campioni raccolti in otto processi indipendenti con ordine invertito, il prefill a 65 token scende da 912,49 a 608,52 ms (−33,31%), a 95 token da 968,19 a 661,30 ms (−31,70%). Entrambe le coppie per ciascuna lunghezza sono favorevoli. Nei cinque turni sintetici con cache, tutte le 20 coppie favoriscono il candidato. [Replica indipendente del prefill](tail-replica.json)

Per il decode, la prima serie usa quattro coppie di processi con due ripetizioni ciascuno. La replica usa sei coppie nella stessa sessione, con riordino, lo stesso prefisso e gli stessi 32 token. Per GQA a due teste la latenza aumenta del 6,37% nella prima serie e diminuisce dello 0,39% nella replica. Per IQ3_S nibble la latenza diminuisce dello 0,78% nella prima serie e del 2,08% nella replica. Cinque coppie su sei favoriscono nibble nella replica, con variazioni da un peggioramento dello 0,83% a un miglioramento del 6,64%. Nibble merita una verifica più ampia, inclusi confronto esterno, checkpoint e comportamento dell’agente. [Processi indipendenti](analysis.json) · [Replica nella stessa sessione](paired-results.json)

### Confronto con riscaldamento uniforme

Un token di warmup a contesto vuoto e una sola preparazione del prefisso per motore. A 4K la serie è del 18 novembre. Dopo l’interruzione durante il primo processo a 8K abbiamo ripetuto entrambe le coppie a 8K il 19 novembre. Il log incompleto del primo tentativo è conservato.

| Profondità | qw3, tok/s nelle due coppie | llama.cpp, tok/s nelle due coppie |
|---:|---:|---:|
| 4096 | 7,18 / 6,16 | 6,32 / 6,05 |
| 8192 | 8,03 / 7,52 | 7,80 / 8,00 |

A 8K la classifica cambia con l’ordine, anche uniformando il warmup. Le due coppie a 4K favoriscono qw3 in questa serie. Il numero di coppie e la variabilità limitano la precisione e la generalizzazione. [Replica 4K](matched/depth.json) · [Replica 8K](matched8/depth.json)
