# Runtime Metal del 27B denso, 19 novembre 2026

Sul base M5 i due esperimenti si attivano separatamente all’avvio con `QW3_METAL_DENSE_PARTIAL_TILES=1` e `QW3_METAL_IQ3_NIBBLE=1`. Il primo estende le proiezioni staged a blocchi di 64–127 token. Il secondo prepara quattro magnitudini dispari per parola della tabella IQ3_S usata dal matvec singolo. La dimensione della tabella resta di 1.024 byte. La configurazione entra nel fingerprint dei checkpoint.

Il benchmark pubblico sullo stesso GGUF, KV F16, 64 layer Metal, pp65, ctx256 e quattro thread ha misurato 914,79 e 916,46 ms per il controllo e 613,52 e 612,83 ms per il candidato. Ogni valore è la media di tre ripetizioni dopo il warmup. Ordine OFF/ON/ON/OFF. La riduzione media di latenza è del 33,0%. [Controllo](prefill-0.log) · [Candidato](prefill-1.log)

## Primo strato DeltaNet

Sul prefisso di 127 token abbiamo mantenuto il riferimento llama.cpp in batch e variato soltanto il primo strato di qw3. Entrambe le opzioni sperimentali erano attive. Il primo esperimento conserva le proiezioni QKV e gate in batch ed esegue la parte restante dell’attenzione token per token. Il secondo esegue anche quelle proiezioni token per token. Le FFN mantengono il percorso ordinario. [Diff diagnostico](layer0.diff)

| Esecuzione | RMSE logits | Errore massimo |
|---|---:|---:|
| Runtime integrato | 0,002268 | 0,009989 |
| Primo strato con proiezioni QKV e gate in batch | 0,002060 | 0,010042 |
| Attenzione del primo strato token per token | 0,001983 | 0,010541 |

Le tre esecuzioni concordano sugli otto token greedy e sui primi otto elementi del ranking. L’errore massimo supera la soglia di 0,005 in tutti i casi. A 65 token, il solo cambiamento del prefill riproduce esattamente le metriche precedenti rispetto a llama.cpp. Le due modifiche al primo strato restano nell’esperimento. [Controllo 127](runtime-127-1-1-0.log) · [Proiezioni batch](ablation-127-1-1-1.log) · [Attenzione seriale](ablation-127-1-1-2.log)

Sul runtime integrato passano il test agente dense con checkpoint e cache del prefisso e il compito che crea, compila ed esegue un programma C, con entrambe le opzioni attive. Con IQ3_S nibble attivo i due confronti seriali standard con llama.cpp rientrano nelle tolleranze. [Agente dense](agent.log) · [Compito C](agent-coding.log) · [Logits seriali](decode-logits.log) · [Test sui confini dei tile](boundaries.log)
