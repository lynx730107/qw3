# Replica IQ3_S constant: vantaggio non confermato

La replica ordinaria scarta la lettura diretta `constant` della tabella
IQ3_S originale. Il +1,507% osservato nel [primo screening](m5_iq3_grid_2026-10-11.md)
non si riproduce: entrambi i blocchi ON perdono rispetto alla media dei
controlli adiacenti. Nessuna variante viene adottata.

La prova è stata eseguita dopo le [regressioni dell'agente e del web](agent_regressions_2026-10-11.md),
con lo stesso sorgente e binario privato già verificati. Cinque processi
OFF/constant/OFF/constant/OFF, tre ripetizioni ciascuno e 60 secondi prima
di ogni processo: tg32 a profondità 128, ctx256, KV F16, tutti i 64 layer
e head GPU, warmup originale, stream fisso seed 1. Prefill e compilazione
restano fuori dal timer; la validazione API è disabilitata per il tempo.
I default packed8, K64 prefill e cache64 DeltaNet restano attivi.

| Blocco | Modalità | Media tg32, ms | Variazione rispetto alla media degli OFF adiacenti |
| --- | --- | ---: | ---: |
| 0 | OFF | 3517,315 | — |
| 1 | constant | 3516,562 | −0,198% |
| 2 | OFF | 3501,901 | — |
| 3 | constant | 3606,937 | −2,798% |
| 4 | OFF | 3510,143 | — |

Il primo ON è +0,021% rispetto al controllo precedente e −0,417% rispetto
al successivo. Il secondo è −2,912% / −2,684%. Il rapporto fra il tempo
medio dei riferimenti adiacenti, 3507,815 ms, e quello dei candidati,
3561,750 ms, indica −1,514% di throughput. È uno screening di due blocchi
candidati, senza una stima di equivalenza o un intervallo di confidenza.
Non si scarta il secondo blocco e non se ne attribuisce la causa a carichi
esterni non misurati.

Le medie OFF hanno dispersione max−min/media dello 0,439%; le medie ON
del 2,537%. Sono conservati tutti i 15 campioni. Ciascun ON registra
7.081 dispatch reali del candidato (97 forward × 73 matvec singoli);
i controlli ne registrano zero. Il matvec IQ3_S paired resta nativo.
Lo swap è 1301,06 MiB prima e dopo ciascuno dei cinque processi.

La variante usa i buffer originali: zero copie dei pesi e zero nuovi
buffer di dati della sessione. Le verifiche numeriche su matrici e modello
restano quelle del primo screening; questa replica non esegue gli otto
prefissi o roundtrip dei checkpoint, perché non emerge un candidato da
adottare. Il runtime e il binario ordinario di produzione restano invariati.
Le velocità assolute non aggiornano il confronto diretto con llama.cpp.

Il [manifest](measurements/iq3_grid_2026-10-11/native_retry/manifest.json)
e l'[analisi](measurements/iq3_grid_2026-10-11/native_retry/analysis.json)
conservano comandi, ambiente, campioni, contatori e osservazioni di swap.
I percorsi dei sorgenti riusano gli snapshot già committati, con hash
verificati; binari e stat del GGUF coincidono con quelli della prova.
