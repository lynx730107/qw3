# Proiezioni concorrenti: diagnostico con coppie ravvicinate

La serie entro un unico processo dà un punto +0,753%, tutte le sei coppie
favorevoli. È un segnale da replicare, non una prova di adozione o del
raggiungimento di llama. Il precedente +2,757% con controlli instabili
rimane [inconcludente](m5_decode_projection_barriers_2026-10-14.md).

Si riusano i tre helper congelati del gate modello bit-esatto delle
proiezioni FFN/DeltaNet. I soli hook nel benchmark impostano la modalità
prima del reset/prefill e verificano i contatori dopo il timer. Rimuovendoli
si ritrova qw3_bench.c originale byte per byte. Restano un unico engine e
una sola sessione ctx256, il warmup originale in modalità baseline,
il flusso seed1 e lo stesso reset/prefill/timed loop. Nessun campionamento
o MTP, nuovi pesi o buffer dati di sessione. I piccoli array CPU dei tempi
sono quelli originali dimensionati per 12 ripetizioni; la RAM totale non
è misurata.

Ordine 0/1/1/0 ripetuto tre volte, 12 campioni tg32 depth128, KV F16,
64 layer/head GPU; 60 secondi prima del processo, nessuna pausa tra le
ripetizioni. Prefill fuori dal timer, API Validation disattiva per i tempi.
Ogni ON conta 2528 finestre (992 FFN, 1536 DeltaNet), 31296 dispatch,
25696 barriere aggiunte e 5600 salti; OFF zero interventi. Il contatore
richiede esattamente 12 ripetizioni, tutti i campioni sono conservati.

| Coppia | Ordine | Variazione throughput |
| --- | --- | ---: |
| 1 | OFF/ON | +0,304% |
| 2 | ON/OFF | +0,770% |
| 3 | OFF/ON | +1,549% |
| 4 | ON/OFF | +0,638% |
| 5 | OFF/ON | +0,330% |
| 6 | ON/OFF | +0,916% |

OFF medio 3595,668 ms, ON 3568,810 ms. Risparmio medio 26,858 ms,
intervallo Student t al 95% [9,469; 44,247] ms, df5 e sei coppie trattate
come indipendenti. La correlazione temporale entro lo stesso processo
può indebolire questa assunzione: non si confonde l'intervallo con una
replica tra processi indipendenti. Medie per ordine 26,113/27,603 ms
risparmiati, entrambe positive. Spread max−min/media 2,698% OFF e 1,545% ON.

CSV 176.761.680 byte di sessione; sorgenti, snapshot, binario, stat del
GGUF e binari di produzione invariati. Gli helper GPU sono quelli del
gate modello precedente. I prefissi estesi/checkpoint, le regressioni
dell'agente e il confronto finale con llama rimangono da eseguire prima
dell'adozione. La produzione resta invariata.

[Manifest](measurements/projection_pairs_2026-10-14/manifest.json),
[analisi di tutti i campioni](measurements/projection_pairs_2026-10-14/analysis.json)
e [verifica indipendente](measurements/projection_pairs_2026-10-14/verification.json).
Non si archiviano binari, GGUF o checkpoint e non si aggregano le serie.
