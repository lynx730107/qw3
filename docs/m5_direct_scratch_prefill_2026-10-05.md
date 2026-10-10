# Proiezioni e output di attenzione direttamente nello scratch — M5

Prova del 5 ottobre 2026, sul branch `codex/m5-deltanet-prefill`, successiva
alla campagna sulle [proiezioni quantizzate](m5_projection_prefill_2026-10-04.md).
Il prototipo è privato ai tre eseguibili diagnostici; non modifica il motore
ordinario, gli shader, il GGUF, il campionamento o il percorso dell'agente.

## Ipotesi e implementazione

Il prefill denso calcola Q/K/V di GQA oppure QKV/Z di DeltaNet in buffer
compatti, poi copia ciascuna matrice nello scratch interlacciato per token.
Dopo l'attenzione copia il risultato dallo scratch a `prefillX1` per la
proiezione di output. I kernel delle proiezioni supportano già offset e stride.
La prova usa questi parametri per scrivere le prime proiezioni direttamente
nello scratch e leggere la proiezione di output direttamente dallo scratch.

Su un batch completo del modello a 64 layer, 48 layer DeltaNet e 16 GQA,
si possono eliminare 144 copie delle proiezioni e 64 compattazioni dell'output.
Sono **208 operazioni di copia**, non 208 command buffer o attese CPU.
Le copie del baseline usano già il kernel GPU introdotto nella campagna
[precedente](m5_prefill_copy_2026-10-02.md).

`tests/scratch_candidates.inc` intercetta soltanto i sei punti del prefill
interessati. Il layout viene passato dalle variabili del core che già lo
calcolano, evitando di duplicare il calcolo di dimensioni e offset. Il
dispatch privato in `tests/scratch_matmul.inc` riproduce quello ordinario,
con controlli dei limiti per i buffer strided e i quattro campi di layout
modificati. Una verifica di normalizzazione del sorgente conferma che il
resto del dispatch coincide esattamente con l'originale. Non vengono
compilati nuovi shader: entrambe le configurazioni usano le stesse pipeline,
geometria, dequantizzazione, prodotti, accumuli e precisioni.

La proiezione di output conserva il buffer compatto `denseBatchDown` e il
kernel di somma del residuo. Non si fondono residuo o normalizzazioni.
Le allocazioni temporanee richieste dal percorso ordinario possono ancora
esistere: questo esperimento misura la rimozione delle copie, non una nuova
politica di memoria. Batch parziali di attenzione e chunk con meno di
16 token mantengono il percorso ordinario. Il decode a un token è invariato.

La selezione privata `QW3_PRIVATE_DIRECT_SCRATCH` accetta `projections`,
`output` e `all`; la campagna completa misura `all`. I contatori verificano
il numero di proiezioni modificate, proiezioni di output modificate e copie
saltate. Il totale delle copie saltate deve essere uguale alla somma dei
due contatori di dispatch. OFF deve avere tutti i contatori a zero.

## Correttezza

Il [manifest di qualità](measurements/direct_scratch_quality_2026-10-05/manifest.json)
conserva hash del motore e degli shader, degli eseguibili, dei diagnostici,
comandi e swap. Gli snapshot dei diagnostici sono accanto ai log.

Il controllo completo `all` confronta **tutto il vocabolario** con uguaglianza
numerica esatta e finitezza dopo il prefill, dopo il ripristino del payload e
ad ogni passo dei 32 token greedy di continuazione. Confronta inoltre la
storia dei token e tutti i byte dei KV F16 vivi, dello stato DeltaNet e della
convoluzione. Esclude la capacità KV inutilizzata e gli scratch derivati di
FlashAttention, come le campagne precedenti.

Prefissi: 1, 15, 16, 31, 127, 128, 129 e 159 token. Il caso 128 viene esteso
a 159 prima della continuazione, verificando un nuovo chunk con stato
ricorrente non nullo. 129 e 159 verificano anche il confine del batch da 128.
Tutti gli otto casi passano. Il log del caso 128 riporta la lunghezza finale
159, così come il distinto caso con prefisso iniziale 159.
Il percorso candidato è stato eseguito: 1152 proiezioni, 512 output e
1664 copie saltate. [Log completo](measurements/direct_scratch_quality_2026-10-05/all.stdout).

Il controllo isolato delle matrici copre tutte le **24 combinazioni reali di
formato e dimensioni** delle proiezioni di attenzione. Per ciascuna confronta
16, 31 e 128 token con 1, 33 e il numero reale delle righe di output:
**216 casi**. Le righe 1 e 33 reinterpretano un prefisso limitato del tensore
reale per verificare le code; non sono tutte forme effettive del modello.
Gli input sintetici mantengono offset non nulli e stride con padding dispari.
Tutti gli output sono finiti e identici bit per bit al kernel compatto;
i gap di output e il buffer di input restano invariati. Layout invalidi e
un'ultima riga fuori dall'allocazione sono rifiutati prima di codificare
lavoro GPU. [Log matrici](measurements/direct_scratch_quality_2026-10-05/matrices.stdout).

## Metodo di velocità

Stesso `Qwen3.8-27B-UD-Q3_K_XL.gguf`, 64 layer Metal, KV F16, contesto
allocato 512, quattro thread ausiliari e ID sintetici seed 1.
pp128 puro e tg32 a profondità 128 sono processi separati. tg32 comprende
32 forward ordinari con logits; non usa campionamento o speculazione.
Warmup, compilazione e preparazione della profondità sono esclusi.

Ordine OFF / candidato / OFF / candidato / OFF, tre ripetizioni per fase e
40 secondi di pausa prima di ogni blocco. Nessun profiler intrusivo o
contatore GPU è attivo. Lo script registra comandi, hash, contatori privati,
campioni, swap prima/dopo e confronti con i controlli immediatamente vicini.
Un vecchio processo `qw3-test` è fermo nel debugger; nessun altro processo
di inferenza attivo è stato osservato. Il processo fermo non è stato terminato.

Criterio applicato prima della misura: almeno +5% pp128 ripetibile,
nessuna regressione tg32 ripetibile oltre il 2%, dispersione dei controlli
OFF entro il 5% per entrambe le fasi. Una variazione tg32 non viene
attribuita a questo percorso, usato soltanto nel prefill fuori dal suo timer.

## Risultati

La serie di misura è registrata in
[results.json](measurements/direct_scratch_all_2026-10-05/results.json),
con log e snapshot dei sorgenti diagnostici.

| Misura | OFF, token/s | Variante, token/s | Variazione osservata | Dispersione OFF |
|---|---:|---:|---:|---:|
| pp128 | 153,065 | 155,380 | +1,513% | 1,328% |
| tg32 a profondità 128 | 8,790 | 8,858 | +0,782% | 2,160% |

I due confronti pp128 con i controlli immediatamente adiacenti danno
**+1,978%** e **+1,095%**. Il segno è positivo in entrambi, ma l'entità
resta sotto il criterio del 5%. Il baseline accelera durante la serie:
152,15 → 152,95 → 154,16 token/s. Per questo si riportano anche i confronti
adiacenti e non si usa soltanto il primo controllo. Non viene costruita
una stima di confidenza trattando i token come campioni indipendenti.

I controlli OFF restano entro il limite di dispersione in entrambe le fasi;
non emerge una regressione tg32 oltre il 2%. La piccola variazione tg32
non dimostra un guadagno del decode: le 32 proiezioni per-token cronometrate
sono quelle originali. Il percorso modificato lavora soltanto nella
preparazione della profondità, esclusa dal timer.

In ogni processo candidato i contatori riportano 576 proiezioni,
256 output e 832 copie saltate, corrispondenti a un warmup e tre batch
da 128 token. Tutti i processi OFF riportano zero. Lo swap dell'intera
macchina era già presente e passa da circa 2718 a 2670 MiB durante la
serie; non si afferma che il modello fosse tutto residente, né si attribuisce
la variazione a una singola sessione.

## Esito

**La variante non viene adottata.** Eliminare queste copie mostra un piccolo
segnale positivo nella prova breve, ma non soddisfa il criterio di adozione.
Si conserva il prototipo riproducibile; CLI, agente e benchmark ordinario
continuano a usare il percorso precedente. Non è stato misurato un vantaggio
a prompt lunghi, né sono state misurate separatamente le selezioni
`projections` e `output`.

I tre eseguibili diagnostici compilano senza warning nella build finale.
La verifica dei sorgenti conferma che `qw3.c`, `qw3_metal.m`,
`qw3_bench.c` e tutti gli shader di produzione sono byte per byte identici
all'inizio della prova. Nessuna modifica di inferenza ordinaria viene
adottata, quindi non si rilanciano senza motivo le regressioni dell'agente
già valide per quel percorso. Questa prova non esaurisce la diversa
ipotesi di dequantizzazione direttamente nei registri di TensorOps.

## Riproduzione

Dal root del repository:

```sh
make qw3-scratch-prefill-bench qw3-scratch-prefill-test qw3-scratch-matmul-test
QW3_METAL_KV_F16=1 QW3_METAL_KV_Q8_0=0 caffeinate -disu \
  ./qw3-scratch-prefill-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf all
QW3_METAL_KV_F16=1 QW3_METAL_KV_Q8_0=0 caffeinate -disu \
  ./qw3-scratch-matmul-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf
python3 tools/bench_qw3_candidates.py --output /tmp/qw3-direct-scratch \
  --binary ./qw3-scratch-prefill-bench \
  --order off,candidate,off,candidate,off \
  --candidate-env QW3_PRIVATE_DIRECT_SCRATCH=all --repetitions 3 --cooldown 40
```

La directory di output deve essere nuova. Gli eseguibili opzionali non sono
dipendenze di CLI, agente o benchmark ordinario. Non eseguire altre
inferenze contemporaneamente. Nessuna misura di questa campagna dimostra
parità con llama.cpp, un limite termico sostenuto o un vantaggio a contesti
lunghi.
