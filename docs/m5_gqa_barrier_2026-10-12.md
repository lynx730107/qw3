# GQA: due barriere con lo scratch originale

Esperimento privato sul kernel `qw3_gqa_attend_n_inner`. Le somme parziali
occupano lo scratch di threadgroup iniziale; i dot finali vengono spostati
negli elementi 56–62 già allocati. Le scritture successive delle somme
parziali non li sovrascrivono: si può togliere la terza barriera del ciclo.
Restano due barriere per token di contesto. La prima variante mantiene
i cinque cicli su otto head; la seconda limita quei cicli agli head attivi.
Formule FP32, ordine degli accumuli e delle riduzioni rimangono originali.
Il verificatore ripristina le sole modifiche e ritrova il kernel originale
byte per byte. Nessuna ispezione ISA viene usata per attribuire trasformazioni
fisiche al compilatore.

Il setter privato seleziona il candidato per gruppi da uno a sette head;
il gruppo di otto mantiene il kernel originale. Il modello usa sei head
per gruppo e 256 thread. Le pipeline dichiarano SIMD width 32, massimo
1.024 thread e zero byte statici di threadgroup; lo scratch dinamico resta
256 byte. Non si aggiungono copie dei pesi o buffer di dati di sessione.
Gli oggetti di pipeline e i temporanei CPU della compilazione rimangono:
non è stata misurata l'identità della RAM totale o del suo picco.

## Correttezza

Build senza warning, API Validation attiva durante i controlli numerici.
Passano 288 fixture per variante, **576 totali**, senza caricare i pesi:
gruppi 1/6/7/8, head dimension 32/128/256, contesti 1/2/31/128,
KV F32/F16, dati misti, zeri con segno e scale ampie/piccole. Output finiti
bit per bit, input invariati, offset non zero, guardie e un threadgroup
oltre il limite. Ogni variante registra 216 selezioni e 72 fallback.
Le fixture usano piccoli buffer temporanei, riutilizzati tra baseline
e candidato, e non una seconda sessione del modello.

Sul GGUF 27B originale passano baseline e due varianti con una sola
sessione ctx256 riutilizzata e un oracolo su disco. Prefisso sintetico
128, seed 1, seguito greedy 32, KV F16, offload completo. Coincidono
33 vocabolari completi finiti per fase, token, history, posizioni,
GQA K/V vivi, DeltaNet e convoluzione prima e dopo il seguito.
Il tail temporaneo Flash Attention è escluso. Ogni candidato seleziona
512 dispatch nel seguito, zero fallback; la baseline zero.

Il verificatore inizialmente cercava un'etichetta `gqa_barriers` plurale:
il log reale usa `gqa_barrier`. Corretta soltanto l'asserzione del
verificatore; output numerici e binari immutati. La verifica iniziale
passa per 53 riferimenti congelati, 576 fixture, tre fasi del modello
e 15 campioni ordinari. La correzione è documentata nell'archivio.

## Misure

Decode ordinario tg32 a profondità 128, ctx256, KV F16, tre ripetizioni
con warmup e seed 1. Processi sequenziali, pause di 60 secondi,
`caffeinate` anche sul controller. Compilazione e prefill fuori dal timer,
API Validation disabilitata per i tempi. Packed8, K64 e GDN cache64 attivi.

| Screening | OFF | Variante 1 | OFF | Variante 2 | OFF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Media tg32, ms | 3674,083 | 3564,561 | 3755,091 | 3537,573 | 3509,917 |

Gli OFF variano del **6,724%**, oltre il limite del 5%: screening
inconcludente. La variante 1 misura +4,209% rispetto ai controlli
interpolati (+3,073% / +5,345% contro i singoli); la variante 2 +2,684%
ma +6,149% / −0,782% contro i singoli. Si replica soltanto la variante 1.
Nessun campione iniziale viene eliminato.

| Replica variante 1 | OFF | ON | OFF | ON | OFF | ON | OFF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Media tg32, ms | 3497,158 | 3492,910 | 3496,644 | 3490,629 | 3500,805 | 3495,304 | 3500,239 |

Dispersione OFF 0,119%, ON 0,134%. Media ON 3492,947 ms contro
3498,716 ms dei controlli interpolati: **+0,165%**. I confronti locali
sono +0,114%, +0,232%, +0,149%; tutti battono entrambi i controlli vicini.
È un piccolo segnale favorevole; il +4,209% iniziale non è replicato.
Le differenze delle medie sono inferiori all'oscillazione dei singoli
campioni dentro ciascun processo. Non si costruisce una CI che assuma
controlli indipendenti: i vicini sono condivisi. Si conserva il candidato
privato per eventuali prove successive, senza integrarlo in produzione.

Restano tutti i **36 campioni** (15 iniziali e 21 della replica).
Ogni ON registra 1552 selezioni, zero fallback; ogni OFF zero.
Tutte le righe CSV riportano 176.761.680 byte di sessione.
Lo swap letto prima e dopo i processi resta 812,31 MiB; non misura il picco.
Lo store personale resta identico al preflight numerico, senza pulizie
o scritture del contenuto. Sorgenti ed eseguibili di produzione immutati.

Non viene misurato pp128 né effettuato un nuovo confronto con llama.
Gli otto prefissi/checkpoint e le regressioni dell'agente rimangono
necessari prima di un'eventuale integrazione. L'obiettivo resta aperto:
l'ultimo confronto diretto richiedeva
[circa +2,962% nel decoding](m5_native_reference_after_store_cleanup_2026-10-11.md).

## Archivio e riproduzione

[Manifest numerico](measurements/gqa_barrier_2026-10-12/manifest.json),
[screening](measurements/gqa_barrier_2026-10-12/native/manifest.json),
[replica](measurements/gqa_barrier_2026-10-12/native_retry/manifest.json),
analisi, log e controller sono conservati. Il verificatore controlla
53 riferimenti congelati iniziali, 54 della replica, tutti i campioni,
contatori e buffer riportati dai log. Gli snapshot riutilizzabili fanno
riferimento agli archivi già committati. Il generatore ricrea i quattro
file privati byte per byte dopo averli rimossi. Il dry-run forzato Linux
conferma il rifiuto dei tre target Metal. Nessun binario, GGUF o checkpoint
viene archiviato.

Build: `make qw3-gqa-barrier-test qw3-gqa-barrier-model-test qw3-gqa-barrier-native-bench`.
Fixture: `./qw3-gqa-barrier-test 1` e `2`, senza caricare il modello.
Modello: `./qw3-gqa-barrier-model-test MODEL.gguf`, usando l'ambiente
nel manifest. Benchmark privato: `QW3_PRIVATE_GQA_BARRIER=0|1|2`;
produzione non legge questa variabile. Le velocità assolute di campagne
diverse non costituiscono un confronto con llama.
