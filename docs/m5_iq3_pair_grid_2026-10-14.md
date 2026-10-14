# IQ3_S gate/up: griglia packed su M5

Entrambe le varianti private sono esatte nelle prove eseguite, ma perdono
contro ciascun controllo adiacente nello screening ordinario. Nessuna è
adottata; il runtime mantiene Q4_K uint verificato. Il
[confronto diretto](m5_q4_uint_reference_2026-10-13.md) richiede ancora
+2,383% nel decoding per raggiungere llama.cpp.

Il matvec paired originale prepara `float4[512]` in threadgroup, 8192 byte.
La prima variante conserva la tabella packed `ushort[512]`, 1024 byte,
espandendo gli stessi quattro valori float al momento della lookup. La
seconda legge direttamente la tabella packed device esistente, senza
staging o barriera iniziale. Restano espressioni `dot` FP32, prodotti,
accumuli, traversal K, otto righe/64 thread e riduzioni SIMD originali.
Non si copiano i pesi e non si aggiungono buffer di dati della sessione.
Il fixture passa128 byte dinamici; la route del modello mantiene256 byte,
64×sizeof(float), identici per baseline e candidati. Le pipeline compilate
riportano statico8192/1024/0 byte, SIMD32 e massimo1024 thread.

## Numerica

I tre runner compilano senza warning. Metal API Validation è attiva nei
controlli numerici. Passano96 casi per variante,192 totali: otto forme,
tre pattern (misto, zeri con segno, ampiezze diverse), input offset16/32
e padding riga0/2 byte. Due pesi sintetici distinti gate/up, offset2 byte,
sono limitati complessivamente a circa1 MB. Ogni caso confronta tutti
gli output finiti byte per byte, input e pesi immutati, guardie e un
threadgroup extra oltre il limite. Non tutte le forme sono matrici
effettive del modello.

Sul 27B si riusa un solo engine e una sola sessione ctx256, con oracolo
temporaneo su disco: baseline, packed threadgroup, device diretto.
Prefisso128 seed1 e seguito greedy32, KV F16 e64 layer/head GPU. Coincidono
33 vocabolari completi finiti per fase, token, history, posizione e stato
canonico GQA K/V, DeltaNet e convoluzione prima e dopo il seguito.
Il tail temporaneo Flash Attention è escluso. Ogni candidato registra608
selezioni nel seguito (32×19), baseline zero. Non vengono eseguiti otto
prefissi e checkpoint, perché lo screening non giustifica integrazione.

## Tempo ordinario

Cinque processi OFF/packed/OFF/device/OFF, tg32 depth128, ctx256,
KV F16, offload completo, tre ripetizioni, warmup e60 secondi prima di
ogni processo. Compilazione e prefill sono fuori dal timer; API Validation
è disabilitata durante i tempi. Tutti i15 campioni sono conservati.

| Variante | OFF vicini interpolati, ms | ON, ms | Variazione throughput |
| --- | ---: | ---: | ---: |
| Packed threadgroup | 3458,709 | 3471,491 | −0,368% |
| Device diretto | 3460,200 | 3469,470 | −0,267% |

La prima perde −0,424%/−0,312% sui singoli vicini, la seconda
−0,254%/−0,280%. Medie dei cinque blocchi:
3456,763 /3471,491 /3460,656 /3469,470 /3459,745 ms.
Spread max/min degli OFF0,113%, sotto la soglia5%. Un singolo blocco
per variante non stabilisce un rallentamento universale; non emerge
un beneficio da replicare o adottare. Non si calcola un intervallo
paired indipendente dai controlli condivisi, né si confrontano velocità
assolute con altre campagne o con llama.cpp.

Ogni ON registra1843 selezioni (97×19), gli OFF zero. Tutte le righe CSV
riportano176.761.680 byte di sessione. Swap981,31 MiB prima e dopo tutti
i processi temporali. Sorgenti, snapshot, binari di prova e produzione
e stat del GGUF restano invariati. Lo store personale resta identico
nei controlli numerici. Non viene misurata la RAM totale o il suo picco.

Il primo fixture sintetico ereditava un costruttore che si aspettava
un engine già aperto. Senza `qw3_metal_init`, tentava di compilare la
libreria originale su un device nil. Entrambi i processi si fermavano
prima di inizializzare il device o confrontare output candidati.
Il tentativo, i build log e i manifest/snapshot originali restano in
`initial_constructor/`. La correzione inizializza esplicitamente Metal
prima della compilazione. Non sono stati eseguiti modello o tempi per
quel tentativo.

[Manifest numerico](measurements/iq3_pair_grid_2026-10-14/manifest.json),
[verifica](measurements/iq3_pair_grid_2026-10-14/verification.json) e
[analisi temporale](measurements/iq3_pair_grid_2026-10-14/native/analysis.json)
conservano sorgenti congelati, comandi, hash e log. Il verificatore
recupera helper e kernel originali invertendo soltanto le modifiche
della griglia. Non si archiviano binari, GGUF o checkpoint.

```sh
python3 tests/prepare_iq3_pair_grid.py
make qw3-iq3-pair-grid-test qw3-iq3-pair-grid-model-test qw3-iq3-pair-grid-native-bench
```

Correzione documentale: i messaggi del costruttore stampavano128 byte
dinamici anche nel processo modello. Il backend paired ne passa256;
128 è il valore del fixture. I log/manifest originali non vengono riscritti.
