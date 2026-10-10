# M5: due tile K32 per staging IQ3_S (2026-10-10)

Il lavoro precedente verificato è nel commit `d9ca3f6`. Il nuovo obiettivo
richiesto dall'utente è eguagliare, e possibilmente superare, llama.cpp sia in
pp128 sia in tg32 ordinario a profondità 128, mantenendo modello, precisione,
logits, stato e checkpoint. Una vittoria nel solo prefill non completa
l'obiettivo.

## Riferimento iniziale

Apple M5 base, 24 GB; GGUF locale `Qwen3.8-27B-UD-Q3_K_XL.gguf`,
13.146.393.504 byte, offload completo, KV F16, 4 thread, contesto 256.
Tre coppie di processi AB/BA, cinque ripetizioni e warmup escluso,
40 secondi di pausa prima di ogni coppia. Nessun campionamento o MTP;
i token sintetici sono ripetibili ma differiscono fra i due motori.
llama-bench locale: build 11041, commit del binario `4fea119de`;
il checkout `579ef425a` è distinto dalla revisione del binario.

| Fase | qw3 tok/s | llama.cpp tok/s | Differenza tempo a favore di qw3, ms ± intervallo 95% |
| --- | ---: | ---: | ---: |
| pp128 | 159,914 | 160,642 | −3,628 ± 2,980 |
| tg32, profondità 128 | 8,622 | 8,933 | −129,201 ± 369,363 |

Dispersione delle medie pp128: 1,29%/1,56%; tg32: 3,51%/5,43%.
La serie tg32 supera il limite di stabilità 5% per llama: la differenza
apparente non basta a stabilire un distacco affidabile.
[Comandi e risultati completi](measurements/dense_reference_2026-10-10/results.json).

## Variante e selezione

Il kernel IQ3_S staged128 carica due tile consecutivi K32 in memoria
threadgroup prima di eseguire i due matmul. Dimezza le coppie di barriere
del ciclo K, conservando descrittore K32, dequantizzazione FP16, input FP32,
accumulazione FP32 e ordine delle moltiplicazioni. La memoria necessaria
è 8 KiB, già allocata dal dispatcher originale. Nessuna nuova copia delle
matrici persistente e nessuna conversione del GGUF.

Lo screening privato include IQ3_S e IQ4_XS, K64 e K128. Sessantaquattro
casi di matrice con offset/stride, guardie, code di righe e 128/256 token
passano il confronto finito bit per bit; input immutati e layout invalidi
respinti. Duecento campioni isolati sono conservati. K64 IQ3_S migliora le
due matrici FFN isolate di circa 10%; IQ4_XS e K128 non hanno un vantaggio
coerente. Questi numeri servono solo a scegliere IQ3_S K64 per la prova
completa: non sono un guadagno di throughput del modello.
[Screening](measurements/kstage_screen_2026-10-10/micro_summary.json).

## Correttezza e prove complete

Il prototipo privato e l'integrazione nativa passano ciascuno otto prefissi
1,15,16,31,127,128,129,159; il caso 128 viene anche esteso a 159. Per ogni
caso: intero vocabolario finito bit per bit, K/V vivo, DeltaNet e convoluzione
esatti, salvataggio/caricamento del payload checkpoint e 32 continuazioni
con logits esatti. Il controllo nativo registra 333 dispatch K64 effettivi.
L'integrazione passa inoltre i 216 casi di matrice nativi strided,
due test del fingerprint runtime e la compatibilità checkpoint dell'agente.

Il primo avvio del test checkpoint usava erroneamente l'opzione dense
`QW3_METAL_NGL=64` sul modello MoE predefinito con 40 layer; il test ha
respinto la configurazione prima dell'inferenza. Il tentativo è conservato,
insieme alla riesecuzione riuscita con ambiente QW3/GGML pulito.
[Manifest e log nativi](measurements/kstage_native_quality_2026-10-10/manifest.json).

La prima campagna completa privata alterna OFF/K64/OFF/K64/OFF/K64/OFF,
5 ripetizioni e 60 secondi di pausa per blocco: 35 campioni conservati.
158,673 → 161,075 tok/s, +1,514% aggregato; dispersione OFF 5,189%.
Il campione lento iniziale resta incluso. Confronti con i controlli adiacenti:
+2,732%, +0,101%, +0,781%. La dispersione impedisce di presentare il
+1,514% come un effetto preciso e stabile.
[Campagna privata](measurements/kstage_pp128_2026-10-10/results.json).

Una replica con due sessioni nello stesso motore alterna otto coppie AB/BA,
dopo una coppia di warmup e 60 secondi di pausa. Risultato: +0,973%,
7,710 ms risparmiati per pp128, intervallo Student-t descrittivo ±1,563 ms;
tutte le otto coppie favoriscono K64. Logits e stato restano esatti.
Le coppie consecutive possono essere correlate; il diverso footprint delle
due sessioni rende queste velocità inadatte a dichiarare un throughput
assoluto di una normale sessione singola.
[Replica accoppiata](measurements/kstage_paired_2026-10-10/results.json).

La conferma con il vero `qw3-bench` di produzione confronta K64 disattivato
e attivato nello stesso binario. Sette processi OFF/K64 alternati, cinque
ripetizioni, 60 secondi di pausa: **160,317 → 162,459 tok/s, +1,336%**;
dispersione OFF 1,315%, entro il limite 5%. Tutti i tre confronti con i
controlli adiacenti sono positivi. Nessuna delle 35 ripetizioni eliminata;
sorgenti e binario immutati durante la campagna. Insieme alla replica
accoppiata, questa serie giustifica l'adozione M5 come guadagno piccolo,
senza promettere una percentuale costante.
[Conferma nativa](measurements/kstage_native_perf_2026-10-10/results.json).

## Confronto aggiornato con llama.cpp

Cinque coppie AB/BA, cinque ripetizioni per motore, stessi modello/offload/KV
e contesto 256 del riferimento iniziale. pp128: **qw3 162,091, llama.cpp
162,210 tok/s**, divario circa 0,074%. Dispersioni delle medie 0,836% e
0,231%. Differenza di tempo a favore di qw3 −0,581 ± 4,565 ms, intervallo
Student-t 95% fra coppie: nessun sorpasso confermato in questa serie.
I risultati sono compatibili con una sostanziale parità nel prefill, senza
costituire un test formale di equivalenza o una garanzia per altre lunghezze.
Il confronto nativo A/B e questo confronto fra motori sono campagne distinte;
la variazione assoluta dal vecchio riferimento non viene attribuita tutta
alla nuova variante.
[Confronto aggiornato](measurements/kstage_native_reference_2026-10-10/results.json).

tg32 a profondità 128 misura **8,869 / 9,111 tok/s** per qw3/llama,
divario apparente 2,658%. Differenza di tempo −95,900 ± 135,752 ms;
dispersioni 4,223%/5,334%. Il controllo llama supera ancora il limite 5%,
quindi non si dichiara un distacco stabile né una parità dimostrata nel
decode. Il kernel K64 non modifica i 32 forward cronometrati del decode;
una variazione rispetto alla serie iniziale non è una sua accelerazione.
La serie conserva tutte le 100 ripetizioni, sorgenti e binari immutati.
L'obiettivo sulle due fasi resta aperto.

## Politica nativa

Kernel opzionale, selezionato soltanto su `Apple M5`, con la politica dense
fast abilitata e batch staged128 supportato. Altri dispositivi e librerie
Metal esterne senza il nuovo kernel mantengono il precedente fallback.
`QW3_METAL_DENSE_PREFILL_K64=0` ripristina solo il kernel IQ3_S K32,
conservando packed8 e direct scratch; `QW3_METAL_DENSE_PREFILL_FAST=0`
disabilita l'intera ottimizzazione dense fast. Il matvec a un token non usa
questo kernel. I vecchi diagnostici che sostituiscono il pipeline IQ3_S K32
disabilitano esplicitamente K64, per evitare che nasconda i loro candidati.

I manifest archiviano hash e snapshot dei sorgenti delle rispettive campagne.
Le modifiche successive del harness non alterano gli snapshot storici.
Il binario accoppiato costruito in `/tmp` e il target Make finale hanno hash
diversi: non vengono presentati come lo stesso eseguibile.

## Riproduzione

```sh
make qw3-kstage-matrix-test qw3-kstage-prefill-test qw3-native-kstage-test
QW3_METAL_KV_F16=1 ./qw3-kstage-matrix-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 64 quality
QW3_METAL_KV_F16=1 QW3_METAL_NGL=64 ./qw3-native-kstage-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf all
python3 tools/bench_qw3_candidates.py --binary ./qw3-bench --jobs pp128 \
  --repetitions 5 --order off,candidate,off,candidate,off,candidate,off \
  --candidate-env QW3_METAL_DENSE_PREFILL_K64=1 --cooldown 60 --output NEW_NATIVE_DIRECTORY
python3 tools/bench_dense_reference.py --jobs pp128,tg32_d128 \
  --pairs 5 --repetitions 5 --cooldown 40 --output NEW_REFERENCE_DIRECTORY
```

I percorsi di output devono essere nuovi. Una sola inferenza GPU attiva;
conservare tutte le ripetizioni e non modificare binari/sorgenti nel corso
delle campagne. Il confronto fra motori usa contesto 256; l'A/B nativo
usa contesto 512. Non confrontare direttamente i valori assoluti delle
due campagne per attribuire un guadagno o una vittoria.
