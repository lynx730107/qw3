# Prefill M5: packed8 e proiezioni dirette nello scratch

La combinazione del DeltaNet packed8 e delle proiezioni senza copie intermedie
viene integrata nel motore denso. Il criterio di adozione viene aggiornato su
richiesta dell'utente: conta un guadagno ripetibile, con correttezza invariata e
senza regressioni dimostrate, anche sotto il precedente 5%. Il 5% dei report
precedenti era un criterio sperimentale, non un limite ricavato da llama.cpp.

## Implementazione

`qw3_metal_encode_dense_batch_matmul_strided` generalizza il dispatcher già
esistente con offset e stride per input/output; le chiamate compatte usano lo
stesso helper. Restano invariati shader di proiezione, precisione, ordine dei
prodotti e geometria. Il fallback mapped conserva la propria ABI compatta;
il core sceglie il percorso diretto solo quando tutti i kernel necessari sono
presenti. Le proiezioni Q/K/V o QKV/Z scrivono direttamente nello scratch già
usato dall'attenzione. La proiezione di output legge lo stesso scratch,
conservando l'addizione residua originale. Un chunk completo elimina 208 copie
compute: 144 di staging e 64 di compattazione; non sono 208 command buffer.

Il kernel packed8 conserva stato FP32 key-major, mappatura delle teste, gate e
ricorrenza sequenziale. Otto lane elaborano ogni colonna, con quattro blocchi
float4 per lane e 16 colonne per workgroup. Solo l'albero locale di riduzione
impedisce la riassociazione. La stessa implementazione del prototipo validato
viene compilata nella libreria di produzione e usa il dispatch/norm nativi.

Il default si applica al dispositivo esatto `Apple M5`, a chunk di almeno
16 token e ai kernel supportati. La ricorrenza packed8 richiede inoltre
head_dim=128 e una sessione che usa il prefill denso. MoE, chunk corti e
altri dispositivi conservano la selezione precedente. Le scelte esplicite
legacy/tiled4/tiled2-disable continuano a essere rispettate. Una pipeline
packed8 mancante lascia attivo tiled2. `QW3_METAL_DENSE_PREFILL_FAST=0`
ripristina staging/compattazione e ricorrenza precedenti per il confronto.

I diagnostici storici mantengono il controllo compatto tramite una macro
privata; il loro dispatcher strided ora riusa quello di produzione. Gli
snapshot delle campagne conservano l'implementazione effettivamente misurata.

## Metodo e prototipo combinato

Qwen3.8-27B-UD-Q3_K_XL GGUF locale invariato, Apple M5 24 GB, 64 layer Metal,
KV F16, ctx512, quattro thread, token sintetici seed1. pp128 puro e 32 forward
ordinari con tutti i logits a profondità128 vengono misurati separatamente.
Warmup, compilazione e preparazione della profondità restano fuori dal timer.
Cinque blocchi OFF/candidato/OFF/candidato/OFF, tre ripetizioni e 40 secondi
di pausa per blocco; una sola inferenza GPU alla volta.

[Campagna del prototipo](measurements/packed8_scratch_perf_2026-10-05/results.json):

| Misura | OFF, tok/s | Combinazione, tok/s | Variazione |
|---|---:|---:|---:|
| pp128 | 152,694 | 157,048 | +2,851% |
| tg32, depth128 | 8,776 | 8,803 | +0,312% |

Dispersione OFF: 0,101% pp128 e 0,192% tg32. I due confronti pp128 con
controlli vicini danno +2,892% e +2,823%. Non si sommano il +3% storico di
packed8 e il +1,5% della precedente prova delle copie. Il decode cronometrato
non usa questi kernel di prefill; il piccolo movimento tg32 non viene
attribuito alla modifica.

## Correttezza della combinazione e del codice integrato

La [replica sul vero qw3-bench integrato](measurements/native_prefill_perf_2026-10-05/results.json),
con lo stesso protocollo ctx512, misura 153,279 → 156,447 tok/s in pp128
(+2,067%). I due confronti vicini restano positivi, +3,039% e +0,898%, ma
l'entità varia più che nel prototipo; la dispersione OFF è 1,196%. Si adotta
il guadagno osservato di circa 2%, senza promettere un 3% costante. tg32 misura
8,841 → 8,851 tok/s (+0,109%), con confronti vicini +0,482% e −0,301%
e dispersione OFF 1,425%: nessuna accelerazione del decode dimostrata.

Il [prototipo](measurements/packed8_scratch_quality_2026-10-05/manifest.json)
e l'[implementazione integrata](measurements/native_prefill_quality_2026-10-05/manifest.json)
passano ciascuno otto prefissi: 1,15,16,31,127,128,129,159. Il caso 128 viene
esteso a 159 per verificare lo stato iniziale non nullo. Ogni caso confronta
finitezza e rappresentazione bit per bit dell'intero vocabolario, token e
KV F16 vivi, intero stato DeltaNet e convoluzione. Dopo salvataggio/caricamento
del payload verifica altri 32 forward greedy, con tutti i logits a ogni passo.
Capacità KV inutilizzata e scratch derivati di FlashAttention sono esclusi.

Il controllo integrato confronta le vere API di produzione con il loro opt-out
runtime, usando due sessioni indipendenti. I contatori delle chiamate dal core
confermano 1152 proiezioni dirette, 512 output diretti e1664 copie evitate nel
suite. Le 216 prove sulle 24 forme reali usano il dispatcher strided di
produzione: input/output con offset non nullo, stride con padding, code di
output 1/33/complete, guardie intatte, input invariato e layout invalidi rifiutati.
Non viene mantenuta una seconda copia del dispatcher per questi test.

Le [regressioni del default finale](measurements/native_prefill_regressions_2026-10-05/manifest.json)
passano senza modificare le soglie: bring-up denso Metal, logits/greedy/prefill
MoE, agente denso con checkpoint e riuso del prefisso, fingerprint runtime
e guardia di verifica dell'agente. CLI, agente, benchmark e build CPU vengono
ricostruiti. La nuova opzione di prefill entra automaticamente nell'identità
runtime dei checkpoint/cache. Il successivo
[riesame di tre vecchie alternative](m5_revisited_optimizations_2026-10-05.md)
conferma la scelta combinata; nessuna variante dei metadati viene promossa.

## Riferimento llama.cpp aggiornato

[Tre coppie alternate AB/BA](measurements/packed8_scratch_llama_2026-10-05/results.json),
ciascuna con tre ripetizioni e 40 secondi di pausa, confrontano il prototipo
corretto con llama-bench locale, build 11041 al commit `4fea119de` secondo
il JSON dell'eseguibile. Il checkout locale è invece alla revisione
`579ef425aba8563eee4e85c769e66543c7dac323`; le due identità vengono conservate
separatamente, insieme all'hash del binario misurato. Stesso GGUF, ctx256 per entrambi,
KV F16, full offload incluso output e warmup escluso. llama.cpp usa
FlashAttention e b/ub128. I token sintetici sono fissi per ogni motore ma
non sono lo stesso stream tra motori. Questo confronto ctx256 resta distinto
dalle campagne interne ctx512.

| Misura | qw3 combinato, tok/s | llama.cpp, tok/s |
|---|---:|---:|
| pp128 | 157,659 | 159,232 |
| tg32, depth128 | 8,794 | 9,023 |

Il divario medio è circa 1% nel prefill e 2,5% nella generazione. Le differenze
appaiate di latenza, baseline llama meno candidato qw3, sono rispettivamente
−8,02 ±8,39 ms e −92,26 ±126,15 ms: intervalli95% Student-t con df 2 sulle tre
coppie, non sui singoli token. Entrambi includono zero; non si dichiara né
un sorpasso né una parità dimostrata. I 160 tok/s storici sono vicini alla nuova
misura 159,23, ma non fissano un tetto del motore o dell'hardware.

Lo swap è presente durante le campagne; i manifest registrano prima/dopo
ogni processo. I dati sono prove brevi con warmup su M5 fanless, non misure
di throughput sostenuto o di contesto lungo.

## Riproduzione

```sh
make qw3-bench qw3-native-prefill-test qw3-native-prefill-matmul-test
QW3_METAL_KV_F16=1 QW3_METAL_KV_Q8_0=0 caffeinate -disu \
  ./qw3-native-prefill-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf all
QW3_METAL_KV_F16=1 QW3_METAL_KV_Q8_0=0 caffeinate -disu \
  ./qw3-native-prefill-matmul-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf
python3 tools/bench_qw3_candidates.py --output /tmp/qw3-native-fast-prefill \
  --binary ./qw3-bench --order off,candidate,off,candidate,off \
  --candidate-env QW3_METAL_DENSE_PREFILL_FAST=1 --repetitions 3 --cooldown 40
```
