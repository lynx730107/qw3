# Piano di sviluppo: Qwen3.8-27B denso Q3_K_XL

## Obiettivo

Estendere `qw3` per eseguire il modello locale
`/Users/pasqualecapasso/Downloads/models/Qwen3.8-27B-UD-Q3_K_XL.gguf`,
mantenendo il supporto al modello MoE esistente e la stessa filosofia del
progetto: runtime verticale, dipendenze minime, Metal come backend principale,
CPU come riferimento di correttezza e `llama.cpp` come oracolo funzionale e
prestazionale.

Il lavoro parte dal branch `feature/qwen3.8-27b-dense`. Il riferimento locale
di `llama.cpp` e' `/Users/pasqualecapasso/Downloads/llama.cpp`, al commit
`4fea119de` al momento della stesura di questo piano.

Come riferimento aggiuntivo per l'ottimizzazione Metal viene usato
[`incoai/splash`](https://github.com/incoai/splash), commit analizzato
`f58d36ddb046726adc8937ab67d07bdde015c0d6` (licenza Apache-2.0). Splash usa
lo stesso Qwen3.8-27B e forme architetturali coincidenti, ma un formato Q4
ripacchettato specificamente per i suoi kernel. Le sue prestazioni sono quindi
un riferimento architetturale, non un confronto diretto con il nostro GGUF
`Q3_K_XL` misto.

## Stato implementazione

Prima tranche completata sul branch di lavoro:

- introdotti i profili espliciti MoE e dense con validazione stretta dei
  metadati e del numero di tensori;
- distinti i 64 layer trunk dal blocco 64 NextN/MTP e associati tutti i
  tensori del GGUF denso, inclusi FFN e tensori MTP;
- mantenuto e verificato il binding del GGUF MoE esistente;
- resi pubblici `--inspect` e `--layer-types`, utilizzabili senza creare una
  sessione di inferenza;
- resa la stima memoria dipendente dal profilo caricato;
- aggiunti decoder e matvec CPU per tutti i formati realmente presenti nel
  GGUF (`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `IQ2_XXS`,
  `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`);
- aggiunto un test differenziale deterministico contro i decoder esportati
  dal `llama.cpp` locale; tutti i formati richiesti superano il confronto;
- implementato il riferimento CPU SwiGLU denso e verificato sul layer 0 del
  GGUF reale (gate `IQ2_XS`, up `IQ2_S`, down `IQ3_XXS`);
- resi dipendenti dal profilo il forward CPU DeltaNet/GQA, le mappe layer, lo
  stato ricorrente, la KV cache e l'output; il pass completo sui 64 layer
  trunk ora viene eseguito su CPU, escludendo correttamente il blocco MTP;
- aggiunto al probe opzionale `QW3_PROBE_DENSE_FORWARD=1` un forward completo
  di un token con argmax, utile come smoke test senza template chat;
- aggiunto `make test-model-profiles`, eseguito sui due GGUF locali.
- aggiunto `make test-dense-logits`, che confronta il vettore logits completo
  e otto token greedy dal token grezzo 66 contro `llama.cpp` `4fea119de` su
  Metal: top-8 e sequenza greedy coincidono, con RMSE `0.0001391` e massimo
  scarto assoluto `0.0007711` sulla fixture corrente.
- resi dipendenti dal profilo anche i payload delle sessioni CPU; il nuovo
  formato v2 aggiunge checksum, continua a leggere il formato v1 e viene
  verificato sul modello dense da `make test-dense-session` con quattro passi
  decode identici dopo il ripristino e corruzione rifiutata.
- avviata la Fase 3 rendendo l'allocazione della sessione Metal dipendente dal
  profilo: il modello denso alloca 16 layer KV GQA, 48 stati DeltaNet e buffer
  dimensionati per embedding 5120, GQA 24/4 e inner 6144;
- implementato e confrontato con la CPU un layer DeltaNet/FFN denso completo
  (layer 0), inclusi convoluzione causale, ricorrenza con stato persistente,
  gated RMSNorm, residual e SwiGLU;
- implementato e confrontato con la CPU un layer GQA/FFN denso completo
  (layer 3), con proiezioni Q/K/V miste `IQ4_NL`/`Q4_K`/`Q5_K`, RoPE,
  attenzione, output `Q4_K` e FFN `Q3_K`/`IQ3_S`/`IQ4_XS`;
- verificata la cache KV GQA su due token sia F16 sia F32;
  `make test-dense-metal-bringup` copre i confini intermedi e i due layer
  completi;
- completati i matvec Metal per tutti i 13 formati quantizzati del GGUF nel
  percorso x1, gli otto formati presenti nelle proiezioni da inner, e
  l'output logits `Q5_K`; il test differenziale usa tensori reali;
- composto il decode Metal sui 64 layer trunk, con cache KV F16/F32 e stato
  DeltaNet persistente; due token consecutivi e otto token greedy coincidono
  con la CPU, inclusa la sequenza della fixture `llama.cpp` dal token 66;
- abilitata la generazione densa Metal con offload completo, verificata da
  prompt breve, e adattato il payload della sessione Metal al profilo denso;
  roundtrip di quattro passi decode esatto. Offload parziale e KV Q8 vengono
  rifiutati per il denso finche' non sono validati;
- come primo passo della Fase 4, i kernel del decode denso vengono inviati in
  un unico command batch per token, con sincronizzazione prima di avanzare lo
  stato della sessione. Il prefill resta token-per-token: non e' ancora un
  matmul multi-token ne' soddisfa i gate `pp128`/`pp512`/`pp4096`. Una prova
  differenziale di 32 passi greedy con KV F16 mantiene tutti gli argmax CPU e
  Metal uguali (massima differenza logit osservata 0.00233);
- avviata la Fase 5 per l'agente: `--ngl` viene validato sul profilo caricato,
  il denso rifiuta subito offload parziale e KV Q8, e `--chdir` conserva i
  percorsi di modello, store e kernel Metal. `make test-dense-agent` copre
  generazione, checkpoint e prefix cache; `--compact-tools` limita il prompt
  ai quattro tool locali `read`/`write`/`edit`/`bash` per contenere il prefill.

L'inferenza densa e' abilitata sui backend CPU e Metal. Il gate numerico
minimo della Fase 2 e il decode corretto della Fase 3 sono chiusi sul modello
locale e sulla fixture dal token 66. Il prefill Metal denso ha un percorso
batch misto, ma GQA e parti di DeltaNet/FFN restano token-per-token ed e'
ancora lontano dal target prestazionale. Restano l'ottimizzazione della Fase 4,
la validazione estesa del template chat e dell'agente della Fase 5 e i benchmark
di ottimizzazione.

## Principi di ottimizzazione derivati da Splash

L'analisi del runtime e dei kernel di Splash suggerisce di adottare questi
principi, compatibilmente con la filosofia verticale di `qw3`:

1. **Kernel distinti per fase e forma esatta.** Decode e prefill non devono
   condividere per forza lo stesso kernel. Dopo il fallback corretto, creare
   varianti compilate per le forme 5120, 6144, 10240, 14336, 16640 e 17408 del
   modello, senza introdurre un planner generico.
2. **FFN denso fuso per confini naturali.** Nel decode, calcolare gate e up
   dalla stessa attivazione nello stesso dispatch e produrre direttamente
   `silu(gate) * up`; il down deve sommare il residuo nell'epilogo. Nel prefill,
   mantenere gate separato solo quando questo migliora occupazione e scratch,
   fondendo almeno up+SwiGLU e down+residuo.
3. **Piani di esecuzione deterministici.** Scegliere tile, numero di
   simdgroup e griglia persistente in base a fase, forma, tipo GGUF e famiglia
   GPU. La scelta deve essere costruita una volta all'avvio, non ripetuta nel
   percorso caldo.
4. **Griglie persistenti e bilanciate.** Per il decode memory-bound, una
   threadgroup deve elaborare piu' tile in stride. Il numero di gruppi va
   calibrato sul numero di core GPU per evitare sia griglie troppo corte sia
   troppe wave che rileggono input e metadati.
5. **Metal Tensor API dove e' realmente adatta.** Il progetto possiede gia'
   il probe Metal 4 e il Mac di sviluppo espone l'API. Usarla per prefill e
   formati che possono essere consumati senza conversione distruttiva, con
   accumulo FP32. Non forzare i blocchi `Q*_K`/`IQ*` dentro il layout Q4 affine
   di Splash: occorrono loader e tile nativi per i tipi GGUF.
6. **GDN e attenzione specializzate.** Compilare varianti esatte per 48 value
   heads e GQA 24/4, eliminando branch di geometria nei kernel. Valutare un
   singolo kernel GDN decode che unisca preparazione, update ricorrente, norm e
   gate, una volta validati separatamente i singoli stadi.
7. **KV Q8 paginata come fase successiva.** Splash ottiene una parte del suo
   vantaggio da KV Q8 paginata, cache di prefissi e allocazione elastica. Per
   `qw3` la prima milestone resta KV F16 corretta; Q8 e paging entrano solo
   dopo regressione logits e session checkpoint.
8. **Scratch e memoria pianificati all'avvio.** Preallocare arena separate per
   decode e prefill, dimensionate sul massimo realmente ammesso. Il budget deve
   considerare `recommendedMaxWorkingSetSize`, pesi mmap, stato DeltaNet, KV,
   scratch e margine operativo, rifiutando configurazioni che non entrano.
9. **Misure per operatore.** Ogni scelta di tile/fusione deve avere benchmark
   caldo dedicato e confronto end-to-end. Conservare la baseline e promuovere
   una variante solo se migliora piu' forme/contesti senza cambiare logits.

Elementi di Splash non adottati nella prima implementazione:

- il formato pesi Q4 group-64 con tile di storage da 256 colonne, perche' non
  e' compatibile con la GGUF mista senza conversione o nuova perdita;
- DFlash/speculative decoding, che richiede un draft addestrato e un package
  pesi aggiuntivo;
- continuous batching e scheduler multi-request, estranei al primo obiettivo
  di correttezza della CLI/agente locale;
- il formato package proprietario di Splash: `qw3` continua a leggere il GGUF
  direttamente e a mappare i pesi senza copia.

Se in seguito viene adattato codice Apache-2.0 invece dei soli principi, ogni
file derivato deve riportare provenienza, commit e licenza; non vanno copiati
kernel che assumono il packing Q4 di Splash.

## Dati verificati sul modello reale

| Campo | Valore |
| --- | ---: |
| Architettura GGUF | `qwen35` |
| Nome | `Qwen3.8-27B` |
| Tensori | 866 |
| Blocchi dichiarati | 65 |
| Blocchi trunk eseguiti | 64 |
| Blocchi NextN/MTP | 1, il blocco 64 |
| Embedding | 5120 |
| Vocabolario | 248320 |
| FFN denso | 17408 |
| Teste Q / KV | 24 / 4 |
| Dimensione testa | 256 |
| Contesto dichiarato | 262144 |
| Pattern attenzione | 3 DeltaNet + 1 full attention |
| Layer DeltaNet / full attention nel trunk | 48 / 16 |
| DeltaNet state / group / rank / inner | 128 / 16 / 48 / 6144 |
| RoPE | dimensione 64, base 10000000 |

Il blocco 64 contiene i tensori NextN/MTP. `llama.cpp` non lo esegue nel pass
principale: il primo obiettivo e' quindi generazione standard sui 64 layer del
trunk. Il supporto MTP resta fuori dal primo rilascio.

La GGUF non usa un solo formato Q3. I pesi contengono una miscela di `Q2_K`,
`Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`,
`IQ3_XXS`, `IQ3_S`, `IQ4_NL` e `IQ4_XS`. In particolare, embedding e output
sono rispettivamente `Q3_K` e `Q5_K`. La copertura delle quantizzazioni e'
quindi parte del percorso critico, non un'ottimizzazione facoltativa.

## Scelta architetturale

Non trasformare `qw3` in un runtime generico. Introdurre invece due profili
espliciti e rigorosamente validati:

1. Qwen3.6-35B-A3B `qwen35moe`, gia' supportato.
2. Qwen3.8-27B `qwen35` denso, oggetto di questo lavoro.

Un piccolo `qw3_model_spec` deve raccogliere dimensioni, numero di layer trunk
e MTP, schema FFN, pattern di attenzione e limiti di contesto. Il loader sceglie
il profilo dai metadati GGUF e rifiuta combinazioni non conosciute. I kernel
ricevono le dimensioni dal profilo, ma possono mantenere specializzazioni per i
due modelli. Questa soluzione conserva la leggibilita' verticale senza
duplicare loader, tokenizer, sessioni, attention e DeltaNet.

I pesi per layer vanno separati in una parte comune e una variante FFN:

- `dense`: `ffn_gate`, `ffn_up`, `ffn_down`;
- `moe`: router, esperti routed e shared expert esistenti.

Gli array attualmente dimensionati a 40 layer e le costanti replicate nel
backend Metal vanno resi dipendenti dal profilo o portati a un massimo interno
controllato. Il loop di inferenza deve usare 64 layer per il nuovo modello e
ignorare esplicitamente il layer MTP 64 nel pass standard.

## Fase 0 - Baseline e fixture da llama.cpp

Prima di modificare l'inferenza:

- costruire `llama-cli` e usare `llama-bench` dal commit di riferimento;
- salvare tokenizzazione, top logits e sequenza greedy per prompt brevi
  (`ciao`, testo inglese e un prompt agent-like);
- registrare `pp128`, `pp512`, `pp4096` e `tg32` a profondita' 0, 4096 e 16000,
  compatibilmente con memoria e temperatura del Mac;
- acquisire vettori intermedi almeno dopo embedding, layer 0, layer 3,
  layer 63, norm finale e output;
- registrare configurazione KV, Flash Attention, batch e commit esatto di
  `llama.cpp` insieme alle misure.

Gate: fixture riproducibili e benchmark di riferimento salvati prima di ogni
ottimizzazione.

## Fase 1 - Profilo, loader e binding dei tensori

- Aggiungere i metadati `qwen35.feed_forward_length` e
  `qwen35.nextn_predict_layers` al parser.
- Distinguere `n_layer_all = 65` da `n_layer = 64`.
- Accettare `general.architecture = qwen35` solo con le dimensioni verificate.
- Associare i tensori densi `ffn_gate/up/down` e i tensori GQA/DeltaNet gia'
  condivisi con il percorso MoE.
- Riconoscere e validare i tensori MTP senza caricarli nel grafo standard.
- Eliminare dal percorso comune le assunzioni fisse 40/2048/16/2/32/4096.
- Aggiungere un comando informativo che stampi profilo, layer effettivi,
  quantizzazioni e stima memoria senza avviare una sessione.
- Mantenere invariati i test e il caricamento del modello MoE.

Gate: la GGUF densa viene validata e associata completamente; un tensore
mancante, una forma errata o un tipo non supportato produce un errore preciso.

## Fase 2 - Riferimento CPU e quantizzazioni

Portare dal `llama.cpp` locale, con attribuzione e note di licenza, le sole
strutture di blocco e primitive necessarie per i formati presenti nella GGUF.
Creare un dispatch unico `tensor_matvec` invece di controlli di tipo sparsi nei
singoli layer.

Ordine di lavoro:

1. test di dequantizzazione/dot per ogni formato, confrontati con `ggml` su
   blocchi deterministici;
2. embedding `Q3_K` e output `Q5_K`;
3. proiezioni DeltaNet e GQA miste;
4. FFN denso SwiGLU: `down(silu(gate(x)) * up(x))`;
5. pass completo CPU su prompt minimo.

Il percorso CPU non deve essere veloce: serve come oracolo locale indipendente
dal backend Metal. Deve pero' usare l'intera GGUF senza dequantizzarla in RAM.

Gate: vettori intermedi e top logits allineati a `llama.cpp`; token greedy
identici sui prompt fixture.

## Fase 3 - Metal decode corretto

Stato: gate di correttezza decode chiuso sulla fixture locale. Il forward
Metal completo attraversa 64 layer con tutti i formati quantizzati richiesti.
Con cache KV F16/F32 i logits di due token e otto passi greedy corrispondono
alla CPU; la sequenza greedy coincide anche con `llama.cpp`. La generazione
CLI e il roundtrip della sessione Metal densa passano. Restano ottimizzazione,
validazione su piu' prompt e supporto esplicito per offload parziale e KV Q8.

- Passare al backend Metal il profilo completo della sessione.
- Ridimensionare cache e stato per 16 layer GQA e 48 layer DeltaNet.
- Parametrizzare i kernel esistenti per 5120 embedding, 24/4 teste, inner 6144
  e rank 48.
- Aggiungere kernel matvec quantizzati per tutti i formati realmente presenti.
- Implementare il FFN denso in tre stadi iniziali chiari: gate/up, SwiGLU,
  down; fondere solo dopo la validazione.
- Tenere logits e sampling su GPU, leggendo sul lato CPU solo quanto richiesto
  dall'interfaccia.
- Confrontare CPU e Metal a ogni confine di layer, rifiutando NaN/Inf oltre
  alle normali tolleranze numeriche.

Gate: decode di un token, prompt breve e almeno otto token greedy coincidono
con il riferimento; il modello MoE continua a superare la suite esistente.

## Fase 4 - Metal prefill

Il fallback token-per-token e' corretto ma lento. Il FFN denso domina il costo
e non puo' usare il percorso MoE. Implementare
prima un kernel batch universale corretto, poi specializzare i formati piu'
frequenti (`IQ4_XS`, `IQ3_S`, `IQ3_XXS`) e quelli sui tensori critici.

- Riutilizzare Flash Attention e Gated DeltaNet esistenti dopo la
  parametrizzazione.
- Implementare gate e up nello stesso command batch, SwiGLU compatto e down
  senza materializzare pesi dequantizzati.
- Usare chunk di prefill adattivi per limitare lo scratch su macchine da 24 GB.
- Conservare un fallback Metal generico per i formati rari, cosi' la
  correttezza non dipende subito da tredici kernel altamente ottimizzati.
- Promuovere una specializzazione solo dopo regressione logits e benchmark
  ripetuto caldo/freddo.

Gate: `pp128`, `pp512` e `pp4096` corretti; nessuna crescita di memoria per
token oltre KV e scratch previsto; throughput misurato contro la baseline
`llama.cpp` nelle stesse condizioni.

## Fase 5 - Sessioni, CLI e agente

- Rendere dinamici `--ngl`, conteggio layer e stime memoria.
- Includere profilo, dimensioni e firma dei tipi tensoriali nel fingerprint
  delle sessioni; i checkpoint MoE precedenti devono essere rifiutati in modo
  pulito se incompatibili, non interpretati con il nuovo layout.
- Adeguare serializzazione di KV, stato DeltaNet e mappe layer.
- Verificare tokenizer `qwen35`, special token e template Qwen3.8, inclusi
  developer message, reasoning effort e formato tool-call.
- Eseguire smoke test dell'agente con almeno `read`, `write`, `edit` e `bash`.
- Aggiornare help, README e default dei test permettendo di selezionare il
  modello tramite variabile d'ambiente, senza copiare la GGUF nel repository.

Gate: salvataggio/ripristino sessione, prefix cache e ciclo agent/tool
funzionanti sul modello denso; compatibilita' MoE preservata.

## Fase 6 - Ottimizzazione guidata dai profili

Solo dopo il completamento dei gate precedenti:

- profilare separatamente embedding/output, DeltaNet, GQA e FFN denso;
- ottimizzare prima decode, poi prefill, misurando anche il working set;
- confrontare le forme dei kernel con le implementazioni Metal correnti di
  `llama.cpp`, senza introdurre `ggml` come dipendenza runtime;
- valutare fusioni gate+up, SwiGLU in-place, down tiled/MPP e riuso RHS;
- mantenere ogni variante dietro flag finche' non supera logits, stabilita' e
  benchmark su piu' ripetizioni;
- documentare anche gli esperimenti scartati, come gia' avviene per il MoE.

Target prestazionale richiesto: almeno eguagliare il `llama.cpp` locale sia su
`pp128` sia su `tg32`, a parita' di modello, offload e tipo KV. Le misure dei
singoli kernel sono diagnostiche, non sostituiscono questo criterio end-to-end.

Stato misurato (Apple M5 fanless, Q3_K_XL, 64 layer Metal, singola prova
`pp128/tg32`, senza warmup): `qw3` 5.24/4.71 tok/s; `llama.cpp`
150.0/7.70 tok/s. I valori possono calare in prove consecutive per throttling.
La selezione adattiva dei threadgroup matvec e' stata validata con la suite
differenziale Metal completa; su prompt breve il throughput e' passato da
circa 4 a circa 5 tok/s. Profilo sincrono: FFN gate/up e down dominano il
decode. Raggruppare quattro token consecutivi in un solo command buffer non ha
prodotto un guadagno misurabile ed e' stato scartato: manca il riuso dei pesi
tra token. Il prossimo intervento ad alto impatto e' matmul quantizzata
multi-token con scratch per-token e GQA/DeltaNet batched, partendo dai formati
FFN prevalenti IQ3_S/IQ4_XS, mantenendo il fallback tokenwise finche' logits,
greedy e checkpoint non coincidono.

Prova isolata dei kernel Metal tiled/MPP esistenti (mappa a un solo esperto):
proiezione FFN `5120 x 17408` per 128 token in 15.6-20.0 ms con IQ3_S e
3.8-6.5 ms con IQ4_XS in prove isolate non termicamente controllate. Scarto
rispetto al CPU sui token 0, 31, 32 e 127:
IQ3_S max 0.000833, RMS 0.000198; IQ4_XS max 0.001161, RMS 0.000245.
Il modello contiene 70 matrici FFN IQ3_S e 69 IQ4_XS su 192, quindi i due
kernel coprono una parte rilevante ma non bastano da soli. Il nucleo FFN batch
completo (gate/up/SwiGLU/down
e somma residua in un command buffer) passa il confronto CPU su 128 token:
circa 30 ms per un layer tutto IQ3_S, circa 12 ms per uno tutto IQ4_XS,
RMS <= 0.00033 sui token 0, 31, 32 e 127. Sono disponibili e testate anche
le copie di riga x0/x1 fra buffer tokenwise e batch. Il prefill per-layer e'
ora attivo di default con batch fino a 128 token e fallback per i formati FFN
non ancora batch. Confronto differenziale su 128 token: stesso argmax, logits
maxdiff 0.00909/RMS 0.00169; stato DeltaNet del primo layer identico;
continuazione del token successivo con stesso argmax. `test-dense-agent` e
`test-dense-metal-bringup` passano con il batch di default. Una misura
`pp128` senza warmup e' passata da 5.45 a 7.2-7.9 tok/s in prove successive;
`tg32` resta 4.78 tok/s, ancora distante da 150.0/7.70 di `llama.cpp`.
Il prossimo intervento principale e' batched matmul delle proiezioni
attenzionali e dei formati FFN rimanenti: il tokenwise interno al layer
impedisce tuttora il riuso dei pesi nella maggior parte del prefill.
Profilo `pp128` con batch di default prima del percorso misto: 36 layer FFN
supportati impiegano circa 4.5 s nell'attenzione tokenwise e 0.8 s nella
FFN batch; 28 layer con fallback FFN impiegano circa 12.5 s complessivi.
Il percorso misto usa in batch anche le proiezioni IQ3_S/IQ4_XS dei 13 layer
con due matrici supportate: 49 layer su 64 sono ora almeno parzialmente
batchizzati, 15 restano tokenwise. Confronto logits su 128 token e
continuazione superato (stesso argmax, maxdiff 0.0119/RMS 0.00224),
`test-dense-metal-bringup` e `test-dense-agent` verdi. Una misura senza
warmup ha dato `pp128/tg32` 9.11/5.06 tok/s; una successiva misura profilata
ha dato solo `pp128` 7.02 tok/s; overhead del profilo e carico termico non
sono stati separati, quindi non si presume il picco sostenuto.

Le proiezioni DeltaNet QKV e gate/Z IQ3_S/IQ4_XS usano ora il matmul batch
sull'input normalizzato per layer; la convoluzione e lo stato ricorrente
restano in ordine token per token. Nei 48 layer DeltaNet, 40 QKV e 34 gate/Z
sono supportati. Il confronto batch contro tokenwise su 128 token mantiene
argmax e continuazione del token seguente: maxdiff logits 0.01538,
RMS 0.00249; stato DeltaNet del primo layer maxdiff 0.000627,
RMS 0.0000213 (non piu' identico bit a bit per il diverso kernel). Le suite
`test-dense-metal-bringup` e `test-dense-agent` passano. Una misura senza
warmup ha dato `pp128/tg32` 10.72/4.89 tok/s; un profilo `pp128` successivo
ha dato 6.39 tok/s. Le misure non sono termicamente controllate. Il profilo
mostra ancora molti layer con FFN tokenwise a circa 0.6-1.3 s per 128 token,
mentre i layer con FFN batch sono molto piu' rapidi. Priorita': formati FFN
rimasti e proiezioni GQA, poi riduzione dell'overhead nel decode. Il gate
resta la parita' end-to-end con `llama.cpp` su entrambi `pp128` e `tg32`.
Tra i 15 layer FFN completamente tokenwise, i pesi sono soprattutto IQ2_S,
IQ3_XXS, IQ2_XS e Q3_K: estendere solo IQ3_S/IQ4_XS non puo' eliminarli.
Un profilo sincrono di un token generato ha misurato circa 98 ms per gate/up
e 56 ms per down sui 260 ms totali; i tempi includono le sincronizzazioni del
profilo e non rappresentano il throughput normale. La FFN resta quindi anche
il bersaglio principale per `tg32`.

Kernel batch4 per IQ2_S e IQ3_XXS: una decodifica del peso alimenta quattro
token. Prove isolate `5120 x 17408`, 128 token, contro la CPU: IQ2_S 89.5 ms,
maxdiff 3.34e-6; IQ3_XXS 110.7 ms, maxdiff 2.63e-6. La copertura FFN con
almeno due proiezioni batch sale da 49 a 61 layer su 64; DeltaNet QKV da
40 a 46 su 48 e gate/Z da 34 a 40 su 48. Il confronto modello completo su
128 token mantiene argmax e continuazione, con maxdiff logits 0.01393 e
RMS 0.00212. Una prova `pp128/tg32` immediatamente dopo varie prove GPU ha
dato 9.19/3.45 tok/s, contro 10.72/4.89 in una prova precedente; il profilo
sincrono `pp128` ha dato 7.60 tok/s contro 6.39 prima. La variabilita'
termica non consente di attribuire un miglioramento end-to-end a questi
kernel. Inoltre i loro 89-111 ms/proiezione sono molto piu' lenti dei kernel
MPP IQ3_S/IQ4_XS (4-20 ms): la prossima iterazione deve usare tiling/matmul
piu' efficiente, non solo coprire altri formati. Il target `llama.cpp` resta
aperto.

La variante Metal 4 con input F32 diretto e matmul tiled per IQ2_S/IQ3_XXS
riduce le prove isolate `5120 x 17408`, 128 token, a 10.1/8.35 ms; maxdiff
contro la CPU 0.00227/0.00187. Sotto 16 token o senza tensor API resta il
batch4 corretto. Un primo tentativo con staging dell'input in shared memory
era rapido ma numericamente errato ed e' stato eliminato. Il confronto
modello completo su 128 token e continuazione passa: logits maxdiff 0.01741,
RMS 0.00306, stesso argmax. Una singola prova end-to-end senza warmup ha dato
`pp128/tg32` 15.27/4.70 tok/s, contro 9.19/3.45 con i kernel batch4 nella
prova precedente. Le due prove non sono termicamente controllate; il guadagno
dei kernel isolati e' solido, quello end-to-end va confermato in serie A/B.
Resta ancora circa 10x sotto `llama.cpp` su `pp128`; `tg32` non ha beneficiato
del matmul batch e resta sotto il target.

Il formato Q3_K ora ha un kernel Metal batch4 che riusa ciascun blocco di pesi
per quattro token. Il test isolato di una proiezione FFN reale su 128 token
confronta quattro righe con la CPU (maxdiff 1.91e-6), ma impiega circa 351 ms.
Con `QW3_METAL_DENSE_Q3_K_BATCH=1` entra nel percorso FFN batch misto e nelle
proiezioni DeltaNet; il confronto end-to-end su 128 token mantiene argmax e
continuazione (maxdiff logits 0.01686, RMS 0.00293; maxdiff dopo un ulteriore
token 0.01361). Una breve serie A/B/A `pp128` sullo stesso Mac ha misurato
16.00/12.82/15.78 tok/s con Q3_K batch spento/acceso/spento, a indicare una
regressione malgrado l'incertezza termica. Il kernel resta disponibile ma
disattivato di default finche' non si usi un matmul tiled piu' efficiente.

La successiva variante Q3_K Metal 4 tiled usa direttamente l'input F32 e
porta la stessa proiezione isolata da 351 a 4.8 ms per 128 token (maxdiff
contro CPU 0.00138, RMS 0.000321). Il confronto sul modello completo a 128
token mantiene argmax, stato DeltaNet e continuazione (maxdiff logits
0.01573, RMS 0.00292). In tre ripetizioni `pp128` consecutive ha dato
18.76 ± 0.20 tok/s, contro 15.41 ± 0.29 con Q3_K batch disattivato; le
singole prove fredde sono state piu' variabili. Il tiled Q3_K e' ora scelto
automaticamente solo su Metal 4 per batch di almeno 16 token; `=0` forza il
vecchio percorso, `=1` abilita anche il fallback batch4 sperimentale. Restano
GQA e i formati FFN non batchizzati, oltre al target `llama.cpp`.

Il tiled Metal 4 Q4_K/Q5_K con scala e minimo GGUF condivisi copre anche la
FFN dell'ultimo layer: la copertura batch sale a 63 layer su 64. Prove isolate
su 128 token: Q4_K `5120 x 1024` 0.97 ms (maxdiff CPU 0.00280), Q5_K
`5120 x 17408` 23.6 ms (maxdiff 0.00453). Il confronto modello completo
passa con stesso argmax e continuazione (maxdiff logits 0.02025, RMS 0.00330).
Tre ripetizioni `pp128` hanno dato 20.84 ± 0.47 tok/s, contro 17.80 ± 0.85
disabilitando solo Q4_K/Q5_K. Il percorso e' selezionato automaticamente su
Metal 4 per batch di almeno 16 token; `QW3_METAL_DENSE_Q4_Q5_K_BATCH=0`
consente il controllo A/B. Rimane tokenwise la FFN del layer 14
(`IQ2_XXS/IQ2_S/IQ2_XS`) e l'attenzione GQA.

Le proiezioni GQA Q/K/V con formati batch supportati vengono ora calcolate
sull'intero batch prima del pass causale; per ogni token si copia soltanto la
riga di proiezione necessaria, mantenendo il matvec tokenwise per i formati
non ancora batchizzati. L'attenzione causale e la cache KV restano ordinate
token per token. Il confronto 128-token contro il percorso seriale passa con
stesso argmax, stato DeltaNet e continuazione (maxdiff logits 0.01879,
RMS 0.00315). Tre ripetizioni `pp128` misurano 24.01 ± 0.38 tok/s contro
20.54 ± 0.50 con `QW3_METAL_DENSE_GQA_BATCH=0`. Il percorso e' attivo per
batch di almeno 16 token; `=1` forza anche i batch piu' piccoli per test.
Il target `llama.cpp` rimane lontano: occorre batchizzare anche le operazioni
di attenzione e i formati residui, oltre a ridurre il costo del decode.
Una prova successiva senza warmup `pp512/tg32` (quattro chunk da 128) ha dato
17.44/3.60 tok/s; `tg32` isolato in tre ripetizioni 3.53 ± 0.15 tok/s.
Il Mac era gia' caldo dopo molte prove GPU: questi valori non vanno confrontati
direttamente con il `tg32` freddo precedente, ma mostrano che il decode e la
tenuta termica restano problemi aperti.

## Budget memoria previsto

Con KV F16, a contesto 32768 i 16 layer GQA richiedono circa 2 GiB di KV; lo
stato ricorrente DeltaNet e' circa 145 MiB, oltre a scratch e modello mmap da
circa 12 GB. Il contesto dichiarato di 262144 non e' un target realistico sul
Mac da 24 GB: il primo supporto deve essere validato a 8k, 16k e 32k, con
allocazione preventiva e messaggio chiaro quando il budget stimato supera il
working set consigliato dal device.

## Definition of done

Il supporto e' completato quando:

- il modello denso viene riconosciuto senza flag speciali e il layer MTP non
  entra accidentalmente nel pass standard;
- CPU e Metal non hanno formati tensoriali mancanti per questa GGUF;
- fixture logits e greedy concordano con `llama.cpp` entro tolleranze definite;
- prefill, decode, session checkpoint e agente superano test dedicati;
- la suite del modello MoE resta verde;
- memoria e prestazioni sono documentate con comandi riproducibili;
- README dichiara esplicitamente entrambi i modelli supportati e i limiti
  hardware validati.
