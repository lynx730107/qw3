# Vecchie ottimizzazioni sul nuovo default M5

Questa prova segue l'[integrazione packed8 e scratch diretto](m5_packed8_scratch_prefill_2026-10-05.md).
L'utente chiede di riesaminare le idee precedenti per recuperare altri punti
percentuali. Non si reinterpreta il vecchio +1,5% Q3_K come un guadagno certo:
era un solo blocco candidato. Si verifica anche IQ3_S separatamente da IQ4_XS,
che nella vecchia prova aggregata penalizzava la variante.

## Metodo

Stesso Qwen3.8-27B-UD-Q3_K_XL GGUF, Apple M5 24 GB, 64 layer Metal,
KV F16, ctx512, quattro thread, pp128 sintetico seed1, nessun campionamento.
Una sola inferenza GPU attiva. Ogni candidato usa cinque blocchi
OFF/candidato/OFF/candidato/OFF, tre ripetizioni, pausa di 40 secondi;
compilazione e warmup esclusi. Il controllo usa già il nuovo default con
packed8 e proiezioni dirette. Non si confrontano velocità assolute tra
serie con binari diversi.

I nuovi target `qw3-revisit-prefill-test` e `qw3-revisit-prefill-bench`
riusano i vecchi shader di prova, mantenendo attivo il nuovo percorso di
produzione. Il vecchio diagnostico resta riproducibile con il suo controllo
compatto. Una nuova selezione privata isola solo IQ3_S: il contatore IQ4_XS
modificato deve rimanere zero. Nessuna variante dei metadati entra nel default.

La [verifica numerica](measurements/revisit_prefill_quality_2026-10-05/manifest.json)
passa per entrambi i candidati: intero vocabolario finito e bit-identico,
KV F16 vivi, DeltaNet e conv byte per byte, payload salvato/caricato e 32 passi
di continuazione con tutti i logits a ogni passo. Prefissi 1,15,16,31,127,128,
129,159; 128 viene esteso a 159. Q3_K esegue 33 dispatch modificati; IQ3_S
333, con zero Q3_K e zero IQ4_XS modificati nella sua prova isolata.

## Metadati: nessun recupero convincente

| Vecchia idea | Variazione pp128 sul nuovo default | Due confronti vicini | Dispersione OFF |
|---|---:|---:|---:|
| [Q3_K](measurements/revisit_q3_prefill_2026-10-05/results.json) | −0,327% | −0,288% / −0,417% | 0,237% |
| [IQ3_S isolato](measurements/revisit_iq3_prefill_2026-10-05/results.json) | +0,198% | +0,217% / +0,165% | 0,222% |

Q3_K misura 157,596 → 157,080 tok/s: il vecchio +1,5% non si replica e i due
confronti hanno segno negativo. Non viene riattivato.

IQ3_S mostra circa 1,6 ms di vantaggio su un prefill di circa 809 ms. La
variazione è comparabile alla dispersione OFF e inferiore alle deviazioni
standard dei campioni nei blocchi (circa 2–4 ms). Non si dichiara un recupero
di punti percentuali né una significatività statistica da tre ripetizioni.
La variante resta privata. La combinazione con il Q3_K peggiorativo non viene
promossa né trattata come somma automatica di effetti.

Queste prove cronometrano soltanto pp128, la fase modificata dai kernel
staged128. I matvec di decode rimangono identici; non si producono nuovi
numeri tg32 da attribuire a cambiamenti nella preparazione non cronometrata.

## Controllo dell'interazione tra packed8 e copie

Una terza prova confronta il default combinato con packed8 da solo.
`qw3-copy-control-bench` mantiene sempre `QW3_METAL_DENSE_PREFILL_FAST=1`
e il vero kernel packed8 di produzione. Un wrapper privato della sola
funzione di disponibilità strided può ripristinare staging e compattazione;
non cambia i kernel numerici, le proiezioni FFN o la selezione DeltaNet.
I contatori richiedono proiezioni/output diretti nel controllo combinato e
zero dispatch diretti quando le copie sono ripristinate.

La [qualità del confronto](measurements/copy_control_quality_2026-10-05/manifest.json)
passa gli stessi otto casi bit per bit, inclusi checkpoint e continuazione.
Il contatore aggregato 1152/512/1664 appartiene alla sessione combinata;
il confronto usa due sessioni indipendenti e packed8 su entrambe.

La [misura alternata](measurements/copy_control_perf_2026-10-05/results.json)
dà 157,433 tok/s per la combinazione e 155,547 per packed8 con copie
ripristinate: −1,197%. I due confronti vicini danno −1,328% e −0,928%,
con dispersione OFF 0,472%. Si mantiene il default combinato. Questa è una
comparazione diretta tra componenti nello stesso binario; confrontare il
vecchio +3% packed8 con il nuovo +2,07% in serie diverse avrebbe portato
a una conclusione ingiustificata.

Il risultato del riesame è negativo per un'ulteriore promozione: Q3_K perde,
IQ3_S dà solo un movimento marginale, e ripristinare le copie peggiora.
Resta integrato il miglioramento di prefill già verificato, senza cambiare
GGUF, KV, decode, campionamento o comportamento dell'agente.

## Riproduzione

```sh
make qw3-revisit-prefill-test qw3-revisit-prefill-bench
QW3_METAL_KV_F16=1 QW3_METAL_KV_Q8_0=0 caffeinate -disu \
  ./qw3-revisit-prefill-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf iq3_metadata
python3 tools/bench_qw3_candidates.py --output /tmp/qw3-revisit-iq3 \
  --binary ./qw3-revisit-prefill-bench --jobs pp128 \
  --order off,candidate,off,candidate,off \
  --candidate-env QW3_PRIVATE_IQ3_METADATA=1 --repetitions 3 --cooldown 40
```

Per Q3_K usare `q3_metadata` nel test e `QW3_PRIVATE_Q3_METADATA=1` nel
benchmark. Per il controllo delle copie usare `qw3-copy-control-test` con
`all` e `qw3-copy-control-bench` con `QW3_PRIVATE_RESTORE_COPIES=1`.
I manifest conservano hash, comandi, configurazione e swap prima/dopo i
processi; lo swap è presente. Sono confronti brevi con warmup, non misure
termiche sostenute o di contesto lungo.
