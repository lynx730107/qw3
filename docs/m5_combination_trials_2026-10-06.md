# Prove delle combinazioni M5 — 6 ottobre 2026

Le quattro proposte dell'[esame dei sorgenti](m5_combination_audit_2026-10-06.md)
sono state implementate in diagnostici privati e provate in **nove configurazioni**
sul default nativo packed8 con scratch diretto. **Nessuna offre un nuovo
guadagno ripetibile distinguibile dalla variabilità dei controlli.**
La replica del piccolo segnale positivo di A cambia segno; le combinazioni
più ampie con D peggiorano. Nessuna variante è stata integrata nei sorgenti
di inferenza di produzione: il default resta packed8 con scratch diretto.

Bilancio: **260 campioni prestazionali in 52 blocchi**, nessun campione
scartato; **72 casi numerici completi** e **24 casi isolati con guardie**
passati, questi ultimi anche sulla build dei target Makefile definitivi.

## Varianti e metodo

| Sigla | Modifica | Dispatch eliminati per pp128 |
|---|---|---:|
| A, `residual` | Residuo di output attention + RMSNorm FFN | 64 |
| B, `round` | RMSNorm FFN scrive lo stesso `float(half(y))` nei 19 layer IQ3_S/IQ3_S | 19 |
| A+B, `both` | Entrambe nello stesso kernel di normalizzazione | 83 |
| C, `logits` | 640 somme FP32 di otto input condivise dalla testa Q5_K | Nessuno: aggiunge un dispatch di preparazione |
| A+B+C, `all` | Prima combinazione e testa con somme condivise | 83, più la preparazione C |
| D, `down` | Epilogo residuo dopo la GEMM down staged128 IQ4_XS/IQ3_S | 49 |

I conteggi sono strutturali, non percentuali di velocità. A e B mantengono
ordine delle riduzioni e cast; la normalizzazione legge soltanto gli otto
SIMD sum scritti con 256 thread. C conserva geometria, dot product e riduzione
della testa nativa, e include nel timer il nuovo passaggio di preparazione.
D conserva il ciclo GEMM e aggiunge il residuo all'accumulatore FP32 finale,
con fallback per gli altri 15 layer e per chunk non multipli di 128.

Configurazione delle misure: Apple M5, stesso
`Qwen3.8-27B-UD-Q3_K_XL.gguf`, ctx512, KV F16, 64 layer Metal,
quattro thread, seed1, cinque ripetizioni per blocco, warmup escluso,
60 secondi di pausa prima di ciascun blocco e una sola inferenza GPU attiva.
Ogni campagna carica una volta il modello e usa il normale
`run_llama_style_case` di `qw3_bench.c`, con una sessione nuova per blocco.
I diagnostici precompilano le pipeline prima dei blocchi; non introducono
un ciclo di inferenza o un timer semplificato.

pp128 include la produzione dei logits finali; tg32 a profondità 128
esegue 32 forward ordinari con tutti i logits, senza campionamento.
OFF e candidato usano lo stesso binario. Il guadagno aggregato è
`100 * (tempo medio OFF / tempo medio candidato - 1)`; ogni confronto
vicino usa la media dei due controlli OFF che circondano il candidato.
La dispersione OFF riportata è il rapporto massimo/minimo delle medie
dei blocchi, meno uno. Non è un intervallo di confidenza.

## Identità numerica A/B/C

Sono passati **40 casi completi**: otto ciascuno per A, B, A+B, C e A+B+C.
Prefissi 1, 15, 16, 31, 127, 128, 129 e 159; estensione 128→159;
intero vocabolario finito confrontato bit per bit; KV F16 vivi, tutti gli
stati DeltaNet e convoluzione; salvataggio e caricamento del payload
checkpoint; 32 passi di continuazione per caso, con confronto di ogni
logit a ogni passo e degli stati finali.

I contatori confermano l'attivazione reale dei candidati, compresi i
fallback sui prefissi piccoli: A 542 chiamate nella suite, B 57
arrotondamenti incorporati e C 265 teste quando abilitata.
Hash, snapshot dei sorgenti, build e output grezzi sono negli archivi
[qualità A/B](measurements/combination_quality_2026-10-06/manifest.json)
e [qualità C](measurements/sum8_quality_2026-10-06/manifest.json).

## Campagna pp128 A/B/C

21 blocchi, **105 campioni conservati**, ordine diretto e inverso:

`OFF/A/OFF/B/OFF/AB/OFF/C/OFF/ABC/OFF/ABC/OFF/C/OFF/AB/OFF/B/OFF/A/OFF`.

| Variante | tok/s aggregati | Differenza vs tutti gli OFF | Confronti con OFF vicini |
|---|---:|---:|---:|
| OFF | 157,630 | — | — |
| A | 158,185 | +0,352% | +0,303%; +0,461% |
| B | 159,444 | +1,151% | −0,486%; +1,754% |
| A+B | 157,802 | +0,109% | −1,455%; +0,589% |
| C | 155,928 | −1,079% | −0,374%; −1,277% |
| A+B+C | 157,906 | +0,175% | +1,710%; +0,302% |

Gli OFF variano del **2,938%** tra blocchi. A ha due confronti vicini
positivi, ma l'effetto è piccolo rispetto alla variabilità osservata;
B cambia segno. La combinazione A+B non recupera i presunti guadagni
isolati e ha un confronto negativo. C peggiora entrambi i confronti
prefill. A+B+C non dimostra un vantaggio affidabile da integrare.

La deviazione standard dei singoli tempi, raggruppati per modalità, è
9,708 ms OFF; 3,055 ms A; 6,343 ms B; 10,124 ms AB;
13,332 ms C; 6,011 ms ABC. Comprende sia la variabilità dentro i blocchi
sia quella tra blocchi. Tutti i tempi individuali e le deviazioni dentro
ciascun blocco restano disponibili nell'
[analisi](measurements/combination_campaign_2026-10-06/analysis.json).
Sorgenti e binario sono rimasti invariati durante la campagna.

Archivio completo: [risultati pp128](measurements/combination_campaign_2026-10-06/pp128/results.json),
con stdout/stderr della campagna, dei singoli blocchi e snapshot dei sorgenti.

## Campagna tg32 della testa logits

Sette blocchi alternati `OFF/C/OFF/C/OFF/C/OFF`, **35 campioni conservati**.
OFF 8,659 tok/s; C 8,743 tok/s: **+0,973% aggregato**.
I confronti vicini sono **−0,877%, +3,378%, +1,664%**, mentre gli OFF
variano del **4,370%**. La deviazione standard dei tempi individuali
è 77,805 ms OFF e 80,760 ms C. Questo risultato non dimostra un
vantaggio ripetibile: il segno cambia e la variabilità supera l'effetto.

Archivio completo: [risultati tg32](measurements/combination_campaign_2026-10-06/tg32_d128/results.json).
Nessun campione è stato escluso; sorgenti e binario invariati.

## Epilogo down e combinazioni

Sono passati altri **32 casi completi**, otto ciascuno per D, A+D,
A+B+D e A+B+C+D, con gli stessi prefissi, checkpoint, confronti bit
per bit e 32 passi di continuazione descritti sopra. Il totale è quindi
**72 casi completi**, senza differenze numeriche. Il contatore down
conferma 147 epiloghi realmente usati per configurazione nella suite.

Sono inoltre passati **24 casi isolati con guardie**:

- D: IQ3_S e IQ4_XS, 128 e 256 token, 1/33/65/5120 righe di output,
  residui nonzero, input invariati, gap tra righe e guardie esterne
  intatti. Confronto bit per bit con GEMM nativa seguita dal normale
  kernel residuo, 16 casi.
- C: input 256 e 5120, 1/3/65/248320 righe, 32 o tutte le 640 somme
  parziali confrontate con la somma sequenziale FP32 di otto addendi;
  testa Q5_K confrontata con il kernel originale, input e guardie
  invariati, otto casi.

Archivi: [qualità down](measurements/epilogue_quality_2026-10-06/manifest.json)
e [matrici e guardie](measurements/combination_matrix_quality_2026-10-06/manifest.json).
Tutti i sorgenti numerici e i binari sono rimasti invariati durante queste prove.
I 24 casi isolati sono stati nuovamente superati sui
[target Makefile](measurements/combination_matrix_quality_2026-10-06/make_targets/manifest.json)
finali, compilati senza warning.

Campagna D completata: 17 blocchi e **85 campioni conservati**, ordine
`OFF/D/OFF/AD/OFF/ABD/OFF/ABCD/OFF/ABCD/OFF/ABD/OFF/AD/OFF/D/OFF`.

| Variante | tok/s aggregati | Differenza vs tutti gli OFF | Confronti con OFF vicini |
|---|---:|---:|---:|
| OFF | 158,034 | — | — |
| D | 157,876 | −0,101% | −1,357%; +0,543% |
| A+D | 158,010 | −0,015% | −1,111%; +0,872% |
| A+B+D | 155,621 | −1,527% | −0,907%; −2,419% |
| A+B+C+D | 156,212 | −1,153% | −1,078%; −0,039% |

Dispersione OFF **3,147%**. D e A+D cambiano segno tra i confronti
vicini e non mostrano un vantaggio aggregato. Le due combinazioni più
ampie sono negative in entrambi gli ordini. Nessuna variante D viene
integrata. Anche questa campagna conserva tutti i campioni, con
sorgenti e binario invariati.

Archivio completo: [risultati epilogo pp128](measurements/combination_campaign_2026-10-06/epilogue_pp128/results.json).
Le deviazioni dei singoli campioni e dentro ciascun blocco sono nell'
[analisi comune](measurements/combination_campaign_2026-10-06/analysis.json).

## Replica limitata di A

A era l'unico candidato prefill della prima campagna con due confronti
vicini positivi, +0,303% e +0,461%. Per verificare quel piccolo segnale
è stata eseguita una sola replica di sette blocchi
`OFF/A/OFF/A/OFF/A/OFF`, con **35 campioni conservati** e lo stesso protocollo.

OFF **156,060 tok/s**, A **154,341 tok/s**: **−1,102% aggregato**.
Confronti vicini **+0,524%, −0,652%, −2,952%**; dispersione OFF
**1,135%**. Deviazione standard dei tempi individuali: 5,672 ms OFF e
11,447 ms A. Non è stato eliminato il blocco più lento: resta negli
output e nella media. Il segno cambia e il vantaggio iniziale non si
ripete in modo stabile. Anche A resta diagnostica.

Archivio completo: [replica residuo/RMSNorm](measurements/combination_campaign_2026-10-06/residual_repeat_pp128/results.json).
Sorgenti e binario invariati durante la replica.

## Decisione e riproduzione

Si conserva il default nativo packed8 + scratch diretto. Non si sommano
le percentuali delle varianti isolate e non si considera il superamento
occasionale di 160 tok/s una prova del beneficio. Non essendoci un candidato
da promuovere, non è stata introdotta un'integrazione nativa né attribuito
un nuovo risultato al binario di produzione.

Gli hash dei dieci sorgenti di produzione verificati coincidono con quelli
delle regressioni native già passate il 5 ottobre; la
[verifica dei sorgenti](measurements/combination_campaign_2026-10-06/production_check.json)
e l'[esito strutturato](measurements/combination_campaign_2026-10-06/outcome.json)
documentano lo stato finale. Le modifiche del lavoro precedente e dell'utente
sono state conservate. Sono stati aggiunti soltanto diagnostici, strumenti,
target di build e documentazione.

Target disponibili su Darwin/Metal:

- `qw3-combination-test`: qualità A (`residual`), B (`round`) o A+B (`both`).
- `qw3-sum8-test`: qualità C (`logits`) o A+B+C (`all`).
- `qw3-epilogue-test`: qualità D (`down`), A+D (`residual_down`),
  A+B+D (`both_down`) o A+B+C+D (`all_down`).
- `qw3-sum8-matrix-test` e `qw3-epilogue-matrix-test`: prove isolate con guardie.
- `qw3-combination-campaign` e `qw3-epilogue-campaign`: campagne con un motore
  caricato, casi benchmark nativi e controlli alternati.

I runner `tools/run_combination_campaign.py` e
`tools/run_epilogue_campaign.py` archiviano automaticamente i campioni,
i contatori di attivazione, hash e sorgenti. Richiedono una directory
di output nuova e applicano cinque ripetizioni e 60 secondi di pausa.
Esempio di riproduzione della replica A, da eseguire a GPU libera:

```sh
make qw3-combination-campaign
python3 tools/run_combination_campaign.py \
  --output /tmp/qw3-new-residual-repeat \
  --job pp128 --order off,residual,off,residual,off,residual,off
```

I target Makefile delle matrici sono stati aggiunti dopo le campagne cronometrate;
le copie del Makefile negli snapshot conservano esattamente la versione usata
allora. Questa aggiunta non ha modificato flag, kernel o sorgenti numerici.

## Limiti dell'interpretazione

Sono misure end-to-end di questo modello e di questa configurazione,
non una stima universale della velocità dei kernel. La macchina aveva
swap occupato anche con una sola inferenza; i manifest registrano lo
swap prima e dopo ogni blocco. I controlli alternati mostrano la
variabilità reale della campagna; non è stata corretta o rimossa.
Non è stato eseguito un nuovo confronto con llama.cpp, e non si
attribuisce alcun guadagno ai vecchi picchi di 160,7–160,9 tok/s.
