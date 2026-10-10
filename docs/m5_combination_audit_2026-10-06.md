# Combinazioni ancora da verificare sul default M5 — 6 ottobre 2026

**Seguito operativo:** le proposte di questo esame iniziale sono state
implementate in diagnostici privati. Le verifiche numeriche e le nuove
campagne sono raccolte nel [report delle prove](m5_combination_trials_2026-10-06.md).
Il testo sotto conserva le ipotesi e le evidenze disponibili al momento
dell'esame, prima dell'esecuzione.

L'esame dei sorgenti attuali e dei report archiviati individua tre nuove
microottimizzazioni da valutare e un intervento più ampio di seconda priorità.
Questa è una verifica del codice e delle prove precedenti: non sono stati
eseguiti nuovi benchmark e non si attribuisce un guadagno alle idee seguenti.
Il controllo resta il default packed8 con scratch diretto, con lo stesso GGUF,
KV F16 e aritmetica attuale.

## Prima combinazione: residuo, RMSNorm e arrotondamento degli input FFN

Nel prefill denso, tutti i 64 layer completamente batched eseguono:

1. Proiezione di output dell'attenzione in `denseBatchDown`.
2. Addizione a `prefillX0`, in un kernel distinto.
3. RMSNorm da `prefillX0` a `prefillX1`, in un altro kernel.
4. Nei 19 layer con gate e up IQ3_S entrambi, arrotondamento di `prefillX1`
   a F16 e ritorno a float prima delle due proiezioni staged128.

I punti sono visibili in [output + residuo](../qw3_metal.m:5511),
[normalizzazione FFN GQA](../qw3.c:15112),
[normalizzazione FFN DeltaNet](../qw3.c:15174) e
[FFN misto e arrotondamento](../qw3_metal.m:5032).
Il profilo archiviato del 4 ottobre conta 128 normalizzazioni, 128 addizioni
residue e 19 arrotondamenti per pp128. Le prime due quantità comprendono
anche le altre normalizzazioni e l'addizione residua finale FFN: l'intervento
qui proposto riguarda soltanto 64 addizioni e 64 normalizzazioni post-attention.
L'inventario del modello conferma i 19 layer IQ3_S/IQ3_S.

**A — residuo + RMSNorm:** un kernel aggiorna `prefillX0` e produce `prefillX1`
normalizzato, mantenendo separata la proiezione quantizzata. È simile al
[kernel residuo/RMSNorm già esistente](../metal/qw3_core_deltanet.metal:609),
ma il wrapper disponibile legge lo scratch del percorso MoE; per il dense
va adattato alla sorgente `denseBatchDown`. Il percorso dense corrente non
usa questa fusione. Risparmio strutturale previsto: 64 dispatch per pp128.

**B — RMSNorm + arrotondamento F16:** il kernel di normalizzazione può
scrivere `float(half(valore))` nei soli layer che oggi usano
`qw3_round_f16_inplace`. Si elimina quel passaggio senza cambiare la precisione
dei buffer, la GEMM o la politica delle coppie IQ3_S. Risparmio strutturale
previsto: altri 19 dispatch per pp128.

**A+B** è la prima combinazione da provare: fino a **83 dispatch in meno**
su questo pp128, più alcune letture/scritture intermedie evitate. Questo conteggio
non equivale a una percentuale di velocità. A e B condividono la normalizzazione,
quindi non è lecito sommare eventuali guadagni misurati separatamente.
Non risultano campagne di A, B o A+B negli archivi esaminati.

Condizioni numeriche: stessa geometria e ordine delle riduzioni RMSNorm,
stesso arrotondamento del residuo e stesso cast F16 dopo la moltiplicazione
per il peso. La nuova riduzione deve leggere solo i SIMD sum scritti: con
256 thread sono otto, non tutti i 32 elementi di scratch. L'identità bit per
bit va verificata prima delle misure; la presenza del vecchio kernel fuso
non la dimostra automaticamente.

## Seconda combinazione: A+B e lavoro condiviso nella testa Q5_K dei logits

**C — precomputare le somme degli input:**
[il kernel Q5_K attuale](../metal/qw3_core_linear.metal:1013) ricostruisce
`sumx` come quattro somme di otto attivazioni per blocco/lane, per ogni riga
del vocabolario. Queste somme dipendono dal solo vettore di input normalizzato,
non dai pesi della riga. Sul LM head 5120 → 248320 possono essere calcolate
una volta: 640 somme FP32, 2560 byte di scratch derivato.

La prima variante dovrebbe cambiare soltanto la provenienza di `sumx`,
conservando carichi dei quanti, accumulatori, geometria e riduzione originale.
La somma precomputata deve rispettare l'ordine degli otto addendi, partendo
da zero, e coincidere bit per bit con quella originale. Si deve includere
nel timer il dispatch di preparazione. Solo dopo un vantaggio verificato
avrebbe senso valutare l'accorpamento con la normalizzazione finale.

Le [vecchie prove Q5_K](decode_block_cost_2026-10-01.md) cambiavano righe
per SIMD e numero di gruppi e non mostravano un vantaggio credibile. Le
varianti conservate in `tests/q5_decode_candidates.metal` riusano `sumx`
entro poche righe, ma continuano a ricalcolarlo per ogni workgroup: questa
proposta condivide le somme su tutto il vocabolario. Non risulta misurata.

C si può combinare con A+B perché interviene sulla testa finale. Va misurata
in **tg32 con tutti i logits**, e separatamente in pp128 dove la testa viene
eseguita una sola volta. Il limite possibile è la lettura degli 874 MB di pesi
del LM head: risparmiare somme non riduce quei byte. Il piccolo scratch non
fa parte dello stato canonico o del formato checkpoint. Non si propone di
omettere logits o cambiare campionamento/vocabolario.

## Intervento successivo: epilogo della proiezione down FFN

**D — down GEMM + residuo:** oggi
[il FFN batched](../qw3_metal.m:5032) scrive `denseBatchDown`, poi un kernel
legge quel buffer e aggiorna `prefillX0`. Una variante staged128 può aggiungere
il residuo dopo l'accumulo finale e scrivere direttamente `prefillX0`.
È distinta dalla fusione A, che modifica il residuo dell'attenzione.

Il primo sottoinsieme ragionevole è IQ4_XS + IQ3_S: nell'inventario sono
32 + 17 proiezioni down, quindi fino a 49 dispatch residui in meno per pp128.
Gli altri 15 layer devono conservare il fallback. L'accumulo GEMM, il passaggio
finale a FP32 e l'addizione residua devono mantenere il comportamento corrente;
il semplice fatto di portarli nello stesso shader non garantisce gli stessi bit.
L'epilogo potrebbe anche peggiorare il costo del kernel dominante.

D può essere abbinata ad A+B, ma richiede una modifica ai kernel Tensor più
ampia rispetto alle prime fusioni. Non risulta testata nella forma staged128
dense. La fusione SwiGLU nel decode già scartata è un intervento diverso.

## Vecchie combinazioni che non hanno nuova evidenza favorevole

- Packed8 + pipeline DeltaNet è già stato provato: +0,45% aggregato, con
  confronti vicini +0,46% e +0,62%; i miglioramenti isolati non si sommavano.
- Q3_K metadata sul nuovo default perde 0,327%; IQ3_S metadata dà un movimento
  di +0,198% comparabile alla dispersione OFF di 0,222%.
- Alpha/beta prefill e decode danno rispettivamente −1,486% e −1,752% nelle
  campagne archiviate.
- Conv/QK norm, nella replica del 6 ottobre, dà +0,049% aggregato con
  dispersione OFF 0,187% e un confronto −0,343%. Accoppiarla a un altro
  candidato non dimostrerebbe che questa componente aiuta.
- Le fusioni SwiGLU nel decode, il semplice aumento dei SIMD group, i vecchi
  layout IQ4 e le geometrie Q5_K non offrono un guadagno verificato da sommare.

Le fonti sono i report [DeltaNet](m5_deltanet_prefill_2026-10-04.md),
[riesame](m5_revisited_optimizations_2026-10-05.md),
[microfusioni](m5_micro_fusions_2026-10-05.md),
[decode](m5_decode_2026-10-01.md) e [layout](m5_layout_staging_2026-10-02.md).

## Ordine di prova e attribuzione del risultato

Prima costruire A e B in un diagnostico privato; controllare **OFF, A, B, A+B**
con lo stesso binario. A+B deve essere confrontata anche con la migliore
variante singola, oltre che con OFF. Un ordine iniziale è
OFF/A/OFF/B/OFF/A+B/OFF, invertendo A/B in una replica se emerge un effetto.
C va poi confrontata da sola e insieme al miglior prefill; D viene dopo.

Per ciascuna configurazione modificata: intero vocabolario finito bit per bit,
KV F16 vivi, DeltaNet/conv, prefissi 1,15,16,31,127,128,129,159, estensione
128→159, checkpoint salvato/caricato e 32 passi di continuazione. Per C servono
anche controlli isolati dei 640 parziali e delle righe/tail Q5_K; per D controlli
di output, residuo nonzero e guardie dei buffer.

Prestazioni: stesso Qwen3.8-27B-UD-Q3_K_XL, ctx512, 64 layer Metal, KV F16,
quattro thread, seed1; warmup escluso, cinque ripetizioni per blocco, controlli
vicini, 60 secondi fra blocchi e una sola inferenza GPU. pp128 per A/B/D,
tg32 a profondità 128 e pp128 separati per C. Tutti i campioni e i risultati
sfavorevoli vanno conservati. Nessuna promozione da un microbenchmark isolato
o dalla vecchia soglia automatica del 5%: serve un vantaggio ripetibile
distinguibile dalla variabilità e poi una replica sulla build di produzione.

Il [manifest dell'esame](measurements/combination_audit_2026-10-06/manifest.json)
registra hash dei sorgenti e delle evidenze, conteggi e stato delle proposte.
Questo esame lascia invariati sorgenti di produzione e diagnostici esistenti.
