# Generazione del 27B denso a profondità 1024 su Apple M5

Qwen3.8-27B-UD-Q3_K_XL, 64 strati Metal, KV F16, quattro thread, ctx1152. QW3_METAL_IQ3_NIBBLE e QW3_METAL_DENSE_PARTIAL_TILES sono attivi in tutte le modalità. Il confronto misura il tempo necessario per valutare 32 token sintetici successivi al prefisso di 1.024 token, con logits completi a ogni passo. Seed 1, stesso generatore e sequenza di qw3-bench. Prefill, checkpoint, selezione dei token, rendering e tool sono fuori dal timer. [Hash dei sorgenti e dei binari](identity.json)

In tre sessioni indipendenti, il [runner](paired.m) crea un checkpoint del prefisso con i kernel originali. Per ogni variante ricarica quel checkpoint e valuta gli stessi 32 token. La passata numerica confronta tutti i vocabolari, controlla valori finiti e argmax e riscalda le pipeline. Seguono sei blocchi temporizzati, ciascuno con le tre modalità in una diversa permutazione. La velocità aggregata è 192 token divisi per la somma dei tempi di ciascuna modalità. Il controllo di ogni serie è locale. La prima esecuzione IQ3 è stata interrotta durante l’avvio e ripetuta interamente. [Log dell’interruzione](paired3-interrupted.log)

Le due varianti IQ4 interessano i matvec singolo, gate/up e add_x0. La prima recupera da quattro parole immediate i 16 valori signed int8 della lookup. La seconda assegna un valore a ogni lane e usa simd_shuffle. Il controllo usa la tabella FP32 in memoria threadgroup. [Lookup packed](packed.diff) · [Scambio fra lane](shuffle.diff)

Due varianti del matvec singolo IQ3 spostano la scala dopo la somma di 32 termini. Una mantiene gli accumulatori scalari, l'altra usa dot su gruppi di quattro attivazioni. Altre due precomputano i nibble per le 512 voci della griglia e incorporano 1 KiB di costanti nello shader. La prima copia la tabella nella memoria del threadgroup, la seconda la legge direttamente dallo spazio constant. Tutte usano gli stessi pesi del GGUF e la stessa geometria del dispatch. [Scala finale](scalar.diff) · [Dot vettoriale](dot.diff) · [Tabella condivisa](const-stage.diff) · [Lettura diretta](const-read.diff)

Il benchmark pubblico separato per IQ4 packed usa quattro processi in ordine OFF/ON/ON/OFF e due ripetizioni per processo. Medie 9,41 / 8,37 / 8,23 / 9,20 tok/s. Entrambe le coppie sono più lente con la variante attiva. [Primo controllo](0-A.log) · [Prima variante](1-B.log) · [Seconda variante](2-B.log) · [Ultimo controllo](3-A.log)

## Risultati

| Variante | Controllo tok/s | Candidato tok/s | Delta | Blocchi favorevoli | Errore massimo |
|---|---:|---:|---:|---:|---:|
| IQ4 lookup packed | 9.014 | 7.774 | -13.753% | 0/6 | 0.000242 |
| IQ4 simd_shuffle | 9.014 | 6.233 | -30.849% | 0/6 | 0.000000 |
| IQ3 scala finale | 9.194 | 9.120 | -0.809% | 1/6 | 0.000257 |
| IQ3 dot vettoriale | 9.194 | 9.147 | -0.518% | 2/6 | 0.000273 |
| IQ3 tabella precalcolata condivisa | 8.773 | 8.756 | -0.195% | 3/6 | 0.000000 |
| IQ3 lettura diretta costante | 8.773 | 8.750 | -0.260% | 3/6 | 0.000000 |

Tutte le sei varianti concordano sui 32 argmax dei logits confrontati con il controllo. La velocità di 10 tok/s resta da dimostrare. Le modifiche sono confinate ai diagnostici. [Tempi e logits IQ4](paired.log) · [Aritmetica IQ3](paired3.log) · [Tabella IQ3](grid.log) · [Riepilogo numerico](results.json)
