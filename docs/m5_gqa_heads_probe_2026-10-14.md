# GQA: probe a due/una testa con il criterio numerico v2

Su richiesta dell'utente il criterio generale passa dall'identità dei bit
alla correttezza numerica e, come gate finale, al comportamento accettabile
dell'agente. Il [protocollo](optimization_acceptance.md) conserva i vincoli
di memoria e richiede un roundtrip esatto della stessa variante per i
checkpoint. La baseline corrente resta il kernel pubblico a tre teste.

I due candidati privati assegnano due o una testa a ciascun gruppo: 12 o
24 gruppi, contro gli otto della baseline pubblica. Si intercetta solo il
percorso già selezionato per M5, 24 teste query, quattro KV, dimensione
256 e cache F16. Tutte le altre forme conservano il fallback pubblico.
Questa prova non rende valido il precedente candidato generico a una
testa, che differiva in F32/dimensione32. Non si aggiungono buffer di pesi
o sessione, copie dei pesi o scratch. Restano 256 byte dinamici per gruppo.
Si compilano oggetti di pipeline e temporanei CPU prima di caricare i
pesi; la RAM totale e di picco non sono misurate.

## Screening numerico verificato

Entrambi passano 432 casi, di cui 18 selezionano effettivamente il candidato
e 414 verificano i fallback. Si controllano valori finiti, NMSE ≤ 1e-7,
input e guardie esatti, oltre a un gruppo extra oltre il limite. Le
espressioni FP32 e le barriere si ripristinano esattamente rimuovendo la
partizione e i limiti degli array/cicli. Non si cambia precisione.

Un solo engine/sessione ctx256 viene poi riutilizzato con oracolo su disco:
prefisso sintetico128 seed1 e 32 passi alimentati con i token della
baseline, così da confrontare logits sugli stessi contesti. Sono 33
vocabolari completi confrontati; history e posizione restano esatte.

| Variante | Matrici: massimo errore assoluto | Logits RMSE | Logits massimo assoluto | Cambi di argmax |
| --- | ---: | ---: | ---: | ---: |
| Due teste | 0 | 0 | 0 | 0/33 |
| Una testa | 4,7684e-7 | 4,0212e-5 | 8,5481e-4 | 0/33 |

La variante a una testa cambia 24.676 valori nelle fixture e 7.888.325
bit pattern di logits nel confronto completo; non è bit-exact. La
NMSE aggregata dello stato canonico è 3,2016e-10. Passa le soglie v2 su
ogni vocabolario e blocco di stato confrontato. I conteggi per ciascun
seguito sono 512 selezioni, zero fallback; baseline zero.

La build è senza warning e Metal API Validation è attiva. Il contratto
delle nuove metriche accetta un esempio di due ULP e gli zeri con segno,
rifiutando un errore sostanziale e NaN. Sorgenti, snapshot, binari,
eseguibili di produzione e stat del GGUF sono verificati.

Questo è solo uno screening numerico breve. Non prova qualità dell'agente,
perplexity, correttezza su contesto lungo o vantaggio di velocità. Nessun
checkpoint è ancora esercitato in questo screening. Entrambi i manifest
mantengono `agent_behavior_gate_passed=false` e `adoption_permitted=false`.
Nessuno dei candidati è attivato in produzione.

[Manifest](measurements/gqa_heads_probe_2026-10-14/numerical_v2/manifest.json)
e [verifica](measurements/gqa_heads_probe_2026-10-14/numerical_v2/verification.json).

La prima esecuzione resta in `numerical/`: il tag per caso `matrix_exact`
era stato ereditato dal vecchio test e non descriveva il nuovo criterio.
I record `matrix_numeric` riportano già le differenze reali e il riepilogo
era `finite_numerically_valid`. Il tag viene corretto in
`matrix_numeric_valid` e si riesegue l'intero screening in `numerical_v2/`,
con binari distinti, conservando integralmente la prima esecuzione.
