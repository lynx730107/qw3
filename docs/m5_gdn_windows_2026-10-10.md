# M5: estensione e posizione della cache DeltaNet (2026-10-10)

La baseline è il kernel nativo cache64 già adottato in
`1a4e9e0`, non il precedente kernel senza cache. Cinque prove private
mantengono la dimensione della testa 128, l'ordine delle somme FP32,
aggiornamento in-place e normalizzazione. Cambiano solo i valori dello
stato conservati nell'array locale, riusati nel secondo passaggio:

| ID | Variante | Righe conservate, estremi inclusi |
| --- | --- | --- |
| baseline | native first64 | 0–63 |
| 1 | first48 | 0–47 |
| 2 | first80 | 0–79 |
| 3 | first96 | 0–95 |
| 4 | middle64 | 32–95 |
| 5 | last64 | 64–127 |

Il wrapper seleziona le pipeline solo per sessioni dense già preparate e
dimensione 128, con fallback originale per le altre forme. Il selettore
nativo viene disabilitato nel processo diagnostico, poi il wrapper seleziona
esplicitamente la vera pipeline cache64 nativa per la baseline. I contatori
separati verificano dispatch candidati e cache64 di controllo.

Dieci casi di matrice, cinque varianti per 3/7 value heads e 2 query heads,
passano stato e output finiti bit per bit contro la cache64 nativa.
Offset non nulli, guardie di stato in-place e output, gruppo extra fuori
range e tutti gli input immutati. Il riferimento CPU doppio per RMSNorm e
gate, a partire dallo stato aggiornato GPU, ha errore massimo 3,814e-7;
non è un riferimento indipendente della ricorrenza completa.

La misura accoppiata usa due sessioni ctx512, prefisso seed 1 di 128 token,
64 layer Metal, KV F16 e quattro thread. Dopo 60 secondi di pausa per
variante, 72 forward greedy AB/BA: otto warmup e 64 temporizzati. Controllo
dell'intero vocabolario fuori dal timer dopo ogni forward, confronto finale
della history e dello stato canonico K/V, DeltaNet e convoluzione.
3456 dispatch per variante e altrettanti di cache64 baseline. I throughput
di due sessioni descrivono gli effetti relativi, non il decode ordinario di
una sessione singola. Nessun campione viene eliminato.
[Archivio](measurements/gdn_windows_2026-10-10/results.json).

```sh
make qw3-gdn-windows-test qw3-gdn-windows-bench
QW3_METAL_KV_F16=1 ./qw3-gdn-windows-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf
QW3_METAL_KV_F16=1 caffeinate -disu ./qw3-gdn-windows-bench ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 1
# Sostituire l'ID per le altre varianti, dopo la pausa.
```

| Variante | Prima serie | Replica | Blocchi positivi, prima / replica |
| --- | ---: | ---: | ---: |
| first48 | −0,126% | — | 4/8 |
| first80 | −0,200% | — | 4/8 |
| first96 | −0,198% | — | 4/8 |
| middle64 | +0,445% | +0,135% | 5/8 / 6/8 |
| last64 | +0,298% | −0,041% | 6/8 / 4/8 |

Middle64 nella prima serie risparmia 0,505±0,904 ms per forward; la
replica 0,153±0,585 ms. Last64 inizialmente 0,342±0,756 ms, poi
−0,047 ms. Gli intervalli descrittivi al 95% usano Student su otto medie
di blocco e non eliminano autocorrelazione o deriva. **Nessuna variante
viene adottata**: i piccoli vantaggi iniziali non sono confermati dalla
replica. Rimane attiva la cache64 nativa, con il guadagno già verificato
nel benchmark ordinario.
[Replica delle due finestre](measurements/gdn_windows_replica_2026-10-10/results.json).

Le cinque prime serie e le due repliche conservano tutti i 448 campioni
accoppiati, cioè 896 durate di forward. Vocabolario completo finito ed
esatto e stato finale canonico passano in tutte le serie; sorgenti e
binari sono invariati. Il controllo completo su otto prefissi e checkpoint
è disponibile nel test (`MODEL all ID`), ma non viene eseguito per queste
varianti scartate. Le verifiche dei checkpoint della cache64 adottata
sono quelle documentate nel report della precedente integrazione nativa.
