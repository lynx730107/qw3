# M5: due colonne per thread nel decode GDN (2026-10-10)

Tre prove private assegnano a ciascun thread le colonne `j` e `j+64`, con
64 thread per testa invece dei 128 nativi. Cache di 32/48/64 righe per
colonna; accumulatori scalari separati. Le quattro riduzioni SIMD originali
sono rappresentate da due parziali per gruppo, scritti negli slot 0/1 e 2/3.
La baseline rimane la pipeline cache64 nativa. Il dispatch modificato è
confinato a una copia generata del bridge, senza modifiche alla produzione.

**La prova è scartata prima di misurarne le prestazioni.** Il primo caso,
head dimension 128, tre value heads e due query heads, presenta una
differenza di un ULP nello stato: primo elemento, baseline `0x1.72933ep-8`,
candidato `0x1.72934p-8`. Il test si ferma e non esegue forward temporizzati.
[Prima verifica fallita](measurements/gdn_two_columns_2026-10-10/results.json).

La diagnosi rende esplicite diverse contrazioni FMA. `fma(k, d, prior*g)`
riproduce lo stato nativo in quel caso; `fma(prior, g, k*d)` non lo riproduce.
Rimane una differenza nell'output normalizzato. Sono provati accumulatori
FMA espliciti, temporanei volatile, riduzioni separate e un ciclo scalare
per l'output. Nessuna di queste combinazioni completa il controllo esatto.
Il piccolo errore rispetto al riferimento CPU doppio RMSNorm/gate dallo
stato aggiornato (circa 1,6e-7) non viene usato per dichiarare passato il
confronto bit per bit o giustificare un vantaggio prestazionale.
[Contrazioni FMA](measurements/gdn_two_columns_rounding_2026-10-10/results.json),
[normalizzazione](measurements/gdn_two_columns_inner_2026-10-10/results.json),
[ciclo di output](measurements/gdn_two_columns_tail_2026-10-10/results.json).

Il test è poi esteso con un filtro di variante, per impedire al primo
fallimento di nascondere le altre configurazioni. Tutte e tre le cache
falliscono separatamente, sia con la formula originale sia con FMA esplicita
a destra: in quest'ultima lo stato coincide e il primo output diverso è
all'indice fisico 87, baseline `-0x1.4457e6p-7`, candidato `-0x1.4457e8p-7`.
Si arrestano sul caso a tre teste; quello a sette teste non viene completato.
[Verifiche indipendenti e snapshot riproducibile](measurements/gdn_two_columns_independent_2026-10-10/results.json).

Gli shader provati sono conservati con i rispettivi hash. Le modifiche
diagnostiche agli shader generati vengono ripristinate dopo ogni serie.
Non sono eseguiti benchmark, controlli completi del modello o checkpoint
per questa geometria scartata. La produzione conserva cache64 scalare,
precisione, stato e checkpoint già verificati nella precedente integrazione.

```sh
make qw3-gdn-two-columns-test qw3-gdn-two-columns-bench
QW3_METAL_KV_F16=1 ./qw3-gdn-two-columns-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 1
# Gli ID 2 e 3 verificano separatamente cache48 e cache64; il test fallisce come documentato.
```
