# Decode IQ4_XS del 27B su M5

Il benchmark usa Qwen3.8-27B-UD-Q3_K_XL, Metal, 64 layer, KV F16 e quattro thread. Entrambe le opzioni sperimentali QW3_METAL_IQ3_NIBBLE e QW3_METAL_DENSE_PARTIAL_TILES sono a 1. Il comando qw3-bench usa --llama-style -p 0 -d 128 -n 32 -r 2 --seed 1 --ctx-alloc 256. Ogni processo effettua il proprio riscaldamento. I tempi escludono il prefill. Ogni valore è la media di due ripetizioni sintetiche.

| Ordine | Variante | tok/s |
|---|---|---:|
| 0 | Originale | 9,25 |
| 1 | Limite 64 thread | 9,23 |
| 2 | Controllo righe fuori dal ciclo | 9,07 |
| 3 | Controllo righe fuori dal ciclo | 9,19 |
| 4 | Limite 64 thread | 8,97 |
| 5 | Originale | 9,01 |

Le due varianti interessano i tre matvec IQ4_XS singolo, gate/up e add_x0. Il limite a 64 thread è un attributo Metal applicato prima della dichiarazione. Un primo tentativo con l’attributo dopo i parametri ha prodotto un errore di compilazione. Il controllo delle righe seleziona il ciclo senza verifiche solo quando entrambe le righe del gruppo SIMD sono valide. I due controlli differiscono del 2,6%. Le medie descrittive sono 9,13, 9,10 e 9,13 tok/s per originale, limite e righe. Entrambi i candidati rimangono sperimentali.

Per replicare, copiare la directory metal e applicare separatamente [hint.diff](hint.diff) o [rows.diff](rows.diff) al suo file qw3_core_linear.metal. Impostare QW3_METAL_KERNEL_DIR alla directory risultante. [Documentazione Apple](https://developer.apple.com/documentation/metal/mtlcomputepipelinedescriptor/maxtotalthreadsperthreadgroup)

Il [controllo numerico](check.c) parte dal token 42 e confronta i 16 vocabolari completi lungo una continuazione greedy. Metal API Validation attiva, processo separato per ciascuna variante. I tre file da 15.892.480 byte coincidono con cmp. Questo controllo copre un solo prefisso. [Hash SHA-256](hashes.txt)
