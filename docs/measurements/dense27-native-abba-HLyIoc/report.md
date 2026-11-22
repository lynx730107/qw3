# Confronto nativo qw3-bench su Apple M5

Qwen3.8-27B denso, 64 strati Metal, KV F16, quattro thread, contesto allocato 1280, seed 1. Processi sequenziali con warmup, due ripetizioni per fase e 20 secondi di pausa prima di ciascun blocco. Prefill di 128 token sintetici a contesto vuoto. Generazione di 128 token sintetici dopo 1024 token di prefisso preparati fuori dal timer. IQ3_NIBBLE e DENSE_PARTIAL_TILES attivi. [Comandi](run.sh) · [Identità del binario](binary.sha256)

| Blocco | GGUF | Prefill tok/s | Generazione tok/s |
|---:|---|---:|---:|
| 1 | Q3_K_XL | [158.06](1-Q3_K_XL-pp.log) | [9.22](1-Q3_K_XL-tg.log) |
| 2 | IQ3_XXS | [89.93](2-IQ3_XXS-pp.log) | [8.50](2-IQ3_XXS-tg.log) |
| 3 | IQ3_XXS | [72.39](3-IQ3_XXS-pp.log) | [7.10](3-IQ3_XXS-tg.log) |
| 4 | Q3_K_XL | [124.47](4-Q3_K_XL-pp.log) | [6.87](4-Q3_K_XL-tg.log) |

Le medie per blocco sopra vengono da qw3-bench. Il throughput complessivo sotto divide 512 token per il tempo totale delle quattro ripetizioni di ciascuna fase e quantizzazione.

| GGUF | Prefill tok/s | Generazione tok/s | Generazione ms/token |
|---|---:|---:|---:|
| Q3_K_XL | 139.13 | 7.87 | 127.13 |
| IQ3_XXS | 80.22 | 7.69 | 130.03 |

Entrambi i GGUF rallentano tra i rispettivi due blocchi. La dispersione limita la precisione del confronto in generazione.
