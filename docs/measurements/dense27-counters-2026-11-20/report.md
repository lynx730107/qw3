# Generazione del 27B denso su Apple M5 a profondità 1024

Il modello è Qwen3.8-27B-UD-Q3_K_XL. Le prove usano Metal su 64 strati, KV F16, IQ3_NIBBLE e DENSE_PARTIAL_TILES. Il profiler ricompilato sul checkout corrente misura due passi (offset 1 e 9) dopo il prefisso e verifica tutti i logits su 16 passi ripristinando lo stesso checkpoint. Entrambe le esecuzioni producono gli stessi bit. I timestamp richiedono encoder separati e alterano il tempo totale. [Profilo GPU e verifica](profile.log) · [Harness](../dense27_m5_2026-11-18/profile.inc)

| Proiezioni | Matrici per token | GB decimali di pesi | Primo passo ms | Secondo passo ms | GB/s effettivi sui pesi |
|---|---:|---:|---:|---:|---:|
| IQ3_S | 111 | 3,528 | 30,19 | 31,84 | 113,75 |
| IQ4_XS | 156 | 4,972 | 40,65 | 42,85 | 119,08 |

La banda effettiva divide la dimensione delle matrici per la media dei due tempi GPU. I dispatch doppi contano per due matrici. Il totale corrisponde all’inventario del GGUF. Il dispositivo espone GPUTimestamp al confine di stage. La percentuale di stallo e i byte effettivamente letti dalla DRAM restano indeterminati. [Inventario](../projection_cost_2026-10-04/inventory.json) · [Contatori del dispositivo](device.log) · [Enumerazione dei contatori Metal](https://developer.apple.com/documentation/metal/confirming-which-counters-and-counter-sets-a-gpu-supports)

Il kernel di controllo legge sequenzialmente buffer sintetici, con otto letture ordinate per comando, tre riscaldamenti e sei campioni per taglia. Le mediane per 128, 256, 512 e 1024 MiB sono rispettivamente 121,27, 121,38, 123,83 e 124,40 GB/s. Assumendo una lettura per token dei 12,238 GB di pesi attivi, 10 tok/s richiedono 122,38 GB/s di soli pesi. Alla velocità osservata sul buffer da 1 GiB la sola lettura equivalente dura 98,4 ms. Questa stima dipende dalla cache e dalla sequenza degli accessi. [Campioni di banda](bandwidth.json) · [Kernel di controllo](stream.m)

La generazione con qw3-bench senza profiler, seed 1, 4 thread, ctx 1152, warmup e tre ripetizioni da 32 token sintetici, misura 9,09 tok/s con deviazione standard 0,05. I 3518,68 ms medi per 32 token equivalgono a 109,96 ms/token. Per raggiungere 10 tok/s servono 9,96 ms/token in meno. Portare le sole proiezioni IQ3_S e IQ4_XS alla banda del controllo da 1 GiB equivarrebbe a circa 4,4 ms/token nel profilo con encoder separati. Il trasferimento di quel guadagno al percorso nativo resta da verificare. [Benchmark nativo](native.log)
