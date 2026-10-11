# M5: replica della residenza dei pesi, 2026-10-11

La replica non riproduce il precedente vantaggio apparente del 4,407%.
La residenza esplicita del buffer originale rimane una prova privata e non
viene adottata. Non crea copie dei pesi né buffer di sessione aggiuntivi.

| Misura | OFF | ON |
| --- | ---: | ---: |
| tg32, profondità 128, token/s | 9,18258 | 9,21025 |
| Dispersione delle medie dei blocchi | 1,165% | 0,063% |

Il punto aggregato è +0,301%, ma i confronti locali non confermano un
miglioramento ripetibile. Il tempo risparmiato rispetto alla media dei due
controlli adiacenti è +20,409 / +1,875 / −0,432 ms per 32 token. Il primo
OFF è più lento dei successivi e resta integralmente nel calcolo. Non si
attribuisce una causa a quel rallentamento e non si eliminano campioni.

Sette processi OFF/ON/OFF/ON/OFF/ON/OFF, tre ripetizioni ciascuno e 60
secondi di pausa prima di ogni processo conservano tutti i 21 campioni.
Il benchmark privato include il runner ordinario `qw3_bench.c`: tg32 a
profondità 128, ctx256, KV F16, 64 layer e head Metal, warmup e stream fisso
seed 1 uguale nei due modi. Prefill e richiesta di residenza sono fuori dal
timer. Packed8, K64 prefill e cache64 DeltaNet mantengono i default nativi;
gli override QW3/GGML/MTL vengono prima rimossi. L'opzione thread richiesta
è quattro; non determina il numero effettivo di helper CPU del motore.

ON registra esattamente una richiesta, una vista originale e
13.135.413.248 byte associati al set; OFF registra zero per i tre contatori.
Questi byte descrivono il buffer già esistente, non un'allocazione nuova e
nemmeno una misura delle pagine fisicamente residenti. Sorgenti, snapshot,
binario e dimensione/mtime del medesimo GGUF risultano invariati; la build
non produce warning. Il modello misura 13.146.393.504 byte su disco.

Lo swap è invariato prima/dopo ciascun processo: 1397,06 MiB nei primi
quattro, 1389,06 MiB negli ultimi tre. La diminuzione di 8 MiB avviene fra
processi e non viene attribuita alla variante. Non si deduce stabilità di
clock, temperatura o residenza fisica da questi valori.

Questa replica verifica tempi e attivazione; non ripete la suite numerica o
i checkpoint. La [precedente qualità completa](m5_model_residency_2026-10-10.md)
riguarda il suo snapshot e aveva superato i controlli bit per bit. Il
risultato prestazionale attuale non giustifica una nuova suite né
l'integrazione. Le velocità assolute di questa campagna non sono un nuovo
confronto con llama: resta valido il [confronto diretto bilanciato](m5_native_reference_run_cooldown_2026-10-11.md).

[Manifest e comandi](measurements/model_residency_retry_2026-10-11/manifest.json),
[analisi](measurements/model_residency_retry_2026-10-11/analysis.json), tutti
i log e gli snapshot immutabili restano nell'archivio della replica.
