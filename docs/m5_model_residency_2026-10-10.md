# M5: prova privata della residenza dei pesi Metal (2026-10-10)

Il modello GGUF è già esposto al backend attraverso un unico buffer Metal
senza copia, da 13.135.413.248 byte, a partire dall'offset dati 10,48 MiB.
Questa prova inserisce quel buffer originale in un `MTLResidencySet`,
commette la lista, associa il set alla command queue e richiede la residenza.
Non crea nuovi buffer dei pesi né cambia kernel, GGUF o precisione. Richiede
le API Metal disponibili da macOS 15; un errore di creazione o una lista
vuota fa fallire il candidato anziché produrre una misura senza attivazione.

La prova rimane privata in `tests/model_residency_candidates.inc`.
`QW3_PRIVATE_MODEL_RESIDENCY=1` abilita la richiesta nel binario diagnostico;
il binario ordinario non legge questa variabile. La cache DeltaNet nativa
rimane attiva in entrambi i controlli: l'effetto misurato riguarda solo la
residenza dei pesi.

Il controllo completo usa due sessioni del medesimo motore, KV F16,
contesto 512, 64 layer Metal, prefissi seed 1 da 1/15/16/31/127/128/129/159
token, estensione 128→159 e 32 continuazioni per prefisso. Richiesta e
rimozione del set avvengono fra i forward, fuori da qualsiasi timer.
Passano intero vocabolario finito bit per bit, history, stato canonico K/V,
DeltaNet, convoluzione e payload checkpoint salvato e ricaricato.
266 richieste effettive, una allocazione originale; compilazione senza
warning. Non si deduce una garanzia di rilascio immediato delle pagine dalla
rimozione del set: questo controllo verifica la correttezza, non il tempo.
[Qualità completa](measurements/model_residency_quality_2026-10-10/results.json).

Le prestazioni vengono misurate in processi distinti: sette blocchi
OFF/ON alternati, cinque ripetizioni per blocco, pausa di 60 secondi,
tg32 a profondità 128, contesto 512, stesso stream fisso e tutti i logits.
I contatori richiedono una richiesta e una lista non vuota per il candidato,
e zero richieste e allocazioni residenti per OFF. Hash e snapshot includono
il wrapper privato e il benchmark, oltre al backend, kernel e runner.
[Serie prestazionale](measurements/model_residency_perf_2026-10-10/results.json).

```sh
make qw3-model-residency-test qw3-model-residency-bench
QW3_METAL_KV_F16=1 ./qw3-model-residency-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf all
python3 tools/bench_qw3_candidates.py --binary ./qw3-model-residency-bench \
  --jobs tg32_d128 --repetitions 5 --order off,candidate,off,candidate,off,candidate,off \
  --candidate-env QW3_PRIVATE_MODEL_RESIDENCY=1 --cooldown 60 --output /tmp/residency-replica
```

La serie conserva tutti i 35 campioni e produce 8,32940→8,69648 tok/s
aggregati, cioè +4,407%. **Il risultato è inconcludente e la variante non
viene adottata**: le medie OFF sono 3689,97/3752,60/4177,29/3747,39 ms, con
dispersione del 13,207%, oltre la soglia preliminare del 5%. Il blocco OFF
4 rimane nel calcolo e nell'archivio; non si attribuisce al candidato il
vantaggio apparente amplificato da questo controllo. Lo swap totale usato
rimane invariato durante quel blocco, ma questo dato non determina la causa
del rallentamento. Sorgenti e binario risultano immutati e nessun campione
è eliminato. Il controllo numerico positivo dimostra soltanto la correttezza
della prova privata. Servirebbe una nuova serie stabile prima di promuoverla.
