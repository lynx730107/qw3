# Store dell'agente: retention e pulizia dell'11 ottobre 2026

Lo store `/Users/pasqualecapasso/.qw3/agent_store` occupava 1.115.679.945
byte logici, circa 1,116 GB. Dopo la pulizia autorizzata restano 30.109 byte:
soltanto la history, con 1.000 voci e hash invariato. Sono stati rimossi
45 file, 1.115.649.836 byte logici e 1.115.766.784 byte allocati secondo
`st_blocks`. Nessun processo qw3 o file aperto nello store è stato rilevato
prima della rimozione. I contenuti dei prompt non sono inclusi nell'archivio.

| Classe | Prima della pulizia | Policy già presente nel branch |
| --- | ---: | --- |
| `.prefix-v*.kvc` | 4 file, 737.969.408 byte | Conserva al massimo 4 file dopo una preparazione riuscita del prefisso |
| `spills/bash-*` | 8 file | Conserva al massimo 8 spill dopo un output grande salvato con successo |
| `spills/context-*` | 8 file | Conserva al massimo 8 archivi dopo una compattazione archiviata con successo |
| `history` | 1.000 voci, 30.109 byte | Conserva le ultime 1.000 voci; riscrive il file, senza append |
| `hybrid/` | 25 file, 375.794.408 byte | Codice assente da questo branch; nessuna retention generale nel checkout sperimentale esaminato |

Gli spill occupavano complessivamente 1.886.020 byte. Sono stati eliminati,
insieme alle quattro cache e all'intera directory `hybrid`. Le cache si
ricreano al bisogno; il primo avvio può richiedere nuovamente il prefill.
Il checkout vicino `../qw3-exp-lopped`, che contiene altri lavori modificati,
non è stato cancellato o modificato.

## Correzioni all'analisi del 27B

L'hash del prefisso evita duplicati della stessa chiave, ma da solo non
garantisce un limite. Il limite reale è `QW3_AGENT_PREFIX_CACHE_KEEP=4`,
applicato da `agent_prepare_prefix_cache` anche dopo un hit riuscito.
La chiave include versione, backend, contesto, percorso del modello,
configurazione KV/offload e token del sistema, che incorporano anche la data
locale. I quattro file osservati non dimostrano quattro soli set di tool.

La stessa funzione di pruning conserva il file corrente e ordina gli altri
per mtime decrescente. Non è una LRU rigorosa: il cache hit non aggiorna il
mtime. La pulizia avviene durante l'uso, senza un timer o una scadenza; mtime
vecchi dei file sopravvissuti non dimostrano assenza di rolling. Le costanti
`QW3_AGENT_BASH_SPILL_KEEP=8` e `QW3_AGENT_CONTEXT_ARCHIVE_KEEP=8` spiegano
esattamente i 16 spill presenti.

La history usa `linenoiseHistorySetMaxLen(1000)`: l'inserimento espelle la
voce più vecchia e `linenoiseHistorySave` apre il file in modalità `w`.
Non è append-only; i duplicati consecutivi vengono ignorati.

Nel checkout sperimentale, `agent_hybrid_run` crea un `job-*` per ogni turno
ordinario, prima di sapere se ci sarà una delega. I comandi di controllo
seguono un percorso separato. `moe-*` è invece una directory per profilo
del modello, derivato da percorso assoluto e dimensione del GGUF. Al suo
interno vengono salvate conversazioni distinte; senza un ID esplicito,
il nome della conversazione include tempo e PID. `hybrid_finish` rimuove
il manifest attivo e `dense-store/expert.kvc`, ma lascia directory del job,
journal e risultato. La trace `moe.trace.jsonl` viene aperta in append.
È quindi corretta la preoccupazione sull'accumulo hybrid, ma non la
descrizione delle sue unità di crescita.

## Cosa rimane senza una quota

I limiti di cache e spill sono sul numero di file per directory, senza
un tetto complessivo in byte. In particolare lo spill bash conserva
l'output integrale: il limite di visualizzazione non ne limita la
dimensione su disco. Le conversazioni salvate vengono riscritte per lo
stesso ID, ma nuovi ID si accumulano senza retention automatica.
Le trace, quando richieste, sono append-only anche nel branch corrente.
Non esiste quindi una garanzia di dimensione massima per tutto lo store.

La pulizia eseguita non introduce una policy che cancelli automaticamente
conversazioni persistenti. I limiti già presenti restano invariati; il
runtime di inferenza e i binari di produzione non sono stati modificati.

## Verifica senza modello

`make test-agent-store` compila un harness del codice reale dell'agente
e di linenoise. Opera soltanto in una directory temporanea e verifica:

- 12 cache di versioni diverse ruotano a 4; il file corrente è conservato
  anche con mtime più vecchio, e file estranei e directory restano intatti;
- 12 salvataggi tramite `agent_archive_context` conservano gli ultimi 8,
  con contenuto intatto e permessi 0600;
- 1.012 inserimenti nella history conservano le ultime 1.000 voci;
  risalvataggio, duplicato consecutivo e reload non aumentano il file.

Passa anche `sh tests/test_agent_bash_tool.sh`, che produce più di otto
output grandi e controlla la retention degli spill reali. Nessun modello,
sessione Metal o copia dei pesi viene caricato per questi controlli.
Il primo harness aveva interpretato al contrario il ritorno booleano di
`agent_message_ledger_add`; corretto il test, tutti i controlli passano.
I log iniziali sono conservati e includono 11 warning di funzioni inutilizzate
della build CPU del core invariato; la compilazione finale dell'harness
non produce warning. Non si tratta di una regressione del runtime.

[Inventario iniziale](measurements/agent_store_2026-10-11/inventory_before.json),
[pulizia](measurements/agent_store_2026-10-11/cleanup.json),
[inventario finale](measurements/agent_store_2026-10-11/inventory_after.json)
e [verifica](measurements/agent_store_2026-10-11/manifest.json) documentano
dimensioni, hash del codice esaminato, comandi e risultati.
