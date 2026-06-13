# qw3

`qw3` e' un piccolo runtime sperimentale in C/Objective-C per Qwen3.6
35B-A3B in formato GGUF, con backend Apple Metal, benchmark, valutazione e un
agente locale integrato.

Il progetto e' nato come esperimento per replicare in piccolo alcune idee del
progetto DwarfStar/ds4 di Salvatore Sanfilippo: runtime minimale, integrazione
verticale con il modello, sessioni locali, tool nativi e latenza bassa senza
passare da un server HTTP. Non e' una sostituzione di llama.cpp e non pretende
di avere la stessa generalita': e' un laboratorio focalizzato su questo modello
e su Apple Silicon.

## Stato

- Backend principale: Metal su macOS/Apple Silicon.
- Target modello: `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`.
- Generazione Metal: stabile.
- Prefill Metal: in ottimizzazione; il benchmark locale `pp4096` e' intorno a
  600 tok/s su Apple M5 nelle condizioni documentate in
  `docs/metal_prefill_validation.md`.
- Agente: utilizzabile, con tool locali, compattazione del contesto e tool di
  navigazione codice.

I pesi del modello non sono inclusi nel repository.

## Build

Su macOS con Xcode Command Line Tools:

```sh
make
```

Il build predefinito su Darwin produce binari Metal senza suffisso:

```text
qw3        CLI di generazione
qw3-agent  agente locale
qw3-bench  benchmark
qw3-eval   valutazioni
```

I target CPU restano disponibili esplicitamente:

```sh
make cpu
```

Per pulire:

```sh
make clean
```

## Uso Rapido

Generazione:

```sh
./qw3 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --ctx 16000 --nothink -p "ciao"
```

Agente interattivo:

```sh
./qw3-agent -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --ctx 16000 --nothink
```

Benchmark stile llama-bench:

```sh
./qw3-bench -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --llama-style -p 4096 -n 0 -r 3 --no-warmup
```

Regressione logits Metal:

```sh
make test-metal-logits
```

## Agente

`qw3-agent` usa il formato nativo Qwen per i tool call e mantiene la sessione
localmente. I tool principali sono:

- `read`, `more`, `list`: lettura e navigazione file.
- `write`, `edit`: scrittura e modifica file.
- `search`: ricerca testuale.
- `bash`: comandi locali.
- `get_skeleton`: struttura di un file tramite `codenav`.
- `get_function`: corpo di una funzione/metodo tramite `codenav`.
- `semantic_search`: ricerca semantica tramite `colgrep`.

Per evitare di riempire il contesto leggendo file grandi, l'agente e' istruito
a preferire `get_skeleton`, `get_function` e `semantic_search` prima dei read a
blocchi.

Comandi utili nell'agente:

```text
/help
/status
/tools on
/tools off
/think
/nothink
/compact
/quit
```

## Tool Esterni Dell'Agente

I sorgenti di `codenav` sono inclusi nel repository. Il target scarica, se
servono, i parser tree-sitter in `codenavsrc/third_party` e compila il binario:

```sh
make tools
export PATH="$PWD/codenavsrc:$PATH"
codenav get_skeleton qw3_agent.c
```

`colgrep` non e' vendorizzato. Installalo dal suo canale upstream e verifica
che sia nel `PATH`:

```sh
colgrep --help
colgrep init .
```

Senza `codenav` o `colgrep`, l'agente continua a funzionare, ma i tool
`get_skeleton`, `get_function` e `semantic_search` restituiranno errore.

## Note Metal

Opzioni utili:

- `--ctx N`: dimensione del contesto.
- `--kv-f16`: KV cache f16.
- `--ngl N`: numero di layer eseguiti su Metal; i restanti vanno sul path CPU.
- `--nothink`: disabilita il thinking mode nel prompt.

La KV cache q8 e altri flag sperimentali sono ancora da considerare instabili:
prima di promuovere un'ottimizzazione vanno sempre eseguiti i test logits e una
prova no-garbage su prompt reale.

## Validazione

Comandi minimi prima di fidarsi di una modifica Metal:

```sh
make
make test-metal-logits
./qw3 -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --ctx 16000 --nothink --prompt-file ./prompt_perf.txt -n 128
```

Le note di sviluppo e i risultati dei benchmark sono in:

```text
docs/metal_prefill_validation.md
```
