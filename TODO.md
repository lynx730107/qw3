# TODO Metal

## Obiettivo

Chiudere il backend Metal trasformando il path oggi corretto/diagnostico in un runtime incrementale stabile e veloce.

## Stato attuale

- GQA single-token e multi-token da KV cache persistente: verificati.
- Layer completo con residual, `ffn_norm`, MoE top-8/shared e residual finale: verificato.
- Runner 40 layer `--metal-mixed40-test`: verificato.
- Final `output_norm` + `lm_head` `--metal-logits-test`: verificato.
- Argmax dei logits su Metal: verificato e usato dal path `--metal`.
- DeltaNet/conv session persistenti su due token: verificati.
- Layer lineare 0 completo via session buffer: verificato.
- Decode Metal incrementale su sessione persistente: verificato e usato da `--metal`.
- Continuazione greedy diagnostica `--metal-greedy-test`: usa la sessione persistente e confronta CPU/Metal step-by-step.
- Runner Metal-only `--metal-run`: usa la sessione persistente, senza reprefill del prompt a ogni token.
- Runner benchmark `--metal-run-quiet`: separa `prefill_ms`, `generation_ms`, `avg_decode_ms` e `generation_tok_s`.
- Smoke Metal esteso: passa.

## Mancante

1. Decode incrementale reale
   - Prefill del prompt una sola volta: fatto.
   - Decode token-by-token usando KV cache GQA persistente: fatto.
   - Decode token-by-token usando stati DeltaNet/conv persistenti: fatto.
   - Eliminare il reprefill diagnostico dal percorso `--metal` principale: fatto.
   - Eliminare il reprefill diagnostico da `--metal-run` e `--metal-greedy-test`: fatto.
   - Rimane da ridurre i readback interni del runner session slow.

2. Sampling GPU-side
   - Argmax dei logits su Metal: fatto.
   - Readback ridotto ai massimi per blocco e scelta finale del token: fatto.
   - Sampling CLI Metal con sessione persistente e logits sincronizzati su CPU: fatto.
   - Rimane da portare temperature/top-k/top-p/min-p interamente GPU-side.

3. KV cache q8_0
   - Aggiunto percorso Metal opt-in `QW3_METAL_KV_Q8_0=1` per K/V GQA: quantizzazione q8_0 alla scrittura cache e dequantizzazione direttamente nel kernel cached attention grouped.
   - A `--ctx 32768`, la cache GQA K+V della sessione passa da 1280 MiB F32 a 340 MiB q8_0; DeltaNet recurrent/conv rimane F32, coerente con llama.cpp.
   - Correttezza funzionale: `--metal-session-gqa-cached2-test` passa in q8; `--metal-session-decode-test -p ciao` mantiene top0 `8160` (`rmsdiff` atteso circa `0.0128` rispetto al reference F32).
   - Il path resta opt-in finche' non viene validata la qualita' su continuazioni lunghe; nelle misure locali brevi/medie la velocita' e' sostanzialmente neutra (`64`: circa `26.7 tok/s`; `256`: circa `25.8 tok/s`).
   - Esporre opzioni equivalenti concettualmente a `-ctk q8_0 -ctv q8_0`.

4. Ottimizzazione performance
   - Residual update `x0 = x0 + attn` e somma `moe` su session buffer Metal: fatto.
   - Eliminato il readback di `attn` e la ricostruzione CPU `x + attn + moe` per layer nel runner session.
   - Eliminato il readback di `alpha/beta` DeltaNet nel runner session; `sigmoid(beta)` e `gamma=exp(softplus(alpha+dt_bias)*a)` sono calcolati nel kernel recurrent.
   - Final `output_norm + lm_head` collegato ai buffer session (`x0 -> x1 -> logits`); resta il solo readback dei logits per argmax/sampling.
   - Ramo MoE shared spostato su buffer session (`x1 -> scratch -> inner -> x1 -> x0`); resta CPU-assistito il ramo sparse top-8.
   - Router sparse MoE calcolato da `x1` nella sessione e letto come 256 score; top-k/softmax restano CPU-assistiti.
   - Ramo sparse top-8 eseguito sui buffer sessione: gate/up IQ3_S da `x1`, SiLU in `inner`, down IQ4_XS/Q6_K in `scratch`, accumulo scalato diretto in `x0`.
   - Ramo sparse top-8 fuso a livello command-buffer: gli 8 expert del layer vengono encodati in un solo command buffer invece di sincronizzare gate/up/SiLU/down/accumulo separatamente per expert.
   - Introdotto batching command-buffer stile ds4 per il runner session: le primitive `qw3_metal_session_*` possono accodare su un batch aperto e sincronizzare solo ai punti di readback obbligatori.
   - Il runner session apre/chiude il batch attorno ai blocchi tra router readback e logits readback. Misura locale: `--metal-run 32 -p ciao --ctx 256` completa prompt 12 + 32 token in circa `9035 ms`.
   - Profiling locale dopo il batching: il collo di bottiglia visibile era il flush prima del router (`router_sync_ms` circa 170-175 ms/token), non il router f32 in se' (circa 8-12 ms/token).
   - Il dynamic-router e' ora il default: top-8/softmax router su GPU, buffer `routerIds/routerWeights` in sessione e MoE sparse batch.
   - Il path dynamic-router usa prima le view GGUF Metal pointer-based per i tensor expert completi e solo come fallback crea wrapper no-copy temporanei; questo elimina il grosso costo precedente.
   - Il batch MoE e' default nel dynamic-router; `QW3_METAL_NO_BATCH_MOE=1` o `QW3_METAL_DYNAMIC_LEGACY_MOE=1` tornano agli expert-slot dynamic non-batch.
   - `QW3_METAL_CPU_ROUTER=1` oppure `QW3_METAL_DYNAMIC_ROUTER=0` ripristinano il vecchio path CPU-assisted per confronto/debug.
   - Aggiunta cache globale/lazy per il buffer costante `iq3_s kgrid`, evitando riallocazioni ripetute nei matvec expert.
   - Aggiunti kernel fusi `iq3_s` gate+up per sparse MoE default e dynamic-router.
   - Aggiunto ramo sperimentale `QW3_METAL_GPU_ROUTER_TOPK=1`: router top8/softmax su GPU con readback di soli 8 ids/weights. Corretto, ma non default perche' nel profilo breve peggiora il router da circa 8-10 ms a circa 10-14 ms.
   - Misura del vecchio fallback CPU-router dopo kgrid cache + gate/up fused: `--metal-run 8 -p ciao --ctx 128` circa `4072 ms`.
   - Resolver view GGUF corretto in stile llama.cpp: lookup pointer-based `tensor_ptr -> view + offs`, non solo offset globale.
   - `lm_head` session q8_0/q6_K e matvec q8_0 session intermedie leggono dalle view Metal del GGUF invece di copiare il tensor in un buffer temporaneo per token.
   - Anche i pesi f32 hot della sessione leggono dalle view GGUF: RMSNorm, router f32, DeltaNet conv, gate constants, DeltaNet gated RMSNorm e Q/K norm GQA.
   - I wrapper sparse expert IQ3_S/IQ4_XS/Q6_K leggono i pesi expert dalle view GGUF invece di copiare ogni expert in un buffer temporaneo.
   - Shared expert ulteriormente fuso: `gate_shared` e `up_shared` Q8 sono calcolati in un solo kernel pair, e il down shared Q8 accumula direttamente in `x0` con `sigmoid(ffn_gate_inp_shexp)` senza passare da `x1`.
   - Aggiunto esperimento `QW3_METAL_ROUTER_F32_FAST=1` per router F32 row-blocked; non e' default perche' nel profilo breve non migliora i token stabili.
   - Misura default aggiornata dopo dynamic-router pointer-based + batch MoE: `--metal-run 2 -p ciao --ctx 128` mostra token stabili circa `38-40 ms/token` (`router_sync_ms=0`), con generazione argmax breve circa `24 tok/s`.
   - Il path argmax/`--metal-run` non legge piu' tutto il vettore logits su CPU per scegliere il token: `qw3_metal_session_argmax_logits()` riduce direttamente il buffer logits della sessione e legge solo i blocchi argmax. Misura locale `--metal-run 32 --metal-run-quiet -p ciao --ctx 128`: circa `23.0 tok/s`, `avg_decode_ms` circa `43.5 ms`.
   - I buffer parziali dell'argmax logits sono persistenti nella sessione, evitando allocazioni per-token.
   - Nel path greedy/argmax senza readback completo logits, `output_norm` e `lm_head` vengono encodati nel batch finale prima della sync unica. Misura breve `QW3_METAL_PROFILE=1 --metal-run 8 --metal-run-quiet`: circa `24.9 tok/s`; misura lunga `--metal-run 64 --metal-run-quiet`: circa `21.9 tok/s`.
   - Proiezione/cache GQA ottimizzata: Q RMSNorm per 16 head + copia gate e K RMSNorm per KV head sono fusi in 2 dispatch larghi invece di molti dispatch/blit per-head. `--metal-session-gqa-project-test 66` e smoke passano. Misura `--metal-run 32 --metal-run-quiet`: circa `23.9 tok/s`; `--metal-run 64 --metal-run-quiet`: circa `22.1 tok/s`.
   - Cached attention GQA riscritta seguendo la direzione flash/vec di ds4/llama.cpp: il dot Q*K viene ridotto nel threadgroup e accumulato con softmax online, invece di essere ricalcolato per ogni dimensione dell'head. La versione corrente lavora per KV-head e calcola insieme gli 8 Q-head del gruppo GQA, riusando K/V. `--metal-session-gqa-cached2-test 66`, `--metal-gqa-attend4-test 66`, `--metal-gqa-branch4-test 66` e smoke passano. Misura `--metal-run 32 --metal-run-quiet`: circa `26.9-27.0 tok/s`; `--metal-run 64 --metal-run-quiet`: circa `26.8 tok/s`.
   - Il down sparse MoE `q6_K`, usato nei layer finali lenti (`34`, `38`, `39`), usa ora un kernel batch multi-riga: 2 simdgroup elaborano 4 righe per threadgroup invece del dispatch generico per-riga. `--metal-session-decode-test -p ciao` passa (`maxdiff=2.217293e-05`, top0 `8160`) e `make test-metal-smoke` passa. Due misure consecutive `--metal-run 64 --metal-run-quiet -p ciao --ctx 128` danno `27.86` e `27.85 tok/s`, rispetto a circa `26.8 tok/s` prima del cambio.
   - Sul Mac Apple M5 locale con SDK macOS 26.4 sono presenti le API Metal 4 (`MTL4CommandQueue`, command allocator e `MTLTensor`, inclusi `Int4/UInt4`). E' una direzione successiva per ridurre overhead di submission o valutare tensor operations; i formati GGUF hot `IQ3_S`/`IQ4_XS`/`Q6_K` continuano comunque a richiedere kernel quantizzati dedicati.
   - Aggiunto profiler opt-in `QW3_METAL_PROFILE_LAYER_SYNC=1` per stimare i costi layer-by-layer forzando sync a ogni layer; e' diagnostico e rallenta volutamente il runner.
   - Aggiunto esperimento opt-in `QW3_METAL_UNRETAINED_COMMAND_BUFFERS=1`, ma non va usato di default: su Apple M5 produce `Invalid Resource` nel batch runtime.
   - Aggiunto esperimento opt-in `QW3_METAL_FUSED_DOWN_REDUCE=1` per fondere down sparse IQ4_XS e reduce su `x0`; resta non-default perche' nelle misure locali non migliora il path stabile.
   - Restano copie temporanee per input/output CPU dei wrapper diagnostici/non-session; il path runtime default non usa piu' top-k/softmax CPU per il ramo sparse MoE.
   - Accesso ai buffer modello Metal stabilizzato per il percorso sessione tramite resolver pointer-based.
   - Ridurre command buffer separati.
   - Fondere kernel solo dopo aver mantenuto test di correttezza equivalenti.

## Prossimi step consigliati

1. Profilare nuovamente i layer dopo il kernel `q6_K` batch: isolare il prossimo formato/dispatch dominante nel blocco pre-logits e puntare sotto i 35 ms/token.
2. Valutare una migrazione Metal 4 separata e misurabile per command submission/tensor API, senza sostituire i kernel GGUF custom finche' non mostra un vantaggio reale.
3. Validare la KV cache q8_0 su continuazioni lunghe e aggiungere opzioni CLI equivalenti a `-ctk q8_0 -ctv q8_0`.
