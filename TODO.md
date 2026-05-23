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
   - Aggiungere formato q8_0 per K/V GQA.
   - Mantenere DeltaNet recurrent/conv in F32, coerente con llama.cpp.
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
   - Il dynamic-router e' ora il default: top-8/softmax router su GPU, buffer `routerIds/routerWeights` in sessione e MoE sparse con expert slot dinamici.
   - Il path dynamic-router usa prima le view GGUF Metal pointer-based per i tensor expert completi e solo come fallback crea wrapper no-copy temporanei; questo elimina il grosso costo precedente.
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
   - Misura default aggiornata dopo dynamic-router pointer-based: `--metal-run 2 -p ciao --ctx 128` mostra token stabili circa `41-43 ms/token` (`router_sync_ms=0`), con generazione argmax breve circa `22-23 tok/s`.
   - Restano copie temporanee per input/output CPU dei wrapper diagnostici/non-session e rimane CPU-assistito top-k/softmax del ramo sparse MoE.
   - Accesso ai buffer modello Metal stabilizzato per il percorso sessione tramite resolver pointer-based.
   - Ridurre command buffer separati.
   - Fondere kernel solo dopo aver mantenuto test di correttezza equivalenti.

## Prossimi step consigliati

1. Attaccare il sync pre-router dominante: profilare separatamente shared MoE, ramo attention/DeltaNet e residual norm dentro il batch che precede ogni router.
2. Ottimizzare ancora i kernel expert-slot del dynamic-router, ora default: target successivo sotto i 35 ms/token.
3. Introdurre KV cache q8_0 per GQA, mantenendo DeltaNet state/conv in F32.
