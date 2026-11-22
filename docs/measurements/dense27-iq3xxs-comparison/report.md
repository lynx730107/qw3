# Qwen3.8-27B Unsloth su Apple M5

Il candidato è il GGUF UD-IQ3_XXS originale della revisione `4af51b3542f775b0a8e4b310bfb3f17e01c306c9`. Il download di 10.934.860.704 byte ha lo SHA-256 pubblicato da Unsloth. L’inventario degli stessi tensori attivi del trunk e della testa, contando una sola riga di embedding, scende da 12,238 a 10,156 GB per token (−17,0%). I byte sono dimensioni logiche dei pesi. [File ufficiale](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4af51b3542f775b0a8e4b310bfb3f17e01c306c9/Qwen3.8-27B-UD-IQ3_XXS.gguf) · [SHA-256](download.sha256) · [Conteggio dei pesi](active-bytes.json)

## Prima del supporto nativo

In qw3 nativo il controllo Q3_K_XL ottiene 9,08 tok/s su 128 token. Con IQ3_XXS il file si apre e la sessione alloca i buffer, poi fallisce il warmup a profondità 1024. Il percorso Metal del modello denso controlla esplicitamente il formato dell’embedding Q3_K e della testa Q5_K. Il candidato presenta embedding Q2_K, testa Q4_K e matrici FFN IQ1_S e IQ1_M. [Controllo](A1.log) · [Esecuzione IQ3_XXS](native-IQ3.log)

In llama.cpp build 11960 (`8642541b2`), quattro processi sequenziali, ordine A/B/B/A con due ripetizioni da 128 token ciascuno, dopo un prefisso di 1.024 token. Entrambe le quantizzazioni usano Metal, Flash Attention, KV F16, quattro thread, batch e microbatch 128. Le velocità aggregate dividono 512 token per la somma dei quattro tempi di ciascun GGUF. Un ulteriore controllo A eseguito durante il download è escluso. [Comando](compare.sh)

| Quantizzazione | Prima coppia di ripetizioni, tok/s | Seconda coppia, tok/s | Aggregato, tok/s |
|---|---|---|---:|
| Q3_K_XL | 9,19 · 9,21 | 8,93 · 8,71 | 9,005 |
| IQ3_XXS | 10,12 · 9,79 | 9,43 · 9,01 | 9,569 |

Il vantaggio aggregato è 0,564 tok/s (+6,27%). La velocità di entrambi i modelli varia nel corso della prova. Solo la prima ripetizione IQ3_XXS supera 10 tok/s. [A iniziale](llama-A1.json) · [B primo](llama-B1.json) · [B secondo](llama-B2.json) · [A finale](llama-A2.json)

Una prova qw3 a profondità zero, senza warmup, fallisce al primo embedding anche saltando il prefill batch. [Generazione nativa senza prefisso](native-IQ3-depth0.log)

## Qualità con llama-server

Sul README del progetto al commit `74da025`, valutando gli stessi primi quattro blocchi da 512 token con llama-perplexity, la perplessità passa da 6,5585 a 6,7056 (+2,24%). È un singolo campione di documentazione tecnica. [Controllo](ppl-A.log) · [Candidato](ppl-B.log)

Ciascuna quantizzazione supera quattro prove funzionali con llama-server in modalità greedy. Le prime tre richiedono scrittura e compilazione di un piccolo programma C, scrittura ed esecuzione di uno script Python e lettura di tre commit Git. Le azioni sono eseguite in directory temporanee con argomenti vincolati e l’output finale viene verificato. La quarta richiede una chiamata a visit_url e il confronto di quattro campi JSON con i dati restituiti dall’API GitHub. Le prove riguardano il backend llama-server. Le prove successive con qw3-agent sono riportate sotto. [C/Python/Git controllo](tasks-A.jsonl) · [C/Python/Git candidato](tasks-B.jsonl) · [Web controllo](web-A.json) · [Web candidato](web-B.json)


## Supporto nativo qw3

Il loader usa il GGUF originale. Metal legge direttamente l’embedding Q2_K, le matrici IQ1_S/IQ1_M e la testa Q4_K. La selezione delle proiezioni in batch usa la stessa verifica di disponibilità nel pianificatore e nel backend. Le proiezioni IQ1 del prefill usano il kernel per singolo token. Il riferimento CPU decodifica anche i due formati IQ1. Il test dei quantizzatori confronta 64 blocchi per formato con la decodifica di ggml. [Test dei blocchi](../../../tests/test_quant_k.c)

Il controllo CPU/Metal confronta l’intero vocabolario su due passi. Su un prefisso di 128 token, batch e seriale scelgono entrambi il token 194, poi il 195 dopo la continuazione. Per il prefill, RMSE 0,00284, massimo assoluto 0,01525. La porzione di stato DeltaNet controllata rientra nelle soglie del test. [Log del prefill e della continuazione](native-batch.log)

L’agente nativo supera le prove di scrittura e compilazione C, esecuzione Python, lettura Git con cache fredda, calda e disabilitata, e chiamata a visit_url con verifica di quattro campi dell’API GitHub. [C](native-agent-coding.log) · [Python](native-agent-python.log) · [Git](native-agent-git.log) · [Web](native-agent-web.log)

La serie iniziale è stata eseguita in ordine IQ3_XXS, Q3_K_XL, Q3_K_XL, IQ3_XXS. Ogni processo esegue due ripetizioni da 128 token dopo 1.024 di prefisso. Le medie dei blocchi sono 9,53 / 7,83 / 7,12 / 6,44 tok/s. I throughput aggregati, calcolati come token totali divisi per il tempo totale, sono rispettivamente 7,68 e 7,45 tok/s per IQ3_XXS e Q3_K_XL. La deriva tra inizio e fine serie limita il confronto fra i due modelli. [IQ3 prima](native-paired-IQ3_XXS-1.log) · [Q3 prima](native-paired-Q3_K_XL-1.log) · [Q3 seconda](native-paired-Q3_K_XL-2.log) · [IQ3 seconda](native-paired-IQ3_XXS-2.log)

Sulla build finale, il retest del solo IQ3_XXS misura **9,45 tok/s** con profondità 1.024 e generazione da 128 token (due ripetizioni, deviazione standard 0,48). Il prefill di 128 token a contesto vuoto misura **81,56 tok/s** (due ripetizioni). Metal usa tutti i 64 strati, KV F16, quattro thread, IQ3_NIBBLE e DENSE_PARTIAL_TILES attivi. Il controllo con la CLI senza gli ultimi due flag risponde 56 alla domanda su 7 × 8. [Generazione](native-final.log) · [Prefill](native-prefill.log) · [CLI](native-default-cli.log)

Su due sequenze di input, la build finale riproduce la classifica dei primi otto logits e otto token greedy di llama.cpp per entrambi i GGUF. L’errore massimo sull’intero vocabolario per IQ3_XXS è 0,000541. [IQ3_XXS](native-final-logits.log) · [Q3_K_XL](native-original-logits.log)
