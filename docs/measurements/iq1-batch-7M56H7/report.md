# Prefill del 27B denso su Apple M5

Con qw3-bench e prompt da 128 token, il batch Metal per le proiezioni IQ1_S e IQ1_M porta IQ3_XXS da 55,87 a 96,28 token/s (+72,3%). Q3_K_XL raggiunge 101,30 token/s nella stessa serie. IQ3_XXS rimane il 5,0% sotto Q3_K_XL.

| GGUF e configurazione | Primo blocco, tok/s | Secondo blocco, tok/s | Aggregato, tok/s |
|---|---:|---:|---:|
| IQ3_XXS, batch IQ1 disattivato | 55,99 | 55,80 | 55,87 |
| IQ3_XXS, batch IQ1 attivo | 98,83 | 94,16 | 96,28 |
| Q3_K_XL | 100,12 | 102,98 | 101,30 |

Sei processi consecutivi in ordine vecchio, nuovo, Q3, Q3, nuovo, vecchio. Ogni blocco misura tre ripetizioni a profondità zero, con Metal, 64 strati su GPU, KV F16, quattro thread e contesto allocato di 1.280 token. L’aggregato divide i 768 token per il tempo totale delle sei ripetizioni di ciascuna configurazione. La prova misura solo il prefill. [Comandi](final.sh)

## Comportamento in italiano

In modalità greedy con ragionamento disattivato, tutte e cinque le risposte di IQ3_XXS sono identiche con il nuovo kernel attivo e disattivato. Il modello estrae il codice B42, restituisce «NON INDICATO» per un importo assente e recepisce lo spostamento della riunione a giovedì alle 11. Sbaglia il conto dei bulloni, rispondendo 30 a 48 − 15 − 9 + 2 × 6, che vale 36. Nell’email riporta il ritiro da venerdì a lunedì e lascia aperta la data di consegna, ma attribuisce lo spostamento a un «imprevisto logistico» senza averne evidenza. La generazione dell’email è limitata a 160 token. [Prompt](italiano.py) · [Risposte integrali](italiano.jsonl)

Su un prefisso di 63 token e tre continuazioni forzate, il token più probabile coincide in tutti e quattro i confronti. Il massimo RMSE è 0,001031 e la massima differenza assoluta è 0,00561, leggermente sopra le consuete soglie 0,001 e 0,005. Il confronto riguarda due percorsi Metal sul medesimo GGUF. [Log numerico](prod-compare.log)

## Prove con qw3-agent

IQ3_XXS con il kernel attivo supera le prove di compilazione C e di esecuzione Python. Su Git riferisce i tre commit attesi con cache fredda, calda e disabilitata. Nel test web riporta correttamente i quattro campi richiesti dall’API GitHub. Il confronto separato con llama.cpp supera il test del decode sequenziale. [C](prod-coding.log) · [Python](prod-python.log) · [Git](prod-git.log) · [Web](prod-web.log) · [Decode sequenziale](prod-reference.log)

Anche Q3_K_XL risponde 30 al problema dei bulloni e supera le altre tre domande a risposta chiusa. [Risposte Q3_K_XL](italiano-q3.jsonl) · [Test del contesto e degli strumenti](context-tools.log)
