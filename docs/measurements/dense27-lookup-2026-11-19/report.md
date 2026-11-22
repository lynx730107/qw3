# Prompt lookup sul Qwen3.8-27B denso, Apple M5

## Agente e replay

Una build diagnostica di qw3-agent ha salvato, dopo ciascuna generazione, gli ID esatti del contesto presentato al modello e dei token emessi. Sono stati riutilizzati il runtime, il GGUF UD-Q3_K_XL, i kernel Metal e i due flag della precedente prova agente. Il [diff](capture.diff) aggiunge la sola cattura, al termine del timer di generazione. Le vecchie tracce contenevano conteggi aggregati. I campioni in questa directory provengono da nuove esecuzioni del 19 novembre. Le fixture C, Python e Git hanno verificato l'esecuzione e gli output. La prova web ha cercato RFC 9110, aperto la pagina ufficiale e risposto con titolo, giugno 2022 e link. [Log web](web.log) · [Esiti delle fixture](status.txt)

Il [simulatore](replay.py) indicizza finestre interamente precedenti al token appena campionato. Dopo ogni corrispondenza propone fino a sette token dall'occorrenza idonea più recente. Avanza del token già campionato e del solo prefisso di proposte coincidente con l'output registrato. Considera anche gli insuccessi e termina al confine della chiamata dell'agente. Ogni prova da otto posizioni comprende un token già noto e sette da verificare. Un accettato è un token previsto *in più* rispetto a quello già noto. Una ricerca esaustiva indipendente ha riprodotto i conteggi della versione indicizzata in 500 casi sintetici. La modalità ristretta usa solo finestre interamente comprese in messaggi ChatML dell'utente, inclusi gli output dei tool. I messaggi di sistema e i messaggi dell'assistente sono esclusi dalle fonti di questa modalità.

Si contano 12 generazioni, 510 token. Delle tre esecuzioni Git della stessa fixture si usa solo la prima. I test C e Python chiedono esplicitamente di riprodurre codice o testo. Questi campioni favoriscono la copia. Il tasso di corrispondenza descrive le traiettorie osservate, anche nel test Git che usa il sampling di default. Il replay non implementa un campionatore speculativo.

| Compito | Token | Tentativi con chiave 6 | Token aggiuntivi indovinati | Blocchi completi | Tentativi con chiave 12 | Token aggiuntivi indovinati | Blocchi completi |
|---|---:|---:|---:|---:|---:|---:|---:|
| C | 113 | 11 | 44 | 4 | 5 | 23 | 2 |
| Python | 135 | 14 | 56 | 5 | 6 | 37 | 4 |
| Git | 97 | 12 | 41 | 5 | 5 | 35 | 5 |
| Web | 165 | 12 | 28 | 2 | 3 | 8 | 0 |

Con chiave 6 la media è 4,45 token utili per verifica, incluso il primo. Con chiave 12 è 6,42. Limitando la fonte ai messaggi utente/tool e usando chiave 12, sul C si ottengono 19 token aggiuntivi in 4 tentativi, su Python 33 in 5, su Git 35 in 5 e sul web 4 in 2. Sono analisi esplorative dello stesso campione. [Tutte le chiavi e larghezze](replay.json) · [Fonti ristrette](inputs-only.json)

## Ripristino dello stato qw3

Il [diagnostico](checkpoint.c) usa la stessa sessione a ctx8192 per tre salvataggi su file temporaneo e relativi caricamenti. Prefisso di 1.948 token preso dalla prima richiesta C. Ogni ripristino riproduce posizione, token greedy e intero vocabolario del passo successivo. Il payload è 293.947.200 byte. Il salvataggio impiega 356–379 ms e il caricamento 432–704 ms. Il tempo di salvataggio include fflush. Il sistema può usare la cache del filesystem. Non è un benchmark della banda fisica del disco. Con un rollback speculativo occorrerebbe inoltre ricostruire l'eventuale prefisso accettato. [Misure](checkpoint.log)

## Confronto indipendente con llama.cpp

Sulla stessa macchina e sullo stesso GGUF sono stati avviati quattro processi llama-server in ordine OFF/ON/ON/OFF. Ognuno riceve il primo prompt C e l'ultimo prompt web catturati dall'agente. Si misurano i primi 64 token greedy della risposta, con cache del prompt disattivata e prefill escluso. La chiave è di sei token con proposta massima di sette. Il contesto è 8192, KV F16 e batch di prefill 128. Le due implementazioni del lookup differiscono nella gestione delle corrispondenze brevi. I campioni di llama misurano il suo backend e il suo campionatore. [Configurazione](reference.py) · [Token e tempi](reference.json)

| Compito | OFF iniziale | ON | ON | OFF finale | Accettazione in ciascun ON |
|---|---:|---:|---:|---:|---:|
| C | 5,07 | 8,39 | 7,10 | 4,89 | 26 su 38 |
| Web | 5,86 | 5,55 | 4,81 | 5,20 | 0 su 7 |

Le velocità sono in tok/s. Tutte le quattro esecuzioni di ogni compito producono gli stessi 64 ID. Il risultato web riguarda solo la parte iniziale della risposta.

## Verifica di tutte le righe in qw3

Abbiamo scelto il primo blocco interamente prevedibile del turno C con chiave 6. Posizione 1983, dopo il prefill reale e 35 token generati. Lo stesso checkpoint inizializza ogni lato. Per il percorso batch si calcolano tutte le otto teste Q5_K sulle righe finali. Entrambi i percorsi rendono disponibili al tester i logits completi sul lato CPU. [Selezione](select_block.py) · [Diagnostico](block.c) · [Tre coppie dopo riscaldamento](block.log)

Su 1.986.560 valori finiti, 1.986.222 differiscono nella rappresentazione binaria. Gli otto argmax coincidono. RMSE peggiore per riga 0,001478 e massimo assoluto 0,008620, superiori alle soglie di 0,001 e 0,005. I tempi per gli otto token sono 1.488–1.932 ms in seriale e 764–1.015 ms in batch, escluso il checkpoint. Il test numerico giustifica mantenere il verificatore tra gli esperimenti. [Criteri di accettazione](../../optimization_acceptance.md)

## Compito di refactoring con istruzione naturale

Nel compito aggiuntivo sul parser CSV, con l’agente ordinario e un limite di nove chiamate ai tool, l’agente ha letto il codice e riprodotto gli errori ma ha esaurito il budget prima di modificare il file. La verifica indipendente fallisce tre test su quattro. Sui dieci segmenti e 733 token di questa traiettoria, la chiave 12 predice 95 token aggiuntivi in 22 tentativi, sei blocchi completi. Questo campione rappresenta un compito incompleto. [Traccia](refactor/trace.jsonl) · [Verifica indipendente](refactor/check.log)

Sul medesimo progetto CSV, il 20 novembre l'agente con le impostazioni ordinarie di 24 cicli ha completato la modifica dopo 15 chiamate ai tool. I quattro test originali passano anche dopo aver ripristinato il file dei test dalla fixture. Due casi aggiunti a posteriori mostrano limiti nel parser prodotto: perde le virgolette letterali ai bordi del campo e separa una virgola interna se la virgoletta di apertura è preceduta da uno spazio dopo il delimitatore. L'eseguibile agente è stato ricostruito fra le due date e il sorgente è cambiato. Il diverso esito non si può attribuire al solo limite di cicli. Entrambi i test CSV usano il decoder ordinario. [Quattro test originali](refactor/retry/check.log) · [Due casi aggiuntivi](refactor/retry/extra-check.log)

## Prototipo greedy qw3 con rollback GPU, 20 novembre

Il prototipo conserva DeltaNet, convoluzione, logits e scratch di coda Flash nelle righe K/V ancora inutilizzate della sessione Metal. Un margine oltre le otto posizioni e il padding dell'attenzione tiene il salvataggio fuori dall'intervallo letto e scritto. Se la capacità residua è insufficiente la copia viene rifiutata. Si usano copie Metal buffer-to-buffer e un solo modello/sessione. Il prototipo non alloca un buffer di salvataggio dedicato. Con il layout di questo GGUF i dati da conservare occupano 158,57 MiB. Dal payload e dalla capienza delle righe si ricava, per ctx8192 e blocco da otto, una posizione massima di 5592 per questo schema di copia. Il diagnostico verifica il rifiuto vicino al limite. L'impatto sulla memoria residente e sul picco del processo resta da misurare. [Implementazione delle copie](tail.m)

A ciascuno dei tre prefissi, prima dei tempi, abbiamo salvato l'intero payload, valutato otto token volutamente errati, ripristinato il salvataggio GPU e comparato i due payload canonici byte per byte. Coincidono in tutti e tre i casi. Passa anche il controllo di rifiuto a otto posizioni dal limite della cache. Durante la prova web, il salvataggio GPU richiede 2,76–2,78 ms e il ripristino 2,70–2,72 ms. I checkpoint portabili su file includono anche dati KV vivi, checksum e metadati e hanno una funzione diversa.

Per ogni prompt, il generatore ordinario e il candidato usano argmax puro senza penalità. Si riparte dallo stesso checkpoint in ordine OFF/ON/ON/OFF. I tempi iniziano dopo il ripristino. Il candidato cerca una chiave di 12 token nella cronologia e propone i sette successivi. Il backbone valuta gli otto token insieme. Le teste di output sono calcolate in ordine fino alla prima proposta errata, momento in cui si ripristina lo stato e si rivalutano in seriale i token accettati. Per un blocco interamente accettato si conserva direttamente lo stato batch. Il risultato misura la generazione a partire dal prompt catturato. Le chiamate ai tool e le regole di arresto dell'agente sono escluse. [Generatore diagnostico](spec.c)

| Prefisso | ID generati | OFF iniziale | ON | ON | OFF finale | Proposte accettate per prova ON |
|---|---:|---:|---:|---:|---:|---:|
| C, prima chiamata | 64 | 8,510 | 8,915 | 9,014 | 8,133 | 14 su 14 |
| Git, risposta dopo il tool | 72 | 8,586 | 8,654 | 8,623 | 8,637 | Nessuna proposta |
| Web, risposta dopo la visita | 91 | 8,373 | 7,653 | 7,453 | 7,194 | 2 su 7 |

Tutti gli ID di ogni prova coincidono con il controllo del rispettivo prompt. Le velocità sono in tok/s. I controlli web variano sensibilmente e le due coppie mostrano segni opposti. La traiettoria greedy Git differisce dal test dell'agente con campionamento ordinario. Il vantaggio C deriva da due blocchi completamente accettati per esecuzione. Questa dimensione del campione limita la generalizzazione. [C](spec-coding.log) · [Git](spec-git.log) · [Web](spec-web.log)

### Origine dello scarto e selezione indipendente

Sul medesimo blocco C è stato disabilitato il percorso batch per le proiezioni FFN, per le proiezioni DeltaNet, oppure per entrambe in tutti gli strati. Il prefill di partenza e l'oracolo restano identici. Disabilitare le proiezioni DeltaNet modifica anche la selezione della ricorrenza e dell'output di attenzione. Quindi la prova riguarda percorsi completi e non localizza l'errore in un singolo kernel. [Generatore delle varianti](ablate.py) · [Log](ablation.log)

| Percorsi riportati al calcolo per token | Massimo assoluto | RMSE peggiore | Argmax concordi |
|---|---:|---:|---:|
| Nessuno | 0,008620 | 0,001478 | 8/8 |
| FFN | 0,008939 | 0,001571 | 8/8 |
| DeltaNet | 0,009545 | 0,001918 | 8/8 |
| FFN e DeltaNet | 0,001039 | 0,000151 | 8/8 |

### Verifica su prefissi distinti

Dopo aver misurato il ritorno al percorso per token in tutti gli strati, abbiamo provato intervalli contigui iniziali o finali, con entrambi i percorsi FFN e DeltaNet disabilitati per il batch solo nell'intervallo. Dei sette intervalli non vuoti esaminati sul C, soltanto gli strati 0–47 rientrano in entrambe le soglie. Il controllo iniziale e tutti i 7 esperimenti riproducono gli otto argmax. La selezione era fissata prima della prova sul web. [Intervalli](layers.py) · [Risultati C](layers.log)

Sul blocco indipendente a posizione 3187 della risposta web (20 token già generati), gli strati 0–47 riportano la RMSE peggiore da 0,001282 a 0,000899 ma lasciano un errore massimo di 0,005591. Il limite è 0,005. Gli otto argmax coincidono. Abbiamo interrotto la prova prestazionale per questa combinazione. Il successivo controllo su tutti i 64 strati è un esperimento distinto, deciso dopo questo risultato. [Risultati web](holdout.log) · [Regola di selezione](holdout.py)

Con entrambi i percorsi per token in tutti i 64 strati il blocco web scende a 0,001614 di errore massimo e 0,000212 di RMSE peggiore. Su un terzo blocco, dopo 20 token della risposta Git a posizione 1976, il massimo scende da 0,049967 a 0,004303 e la RMSE da 0,007853 a 0,000672. Entrambi i percorsi conservano gli otto argmax anche su Git. Questa variante supera le due soglie nei tre blocchi esaminati. Le prove includono soltanto 24 vocabolari completi. [Controllo web](full-web.log) · [Controllo Git](full-git.log)

Abbiamo cronometrato anche il generatore con il ritorno al percorso per token nei 64 strati. Sul C, due processamenti speculativi dello stesso prompt accettano tutti i 14 token proposti e riproducono tutti i 64 token emessi. In ordine OFF/ON/ON/OFF le velocità sono 5,947 / 5,618 / 5,514 / 5,398 tok/s. La variazione aggregata di throughput è -1.66%, con segno opposto nelle due coppie e controlli variabili. Per entrambe le serie il delta si calcola sui token totali divisi per il tempo totale di ciascuna modalità. Il +7,78% del primo prototipo C riguarda il percorso che supera le soglie di errore: la variante che le rispetta in tutti e tre i blocchi richiede ancora un vantaggio prestazionale replicabile. [Runner](selected.py) · [Tempi](full-timing.log)

### Riproduzione del prototipo

Dalla root di qw3 con la stessa build Metal, la seguente sequenza ripete il confronto sul prompt C. La generazione greedy diagnostica usa i kernel di qw3. Il salvataggio portabile prima di ogni prova e il test di rollback precedono il timer.

```sh
d=docs/measurements/dense27-lookup-2026-11-19
tmp=$(mktemp -d)
cc -O3 -mcpu=native -ffast-math -fobjc-arc -I. -c "$d/tail.m" -o "$tmp/tail.o"
cc -O3 -mcpu=native -ffast-math -std=c99 -I. "$d/spec.c" "$tmp/tail.o" \
  -lm -pthread -framework Foundation -framework Metal -o "$tmp/spec"
export QW3_METAL_NGL=64 QW3_METAL_KV_F16=1 QW3_METAL_IQ3_NIBBLE=1 QW3_METAL_DENSE_PARTIAL_TILES=1
python3 -c 'import json,sys; print(*json.loads(open(sys.argv[1]).readline())["prompt"])' "$d/coding.jsonl" \
  | caffeinate -disu "$tmp/spec" ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 64
```
