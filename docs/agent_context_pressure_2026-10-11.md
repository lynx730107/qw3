# Risultati dei tool, compattazione e interruzione

La compattazione riduceva il vecchio transcript, poi reinseriva integralmente
il risultato del tool che aveva fatto superare il contesto. Un `read` di HTML
poteva quindi riportare subito il prompt a decine di migliaia di token.
Il prefill non controllava la richiesta di interruzione tra i batch; durante
la generazione del riassunto il footer poteva inoltre restare su `prefill`.

Un secondo difetto selezionava la coda recente soltanto a partire da un nuovo
messaggio dell'utente. In un singolo lungo turno con molti tool questo
restituiva la fine del transcript: `tail=0`, perdendo ogni scambio recente.
Il riassunto poteva quindi perdere il punto raggiunto e indurre una nuova
scansione dello stesso intervallo.

Gli archivi locali mostrano un risultato HTML da 29.229 token su un transcript
da 31.796 e un altro da 9.825 token nel transcript da 12.504 dello screenshot.
La [diagnostica](measurements/agent_context_pressure_2026-10-11/production_archive_diagnostic.json)
conserva soltanto conteggi, dimensioni, timestamp e hash. Non copia il testo
della conversazione. Questo individua il lavoro ripetuto sul contesto; GPU al
100% da sola non dimostra un deadlock e non è stato profilato quel processo.

## Correzione

Prima di valutare se compattare, ogni risposta di tool nativo o DSML è misurata
con il tokenizer e il wrapper chat effettivi. Il limite è `ctx/8`, con minimo
128 e massimo 2.048 token, incluso il wrapper. I risultati più grandi vengono
salvati integralmente in `spills/tool-*`, con permessi 0600, mantenendo gli
ultimi otto file. Nel contesto entra un estratto UTF-8 con inizio, fine e
percorso del file. I risultati piccoli restano identici. Un errore di scrittura
viene dichiarato esplicitamente. È un limite sul numero di file, non sui byte.

Anche `read` e `more` misurano la risposta con il tokenizer: restituiscono
soltanto righe complete che entrano nello stesso budget proporzionale al
contesto. Il parametro `lines` è un massimo richiesto. La risposta limitata
riporta `returned_lines` e `next_start`; la continuazione riparte dalla prima
riga non letta. Una singola riga troppo lunga viene segnalata senza saltarla,
con l'indicazione di convertire HTML o estrarre campi più corti su disco.
I limiti preesistenti di navigazione nei sorgenti restano applicati.

`qw3_session_sync_interruptible` controlla un callback prima di ogni batch o
token. L'interruzione drena il lavoro Metal già inviato e restituisce `-2`;
il prefisso completato può essere ripreso con una successiva sync. L'API
precedente resta disponibile senza callback. Il percorso aritmetico e tutti
i kernel sono invariati; non vengono aggiunti buffer di pesi o di sessione.
Le allocazioni temporanee CPU del tokenizer e degli estratti restano presenti.

La sync iniziale e la ricostruzione dopo compattazione usano progressi e
interruzione. La sintesi del riassunto mostra `compacting`. `/compact` passa
al worker, lasciando libero il terminale per `/interrupt`. Il worker conserva
una richiesta arrivata prima dell'avvio del job e tratta la cancellazione
come conclusione normale, saltando l'autosave. Anche la sync del checkpoint
controlla l'interruzione.

La coda può ora cominciare anche da un messaggio completo dell'assistente:
conserva la richiesta con i risultati successivi o la richiesta in attesa del
risultato che verrà appeso dopo compattazione. Non comincia da un risultato di
tool orfano. Mantiene il budget preesistente per la coda e il controllo sul
risparmio di token. Il cursore `last_read_path`/`last_read_next`, già presente
nel runtime, viene incluso con percorso JSON escaped nel prompt di sintesi
e nel messaggio compattato: non dipende dalla memoria del riassunto. Il prompt
di sintesi richiede i campi già raccolti o il file compatto che li
contiene. Il prompt web suggerisce un file di risultati compatto quando una
lista deve sopravvivere alla compattazione. Non aggiunge campi di stato GPU.

Il transcript compattato riporta anche la richiesta attiva dell'utente, prima
della coda assistente/tool: il compito non è affidato soltanto al riassunto
generato. La richiesta viene mantenuta identica se entro `ctx/16`, massimo
512 token; per richieste più grandi resta un messaggio di ripresa con il
percorso dell'archivio completo. La selezione tiene conto dello spazio
effettivo per la coda. Il cursore distingue anche le chiamate `more` nella
protezione dai duplicati: parametri identici con cursore avanzato sono letture
diverse; ripetizioni senza avanzamento restano bloccate. L'ordinale consente
più `more` consecutivi nello stesso batch.

La cancellazione resta cooperativa: attende il batch Metal in corso. Non
è una preemption della GPU. La preparazione del prefix cache all'avvio e
i comandi sincroni eseguiti a terminale inattivo non sono coperti da questa
prova di interruzione.

## Lettura delle pagine web

Il system prompt richiede di cercare nel file della pagina completa e leggere
soltanto intervalli pertinenti con `read`/`more` e `next_start`. Se è disponibile
solo un prefisso, chiede il download completo su disco. Distingue la pagina
completa dalla copia di un risultato già troncato; dimezza gli estratti bash
ancora troppo grandi ed evita `cat` e ripetizioni della stessa estrazione.

È emerso anche un difetto nel tool: la rimozione dei tag con una regex lasciava
attributi HTML quando contenevano `>` tra virgolette. `visit_url` ora usa
`html.parser.HTMLParser`, esclude gli attributi e i blocchi script/style/nav,
conserva le divisioni fra titoli e voci e salva il testo leggibile completo
in `spills/web-*` prima di restituire la preview da 8.000 caratteri. Mantiene
otto pagine con permessi 0600. Il cap di download preesistente resta 2 MB;
una pagina oltre quel limite viene dichiarata incompleta e non salvata come
completa. Gli errori di fetch o scrittura sono espliciti. Il JSON conserva
anche il testo fra parentesi angolari. Restano allocazioni temporanee CPU
per il download e il parsing; non vengono aggiunti buffer del modello.
La proposta iniziale di 20 righe fisse è sostituita dal budget adattivo,
come richiesto durante il lavoro.

## Verifiche

Le prove usano il 27B GGUF originale, Metal con KV F16, validazione API,
processi sequenziali e store temporanei. Nessun secondo modello è caricato
in parallelo. Il test numerico usa una sola sessione ctx256 riutilizzata e
un oracolo su disco, senza una seconda sessione GPU.

Il primo test mirato passa: HTML sintetico da 21.820 token, limiti effettivi
di 2.048 e 512, UTF-8 valido, file completo byte per byte, permessi 0600,
rolling a otto e fallimento di salvataggio esplicito. Un vero `read` inserito
a 12.504 token arriva a 14.336, aggiungendo 1.832 token senza compattazione.
La sync ordinaria, quella con callback non attivo e quella interrotta a 128
token e ripresa fino a 192 coincidono su 33 vettori completi di logits
finiti, 32 token greedy, KV canonico, DeltaNet, convoluzione e token/history.
Sono verificati anche cancellazione immediata prima della divergenza,
pulizia del progresso e ripristino del transcript dell'agente.

Il test finale estende queste verifiche alla lettura adattiva: sullo stesso
testo ctx4096 restituisce 14 righe e ctx32768 ne restituisce 67. `more` copre
tutte le 120 righe senza salti; una riga troppo lunga mantiene `next_start=1`
e la riga successiva resta leggibile esplicitamente. Queste due capacità
sono configurazioni del budget CPU; non vengono create sessioni GPU da
4.096 o 32.768 token in questo test. Il vero `read` HTML finale passa da
12.504 a 14.490 token, aggiungendone 1.986 senza compattazione. Le tre prove
numeriche con la sessione riutilizzata passano nuovamente.

Il terminale reale passa: interruzione del prefill ordinario in 0,930 secondi
e del prefill di `/compact` in 0,860 secondi. Nessun checkpoint viene scritto
dopo la cancellazione; `/quit` termina con exit code 0. Sono latenze di questa
prova ctx4096, non un limite massimo garantito per tutti i batch o contesti.

I log iniziali conservano due errori del controller PTY (invio prima
dell'inizializzazione del terminale e mancato drenaggio durante l'uscita).
Una terza prova ha rivelato l'exit code 1 dopo cancellazione, corretto nel
worker. Il controller Wikipedia iniziale ha assunto erroneamente che tutti
i titoli fossero in corsivo; quattro voci sono invece testo normale.
Il primo test completo di Gadda, limitato a otto chiamate ai tool, ha
scaricato la pagina e letto estratti, ma non ha raggiunto l'elenco prima
di esaurire quel limite. Anche questo esito è conservato.

La prova successiva con 24 chiamate e il primo prompt ha terminato una
compattazione in 109 secondi (7.530 -> 2.078 token), poi ha ripetuto estrazioni
troppo grandi e avviato un'altra compattazione. È stata interrotta dopo
721,93 secondi per correggere quel comportamento; non è un test riuscito.
Un'altra prova con il prompt da 20 righe è stata interrotta dopo 251,07 secondi
quando l'utente ha richiesto l'adattamento al contesto. Trace, motivi di stop,
esiti e sorgenti delle varianti sono conservati. I singoli riassunti hanno un
limite di 768 token: il ciclo osservato consiste nel rilancio di compattazioni
fra tool, non dimostra che quella funzione abbia un ciclo senza fine.

La prova con lettura adattiva ma ancora il parser regex ha completato due
compattazioni in 183 e 203 secondi, da 7.127 a 2.566 e da 7.048 a 2.565 token.
La riga 37 dell'estratto HTML era troppo lunga per 1.024 token: `more` l'ha
segnalata senza saltarla. Il modello ha poi eseguito altre estrazioni bash,
arrivando a una terza compattazione; la prova è stata interrotta dopo
1.262,48 secondi, senza risposta finale. Questo ha motivato la sostituzione
del parser regex, senza considerare il solo limite dei tool una soluzione
completa alla prova web. Gli esiti delle 23 regressioni di quella variante,
i sorgenti congelati e il test interrotto restano in `gadda_adaptive_v4/`.

Il nuovo test di `visit_url`, senza modello, verifica 500 voci e attributi
contenenti parentesi angolari: il file leggibile completo coincide byte per
byte con l'oracolo, la preview è troncata, gli attributi e i blocchi nascosti
sono esclusi. Passano UTF-8, permessi, rolling a otto, JSON, download oltre
il cap e salvataggio impossibile.

La prima prova con il nuovo tool ha però scelto `curl` e convertito il file con
una regex: il percorso corretto era disponibile, ma non utilizzato dal modello.
Tre estratti bash sono stati limitati da oltre 2.900-4.200 a meno di 1.024 token.
La prova è stata interrotta dopo 185,97 secondi prima di una compattazione,
senza risposta finale, per rendere esplicita nel prompt la sequenza
`search_web` -> `visit_url` -> ricerca/lettura del file già pulito.
Le 24 regressioni di quella variante, trace, stop e snapshot restano in
`gadda_parser_v5/`.

La successiva prova ha usato correttamente `search_web`, `visit_url` e il file
leggibile. Ha completato due compattazioni (186 e 164 secondi), entrambe con
`tail=0`, poi ha ripetuto le letture da 119, 134, 151 e 167 e avviato la terza.
È stata interrotta dopo 1.091,43 secondi per correggere la selezione della coda,
senza risposta finale valida. Le 24 regressioni della variante, i sorgenti e
i log restano in `gadda_tail0_v6/`. Una lettura della memoria durante quella
prova ha rilevato swap 844,31 MiB; è un'osservazione, non una misura del picco
né un'attribuzione delle allocazioni.

Il test di ledger finale verifica un lungo turno con un solo messaggio
dell'utente: conserva la richiesta/risposta recente, conserva una chiamata
pendente, esclude risultati orfani e mantiene il cursore esatto anche con
virgolette, backslash e newline nel nome del file. La vecchia selezione avrebbe
restituito una coda vuota in questo caso.

La prova con coda conservata ha terminato due compattazioni in 244 e 175
secondi, mantenendo 1.066 e 1.117 token recenti. È terminata in 802,71 secondi,
ma non passa l'oracolo: soltanto quattro titoli principali corrispondono,
mancano 36 voci e la generazione ripete titoli fino all'intervento della
protezione. `more` era stato falsamente bloccato come identico pur essendo
avanzato il cursore; il messaggio della protezione invitava a rispondere subito.
La richiesta originale non compariva più come messaggio utente dopo la
compattazione. Questo ha motivato il controllo dei duplicati sullo stato di
lettura e il ripristino esplicito della richiesta. Log, risposta errata,
24 regressioni e sorgenti della variante restano in `gadda_tail_v7/`.

Il nuovo test di `more` esegue letture di un vero file di sette righe passando
dal controllo dei duplicati: tutte le righe vengono lette in ordine, anche
con due continuazioni identiche nello stesso batch; una ripetizione dopo EOF
senza avanzamento viene respinta. Il test di ledger controlla la selezione
della richiesta attiva. Il controller web finale conserva anche gli archivi
di compattazione del proprio test controllato, per verificare la richiesta
utente dopo ogni riassunto; non copia lo store personale. Gestisce e registra
anche il timeout, con una sola istanza del modello e un'asserzione di energia
`caffeinate` legata a quel processo.

Passano i 24 controlli della suite finale: le tre nuove prove mirate,
le 11 guardie, i nove target con modello e l'oracolo URL live. Durata totale
396,88 secondi; swap 844,31 MiB a tutti i confini delle prove.
Gli hash di 39 sorgenti, quattro binari, dimensione/mtime del GGUF e
l'inventario dello store personale sono immutati. Il [manifest](measurements/agent_context_pressure_2026-10-11/manifest.json)
registra comandi, ambiente, exit code, durata e riferimenti agli snapshot.
I modelli sono eseguiti sequenzialmente, con store temporanei isolati.
Non è l'intera `test-regression-full` e non è una nuova misura contro llama.

La ricerca finale su Wikipedia è **fallita rispetto all'oracolo di qualità**.
Usa ctx8192, prefix cache disabilitata, `--nothink`, un limite di 24 round
di tool e un massimo di 4096 token per generazione. Il processo del modello
termina normalmente, senza timeout, in 1.331,49 secondi; il controller
restituisce exit code 1 perché mancano 32 dei 40 titoli principali.
La [risposta integrale](measurements/agent_context_pressure_2026-10-11/gadda.stdout)
cita Wikipedia e contiene otto titoli corrispondenti, ma ripete opere e
attribuisce edizioni/date non presenti nella fonte. Le date e le altre
sezioni non sono certificate dall'oracolo dei titoli.

La sequenza reale usa `search_web`, `visit_url`, quattro comandi bash e otto
letture native, di cui una `more`. Il file leggibile completo contiene
tutti i 40 titoli. Le tre compattazioni terminano in 166, 211 e 220 secondi:
7.272 -> 3.865 token con coda 1.310; 7.122 -> 3.801 con coda 1.096;
7.108 -> 3.829 con coda 1.110. Le letture avanzano fino alla fine della
sezione senza il precedente blocco errato di `more`. Tutti e tre gli archivi
controllati conservano la richiesta originale dell'utente; sorgenti, binari
e store personale restano immutati.

Resta un limite distinto: i riassunti generati raggiungono il cap di 768
token e vengono accettati anche quando l'elenco raccolto è interrotto. Per
esempio, il secondo riassunto dice di aver letto le righe 105–156 ma conserva
soltanto i primi cinque titoli principali. Il testo completo resta negli
archivi su disco, ma il modello non lo recupera prima di rispondere. Conservare
cursore e richiesta evita la ripartenza osservata in precedenza, senza
garantire la conservazione dei fatti. La prova non consente di dichiarare
risolto il comportamento della ricerca dopo compattazione né di riprendere
l'ottimizzazione prestazionale come se l'agente avesse superato questa verifica.

Il [risultato](measurements/agent_context_pressure_2026-10-11/gadda-result.json)
e la trace registrano anche questo esito negativo. Il verificatore
`verify_archive.py` distingue integrità dell'archivio e 24 regressioni
tecniche dalla qualità della richiesta naturale (`gadda_quality_passed=false`).
Con `--require-web-pass` deve fallire su questa risposta; l'oracolo originale
del controller web resta rigoroso e non è stato modificato per far passare
l'elenco incompleto. Sono conservate anche tutte le prove precedenti.
