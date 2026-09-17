
Voglio implementare una strategia di "Ricorsione a 1 livello" (noto anche come isolamento del contesto o approccio "Narnia") nel file qw3_agent.c. 

L'obiettivo è creare un nuovo tool nativo chiamato "sub_agent_analyze" che permette all'agente principale di lanciare un sotto-agente isolato ("lavagna pulita"). Questo sotto-agente eseguirà compiti di analisi pesante o lettura di file di grandi dimensioni su una sessione temporanea, restituendo all'agente principale solo il testo sintetico del risultato. In questo modo preserviamo la memoria (context window) del thread principale.

Per fare questo, scrivi il codice in C necessario seguendo queste specifiche tecniche:

1. Dichiarazione del Tool:
Nella funzione dove dichiari i tool nativi (es. native_tool_declarations), aggiungi la definizione per il tool "sub_agent_analyze". Deve accettare due parametri:
- "prompt" (stringa obbligatoria: le istruzioni dettagliate su cosa analizzare o cercare).
- "path" (stringa opzionale: il percorso di un file specifico da caricare e leggere nel sotto-contesto).

2. Logica di Esecuzione (tool_sub_agent_analyze):
Crea una funzione statica dedicata che gestisce l'esecuzione del tool. La funzione deve:
- Allocare una struttura `agent_state sub_agent;` locale sulla stack ed ereditare i puntatori fondamentali dall'agente padre (in particolare `sub_agent.engine = parent_agent->engine;` per condividere l'engine dei pesi del modello in RAM/VRAM senza ricaricarlo).
- Copiare i parametri di configurazione rilevanti (`sub_agent.cfg = parent_agent->cfg;`), ma impostare a FALSE l'abilitazione dei tool nel sotto-agente (`sub_agent.cfg.tools_enabled = false`) per blindare la ricorsione a un solo livello ed evitare loop infiniti.
- Creare una sessione di calcolo temporanea e pulita invocando `qw3_session_create` (usando la dimensione di contesto configurata).
- Inizializzare un transcriptor di token vuoto (`qw3_tokens`) per il sotto-agente, iniettando il system prompt standard.
- Se il parametro "path" è presente e valido, leggere il file di testo e concatenare il suo contenuto all'interno del messaggio di input per l'utente, seguito dal "prompt" di istruzioni.
- Eseguire la generazione del sotto-agente (invocando la funzione di generazione o di singolo turno già esistente nel file, assicurandoti di intercettare l'output di testo generato dall'assistente).
- Eseguire la pulizia tassativa di tutta la memoria locale allocata per il sotto-agente (`qw3_session_free` e `qw3_tokens_free`) prima di uscire, in modo da liberare la KV cache.
- Ritornare la stringa di output generata dal sotto-agente all'agente padre come risposta del tool.

3. Integrazione nel Dispatcher:
Nel punto del file in cui smisti le chiamate dei tool nativi (dove controlli il nome del tool tramite strcmp con i vari tool esistenti), aggiungi il ramo condizionale per "sub_agent_analyze" che intercetta la richiesta e invoca la funzione sopra descritta.

Genera il codice in modo che sia perfettamente coerente con i tipi, le macro, la gestione delle stringhe e lo stile di allocazione/free già presenti in qw3_agent.c.

NOTA AGGIUNTIVA PER I TOOL: 
Voglio che il sotto-agente possa riutilizzare i tool nativi già esistenti nel progetto (come edit, read, write, get_skeleton, get_function e semantic_search). 
Per fare questo, assicurati che 'sub_agent.cfg.tools_enabled' sia impostato su true. 
Tuttavia, per evitare loop ricorsivi infiniti, aggiungi un flag booleano 'is_sub_agent' nella struttura 'agent_state' (se non presente). Imposta 'sub_agent.is_sub_agent = true;' nel sotto-agente, e fai in modo che se un agente ha 'is_sub_agent == true', gli venga impedito di chiamare nuovamente il tool 'sub_agent_analyze'.

piano
 Piano d'Implementazione: Strumentazione della Ricorsione ad Agent Isolate in qw3_agent.c
Obiettivo
Introdurre un meccanismo di ricorsione a 1 livello tramite un nuovo tool nativo chiamato sub_agent_analyze. Questo tool istanzia un agente figlio temporaneo che condivide l'engine dei pesi dell'LLM (VRAM/RAM) ma alloca una sessione (qw3_session) e un transcriptor di token (qw3_tokens) completamente isolati. Al termine dell'esecuzione, la memoria della sessione figlia viene interamente liberata e solo l'output testuale finale viene restituito all'agente padre.

Passo 1: Modifica della Struttura agent_state
Per tracciare lo stato di ricorsione ed evitare loop infiniti (un sotto-agente che chiama un sotto-agente), dobbiamo introdurre un flag booleano nella struct principale.
1. Individua la definizione di typedef struct agent_state (o simile) all'inizio del file.
2. Aggiungi il seguente campo: bool is_sub_agent; // Impostato a true se la struct appartiene a un ciclo ricorsivo
3. Nel main(), dove l'agente principale a viene inizializzato, assicurati che venga azzerato: a.is_sub_agent = false;

Passo 2: Dichiarazione del Tool Nativo
Dobbiamo esporre il tool al modello principale (Qwen).
1. Trova la funzione responsabile della registrazione dei tool nativi (es. native_tool_declarations o dove sono definite le stringhe JSON/XML dei tool).
2. Aggiungi la definizione per sub_agent_analyze. Il tool deve accettare: • prompt (stringa, obbligatoria): Il compito analitico o la richiesta esplicita da risolvere. • path (stringa, opzionale): Il percorso di un file sorgente da caricare direttamente nel contesto isolato prima dell'analisi.
3. Logica di filtraggio: Assicurati che se a->is_sub_agent == true, questa dichiarazione venga omessa o che il tool non venga iniettato nel sistema, così il sotto-agente vedrà tutti i tool tradizionali (read, edit, write, semantic_search, ecc.) tranne se stesso.

Passo 3: Scrittura della Funzione del Tool (tool_sub_agent_analyze)
Crea la funzione statica che gestirà il ciclo di vita del sotto-agente. La funzione deve seguire questa precisa sequenza lineare:
1. Estrazione parametri: Recupera i valori di prompt e path dalla chiamata del tool tramite le funzioni helper del tuo framework (es. tool_param_value). Se prompt è vuoto, ritorna una stringa di errore.
2. Allocazione Stack: Dichiara una struct locale agent_state sub_agent; e inizializzala a zero con memset.
3. Ereditarietà dell'Engine (Zero-Copy): • Imposta sub_agent.engine = parent_agent->engine; (Fondamentale: non riallocare l'engine per non duplicare l'uso della memoria dei pesi). • Copia la configurazione: sub_agent.cfg = parent_agent->cfg;. • Imposta i puntatori di output/logging (output_write, output_ud, ecc.) in modo che i log del sotto-agente siano visibili sul terminale, magari associati a un colore differente (es. AGENT_COLOR_DIM) per distinguerli visivamente dal flusso principale.
4. Blindatura del Livello: Imposta sub_agent.is_sub_agent = true;.
5. Creazione Sessione Isolata: Chiama qw3_session_create(sub_agent.engine, sub_agent.cfg.ctx_size, ...) memorizzando il puntatore in sub_agent.session. Gestisci l'eventuale fallimento ritornando una stringa di errore pulita.
6. Inizializzazione Transcriptor: Inizializza sub_agent.transcript inserendo il system_prompt iniziale.
7. Iniezione del Contesto (File + Prompt): • Se il parametro path è popolato, leggi il file usando la funzione di lettura preesistente (es. read_file_text). • Costruisci una stringa contenente un header delimitatore (es. --- FILE CONTENT: %s ---), il testo del file, il footer delimitatore e, infine, accoda il prompt dell'utente. • Inserisci questa macro-stringa nel transcriptor della sotto-sessione come messaggio di tipo user.
8. Esecuzione del Turno: Invocare la funzione principale di inferenza del framework (es. run_agent_turn(&sub_agent, NULL) o generate_once(&sub_agent, &output_str)). Il sotto-agente eseguirà autonomamente il suo ciclo di ragionamento, chiamando i tool tradizionali (es. semantic_search o get_function) se necessario, accumulando token solo nella sua sessione.
9. Teardown e Deallocazione Tassativa: Al termine del turno (sia in caso di successo che di fallimento), invoca: • qw3_session_free(sub_agent.session); (Rilascia immediatamente la KV cache dal backend). • qw3_tokens_free(&sub_agent.transcript); (Svuota la memoria dei token intermedi). • Libera eventuali buffer temporanei usati per leggere il file.
10. Ritorno: Restituisce la stringa dell'ultimo messaggio dell'assistente generato dal sotto-agente. La stringa deve essere allocata dinamicamente (es. tramite agent_strdup), poiché il dispatcher del padre si occuperà di liberarla dopo averla iniettata nel flusso principale.

Passo 4: Modifica del Dispatcher dei Tool
Dobbiamo intercettare la chiamata nel flusso principale dell'agente padre.
1. Trova la funzione o il blocco if/else if che smista l'esecuzione dei tool in base a call->name (dove gestisci tool_read, tool_edit, ecc.).
2. Aggiungi il controllo per il nuovo tool: if (strcmp(call->name, "sub_agent_analyze") == 0) {     // Controllo di sicurezza a runtime (difesa in profondità)     if (a->is_sub_agent) {         return agent_strdup("error: sub_agent_analyze cannot be called by a sub-agent.");     }     return tool_sub_agent_analyze(a, call); }

Passo 5: Gestione dei Tool di Scrittura (edit / write) nel Sotto-Agente
Se desideri che il sotto-agente possa anche modificare attivamente i file su disco:
1. Lascia che il sotto-agente acceda normalmente a tool_edit e tool_write durante il suo turno ricorsivo.
2. Poiché i tool modificano i file direttamente sul file system, i cambiamenti saranno immediatamente persistiti su disco.
3. Quando il sotto-agente termina e ritorna il testo al padre, il padre non vedrà la cronologia delle modifiche riga per riga, ma riceverà solo il riepilogo finale (es. "Ho analizzato il problema e aggiornato la funzione X nel file Y per correggere il bug").
Se invece vuoi che il sotto-agente sia puramente analitico (Read-Only), aggiungi questa condizione nel dispatcher dei tool di scrittura:
if (a->is_sub_agent && (strcmp(call->name, "edit") == 0 || strcmp(call->name, "write") == 0)) {
    return agent_strdup("error: write operations are forbidden in a sub-agent context.");
}

 