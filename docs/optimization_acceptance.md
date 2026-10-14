# Criteri di accettazione delle ottimizzazioni, versione 2

Dal 14 ottobre 2026, su richiesta dell'utente, l'identità bit per bit tra
baseline e candidato non è più un requisito generale. La priorità è un
agente affidabile, con logits numericamente validi e senza regressioni di
memoria. I risultati storici e le prove negative restano invariati.

## Controlli numerici

Le soglie iniziali sono in `tests/optimization_acceptance.h`. Per le
primitive si richiedono valori finiti e NMSE ≤ 1e-7. Per ogni vocabolario
completo si richiedono RMSE ≤ 0,001 e massimo errore assoluto ≤ 0,005:
sono i limiti numerici già presenti nel confronto dense con llama.
Non sono una garanzia universale di qualità, né sufficienti per adottare
una variante. Non si allargano automaticamente quando un candidato fallisce.

Si riportano anche errori massimi, differenze di bit e cambi di argmax.
Una differenza di token non è automaticamente un fallimento comportamentale.
I confronti dei logits usano gli stessi prefissi e token successivi della
baseline (teacher forcing), per evitare di confrontare contesti diversi.
Le prove dell'agente lasciano invece che il candidato generi liberamente.

Per lo stato numerico si misura la deriva (NMSE ≤ 1e-7 per blocco
confrontato). History, posizione, guardie, pesi e input immutabili restano
esatti. Un checkpoint salvato e ripreso dalla stessa variante deve
riprodurre esattamente logits e stato: la tolleranza non copre errori
di serializzazione o ripristino. Il tail temporaneo Flash resta escluso
dallo stato canonico, come nei controlli precedenti.

## Gate finale: comportamento dell'agente

Prima dell'adozione sono obbligatori confronti baseline/candidato
sequenziali sugli stessi compiti e con store temporanei. Si verifica
l'esito del compito, non l'identità della formulazione naturale.

- Creazione/modifica di file, esecuzione reale di comandi e risultato
  verificato sul filesystem (`test_agent_coding_smoke.sh`, bash ed edit).
- Conversazione Python in due turni: tool corretti, programma eseguito,
  risposta successiva corretta, nessun blocco spurio o verifica ripetuta
  (`test_agent_python_smoke.sh`).
- Web: risultato reale coerente con l'oracolo, JSON corretto e tool
  effettivamente usato (`test_agent_llama_reference.py --candidate-only`);
  parsing, pagine lunghe, spill e retention (`test_agent_web_page.py`).
- Pressione sul contesto, letture adattive, compattazione quando necessaria,
  interruzione/ripresa e assenza di loop (`test_agent_context_pressure.m`).
- Generazione, checkpoint e prefix cache salvati/ripresi
  (`test_dense_agent.sh`), più fingerprint del runtime.

Questi test devono essere applicati al backend candidato effettivo, con
prove di selezione del kernel. Un test `--tool-native` senza modello non
dimostra che il modello scelga correttamente un tool. Una prova su contesto
breve non dimostra l'assenza di derive su contesto lungo: estensione e
pressione sul contesto devono essere documentate prima dell'adozione.

Un solo engine/sessione modello alla volta. Restano vietati nuovi buffer
di pesi/sessione, copie/layout preparati dei pesi e crescita dello scratch
introdotti dall'ottimizzazione. La RAM totale/di picco non va dichiarata
invariata sulla sola base di `session_bytes`.

Solo dopo correttezza e comportamento accettabili si adotta un guadagno
di prestazioni ripetibile sul percorso ordinario. I segnali iniziali non
vengono sommati tra loro. Logits vicini, un benchmark veloce o un singolo
compito riuscito non sostituiscono gli altri controlli.

Il confronto candidato/llama sui logits resta richiesto prima dell'adozione:
la sola vicinanza a qw3 non prova che entrambi siano corretti. Il confronto
deve usare lo stesso GGUF, precisione e prefisso; anche i cambi di ranking
e le continuazioni vanno esaminati con le prove funzionali.

## Stato dell'applicazione

I nuovi probe GQA applicano questo criterio numerico e mantengono il
roundtrip esatto della stessa variante. Il loro manifest resta esplicito:
`agent_behavior_gate_passed=false`, `adoption_permitted=false`, finché non
sono completate le prove del candidato integrato. Le vecchie suite
bit-exact restano disponibili come strumenti diagnostici e per riprodurre
le verifiche storiche; non costituiscono più il gate generale dei nuovi
tentativi.
