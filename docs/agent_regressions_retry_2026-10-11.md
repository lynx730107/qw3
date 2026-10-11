# Nuova verifica delle regressioni dell'agente, 11 ottobre 2026

Passano tutti i 21 controlli eseguiti sul branch a `7d1a49e`: i 19 target
della precedente suite, il test di retention dello store e una visita URL
con oracolo live. Non emerge una regressione negli ambiti coperti. Nessuna
modifica al runtime è necessaria per questi test.

| Ambito | Controlli passati |
| --- | --- |
| Tool e guardie | bash, edit, color, context, repeat, status, tool-parse, trace, verify |
| Fingerprint e store | runtime-fingerprint; cache 4, archivi 8, history 1.000 |
| Modello e stato | dense-agent, prefix-cache, checkpoint, checkpoint-compatibility |
| Coding | scrittura/compilazione/esecuzione C; Python con seguito della conversazione; Git con cache fredda/calda/disabilitata |
| Conversazione | diagramma Unicode e separatore ASCII dopo read, ripresa del checkpoint, runtime CLI/tool |
| Web | una chiamata visit_url; quattro campi della risposta finale coincidono con l'API GitHub letta dal vivo |

I controlli usano gli stessi binari di produzione della suite precedente,
lo stesso `Qwen3.8-27B-UD-Q3_K_XL.gguf`, KV F16, 64 layer Metal e i default
packed8, K64 prefill e cache64 DeltaNet. Gli override privati vengono
rimossi. La validazione API Metal è abilitata. I processi sono sequenziali,
con un solo modello alla volta e senza copie aggiuntive dei pesi.
I contesti dei test restano quelli previsti: 256 per dense-agent,
4.096 per i test generali e 8.192 per Python, Git, formattazione e URL.

I 21 controlli impiegano complessivamente 286,32 secondi. Lo swap rimane
a 1.165,06 MiB prima e dopo ogni controllo. Sorgenti, binari e dimensione/mtime
del modello restano invariati. Questo controllo ai confini dei processi
non misura il picco di memoria. Lo store personale resta identico: soltanto
la history di 30.109 byte, con hash invariato. I test con conversazioni e
checkpoint usano directory temporanee rimosse al termine. Sono conservati
solo log e artefatti testuali della fixture sintetica di formattazione.

Il test URL usa `test_agent_llama_reference.py --candidate-only`: non avvia
llama-server o un secondo modello. La risposta finale coincide con l'oracolo
per `full_name`, `default_branch`, `license` e `archived`. La prova non è
un confronto qualitativo con llama e non ripete la conversazione search_web
sulla banda M5 della [verifica precedente](agent_regressions_2026-10-11.md).

Il primo preflight del controller usava `make -q`, che restituisce non-zero
per la dipendenza `FORCE` del fingerprint anche con binari invariati.
Corretto il controller, la regola viene eseguita e gli hash sono verificati
prima di avviare i test. Il fingerprint non cambia; nessun test dell'agente
era stato avviato nel tentativo iniziale.

Il [manifest](measurements/agent_regressions_retry_2026-10-11/manifest.json)
conserva ambiente, comandi, risultati, durata, swap, hash e riferimenti agli
snapshot immutabili già indicizzati. Il
[controller](measurements/agent_regressions_retry_2026-10-11/run_suite.py),
tutti i log e gli artefatti sintetici sono archiviati. La verifica copre
questi target dell'agente, senza sostituire l'intera `test-regression-full`
o la validazione numerica di un nuovo candidato di ottimizzazione.
