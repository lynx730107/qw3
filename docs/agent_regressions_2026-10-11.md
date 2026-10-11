# Regressioni dell'agente e ricerca web

Passano tutti i 19 target eseguiti e le due prove web successive.
L'agente e la CLI sono ricompilati forzando i sorgenti principali; la build
non produce warning. Non emerge una regressione nei controlli indicati e
non è necessaria alcuna modifica al codice di produzione.

| Ambito | Target passati |
| --- | --- |
| Strumenti e guardie | bash, edit, color, context, repeat, status, tool-parse, trace, verify |
| Fingerprint | runtime-fingerprint |
| Modello e stato | dense-agent, prefix-cache, checkpoint, checkpoint-compatibility |
| Uso reale dei tool | coding-smoke, python-smoke, git-smoke |
| Conversazione e runtime | repeat-format, runtime-regression |

I test con modello usano lo stesso `Qwen3.8-27B-UD-Q3_K_XL.gguf`, KV F16,
64 layer Metal, kernel di produzione, K64 prefill e cache64 DeltaNet.
Gli override privati sono rimossi; nessun candidato IQ3_S viene attivato.
La validazione API Metal è abilitata. I target sono sequenziali; i processi
con modello non sono avviati in parallelo e non si usano copie dei pesi.
I test generali usano ctx4096; dense-agent mantiene ctx256 e Python, Git,
formattazione e web mantengono ctx8192.

Si verificano strumenti reali in directory temporanee isolate: scrittura
e compilazione C, esecuzione Python e seguito della conversazione, lettura
delle tre commit Git con cache fredda/calda/disabilitata, diagramma Unicode
e separatore ASCII dopo un tool, ripresa dei checkpoint e recupero di cache
corrotta. La compatibilità verifica anche spostamento di sorgenti identici
e rifiuto del checkpoint quando i kernel caricati cambiano, con
ricostruzione dal transcript e successiva ripresa.

I 19 target richiedono complessivamente 286,74 secondi. Lo swap è
1309,06 MiB prima e dopo ciascuno. Sorgenti, eseguibili e dimensione/mtime
del modello risultano immutati durante la suite; `qw3-bench` conserva
l'hash della serie di riferimento. Il [manifest](measurements/agent_regressions_2026-10-11/manifest.json)
registra ambiente, comandi, durata, exit code, swap, hash e snapshot.
Tutti i log e gli artefatti di formattazione sono conservati. Questa è la
verifica dei target dell'agente indicati, non l'intera `test-regression-full`
o la validazione dei kernel privati.

## Prove web reali

Una chiamata diretta al tool `search_web` restituisce cinque risultati.
Segue una conversazione completa, con questa richiesta: verificare la
banda di memoria dichiarata per Apple M5 base, prima cercando sul web e poi
aprendo una fonte ufficiale trovata nei risultati; restituire chip, numero
in GB/s e URL visitato in JSON, senza usare la memoria o bash.

L'agente effettua due `search_web` e due `visit_url`. La prima pagina
riguarda Pro/Max: riconosce la discrepanza e cerca di nuovo. La risposta
finale è `Apple M5`, 153 GB/s e
[le specifiche Apple del MacBook Air](https://www.apple.com/macbook-air/specs/),
effettivamente visitate e contenenti il valore. La selezione iniziale
della fonte richiede una correzione: non si descrive un percorso riuscito
con una sola ricerca. Passa anche la riduzione automatica del contesto,
da 7037 a 2072 token, prima della risposta finale. Il normale prefisso
vuoto `<think>...</think>` del transcript è escluso quando si valida il
solo JSON dell'ultima risposta. Durata 220,72 secondi; nessun tool rifiutato.

Il test già presente `test_agent_llama_reference.py --candidate-only`
verifica inoltre una singola visita all'API GitHub: nome del repository,
branch predefinito, licenza e flag archived coincidono con l'oracolo live.
Passa in 35,24 secondi. Non viene avviato llama-server o un secondo modello;
questo esito non è un confronto qualitativo con llama. Lo swap resta
1309,06 MiB nel test di ricerca e passa a 1301,06 MiB nel test URL. Non si
attribuisce la diminuzione a un'ottimizzazione.

[Manifest della conversazione web](measurements/agent_web_2026-10-11/manifest.json),
[manifest del test URL](measurements/agent_web_2026-10-11/url_reference_manifest.json)
e [validazione](measurements/agent_web_2026-10-11/analysis.json) conservano
prompt esatto, comandi, stdout/stderr, trace e transcript. Gli snapshot
immutabili riusano quelli della suite quando i sorgenti coincidono.

Queste verifiche precedono la ripresa delle prove di decoding. Il
[vincolo di memoria e identità bit per bit](m5_decode_headroom_2026-10-11.md)
resta valido; nessun nuovo candidato è adottato da questi test.
