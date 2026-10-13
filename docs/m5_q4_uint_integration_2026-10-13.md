# Integrazione Q4_K uint nel profilo dense M5

Il kernel pubblico `qw3_matvec_q4_k_words` è identico alla variante uint
validata, tranne il nome. Conserva l'ordine scalare FP32, la sequenza K,
le due riduzioni SIMD, la barriera e lo scratch originali. Il kernel byte
resta disponibile. Non si aggiungono copie dei pesi, cache o buffer di
sessione; si aggiunge un oggetto di pipeline e restano i temporanei CPU
di compilazione. Non è stata misurata la RAM totale né il suo picco.

La selezione automatica è limitata a Apple M5 e al profilo misurato:
16 layer full, 48 lineari e buffer di embedding di 5120 float. Richiede
base dei pesi e stride della riga allineati a quattro byte. Un pipeline
opzionale assente, un altro profilo o una vista non allineata mantengono
il kernel byte. Griglia e thread restano originali. Il pipeline viene
selezionato direttamente nelle tre route generiche che eseguono le
proiezioni Q4_K del modello, senza il proxy della prova privata.

Nel worktree isolato i tre eseguibili e il test di qualità compilano
senza warning. Il test usa un engine e una sola sessione ctx256,
alternando il selettore pubblico disabilitato/abilitato con oracoli su
disco. Passano otto prefissi 1/15/16/31/127/128/129/159, estensione
128→159 e 32 continuazioni per fase. Intero vocabolario finito, history
e stato canonico coincidono bit per bit. Passano 16 roundtrip di
checkpoint. Ogni seguito abilitato registra 1152 selezioni effettive
del pipeline pubblico e zero fallback. Metal API Validation è attiva.
Il tail temporaneo Flash Attention è escluso dal confronto canonico.

Passano i controlli bash, edit e fingerprint del runtime, oltre al test
dense dell'agente: rendering, validazione dell'offload/KV, generazione,
checkpoint salvato/ripreso e prefix cache creata/ripresa. I test usano
store temporanei, senza scrivere nello store personale. Sono prove
specifiche, non l'intera suite di regressione per ogni modello/backend.

[Archivio dell'integrazione](measurements/q4_uint_integration_2026-10-13/manifest.json)
conserva 24 sorgenti congelati, comandi, log e hash dei binari. Il
verificatore confronta il kernel pubblico con quello della prova
numerica già archiviata e verifica i sedici risultati e i quattro
controlli dell'agente. Le due sorgenti dei test fingerprint sono anche
confrontate con i blob del commit di base. L'archivio non conserva
binari, GGUF o checkpoint.

Il guadagno privato di +0,934% è documentato nelle
[prove Q4_K word](m5_q4_words_2026-10-13.md). Non è una misura diretta
del vantaggio della versione integrata rispetto a llama. Quel confronto
su pp128 e tg32 a profondità 128 è avviato separatamente, con processi
sequenziali e lo stesso GGUF/precisione. Le sorgenti dell'integrazione
sono ora nel branch corrente, commit `1dc41c8`, dopo le verifiche nel
worktree isolato. Il confronto usa ancora i binari congelati di quel
worktree; la ricostruzione dei binari del branch avverrà al suo termine.
L'obiettivo resta aperto.
