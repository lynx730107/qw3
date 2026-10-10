# M5: sincronizzazioni esplicite per buffer di sessione (2026-10-10)

Il requisito dell'utente resta l'identità bit per bit. Nessun kernel,
precisione o algoritmo viene modificato. La prova privata cambia soltanto
le allocazioni effettuate da `qw3_metal_new_private_buffer` e le dipendenze
tra comandi. La baseline usa i buffer tracked nativi.

## Controlli brevi

Su due sessioni ctx512, GPU64, KV F16, prefisso seed1 di
un token (numero di thread CPU non forzato nel test di qualità), la
validazione API Metal è abilitata (`MTL_DEBUG_LAYER=1`).
Tre varianti hanno controlli sulla modalità effettiva delle allocazioni:

| Variante | Esito |
| --- | --- |
| Barriere prima di ogni dispatch, senza fence tra encoder | Fallisce al primo logit: 6,95593739 contro 0,985501587 |
| Fence tra encoder, senza barriere aggiuntive tra dispatch | Prefisso, checkpoint e 32 continuazioni esatti |
| Entrambe | Prefisso, checkpoint e 32 continuazioni esatti |

Le fence attendono all'apertura di ogni encoder compute/blit e vengono
aggiornate immediatamente prima della chiusura. Non si attende una fence
dopo averla aggiornata nello stesso encoder, come richiesto dall'SDK.
Il generatore intercetta 47 aperture/chiusure blit e i tre punti di chiusura
compute. L'altra opzione intercetta 165 siti di dispatch, inclusi quelli
che usano receiver `direct` e `round_enc`. Tutti i pesi e i buffer Shared
rimangono quelli nativi.

Per le varianti 2 e 3: 1781 attese, 1782 aggiornamenti, 68 allocazioni
tracked e 68 untracked, 207.149.568 byte per modalità (somma delle
allocazioni nel corso del test, non picco residente). La variante 3
aggiunge 32.338 barriere; la 2 nessuna. Intero vocabolario finito e bit
exact dopo ogni forward, stato canonico K/V, DeltaNet e convoluzione
esatto. [Sorgenti, log e manifest](measurements/hazard_fences_2026-10-10/manifest.json).

## Chiamata API nativa non valida individuata

La prima esecuzione completa della variante 2 supera i prefissi 1 e 15,
poi la validazione Metal abortisce al prefisso 16: `didModifyRange` viene
chiamato sul buffer Shared della maschera Flash Attention. L'header
`MTLBuffer.h` dell'SDK locale specifica che la chiamata è ammessa soltanto
per i buffer Managed. Il log originale con exit code -6 resta conservato;
non è un fallimento numerico attribuito alle fence.

La produzione ora chiama `didModifyRange` solo se la modalità reale è
Managed. I byte scritti nella maschera, i kernel e tutti i parametri
rimangono invariati. Un controllo isolato usa buffer tracked in entrambe
le sessioni e confronta la chiamata Shared precedente con il nuovo guard,
con la validazione API disabilitata solo per poter eseguire la vecchia
chiamata. Gli otto prefissi 1/15/16/31/127/128/129/159, l'estensione
128→159, gli stati canonici e il vocabolario completo sono esatti bit per
bit. Ogni prefisso include roundtrip del checkpoint e 32 continuazioni.
Sono esercitate sette scritture della maschera per modalità; nessuna
fence o barriera aggiuntiva, 220 allocazioni tutte tracked.

Il formato dei checkpoint non cambia. Il fingerprint del runtime cambia
con il sorgente host, secondo la politica conservativa già esistente;
il confronto numerico non aggira i controlli di compatibilità dei build.
La compilazione della produzione è senza warning e i due test del
fingerprint passano. Il test di compatibilità dell'agent passa con il
modello 27B e ctx4096. Un primo tentativo con ctx512 fallisce prima
dell'inferenza perché il prompt dell'agent richiede circa 1521 token;
il log resta conservato e non viene attribuito alla modifica della maschera.
[Controlli dopo la correzione](measurements/hazard_fences_fixed_2026-10-10/manifest.json).

Anche le fence superano tutti gli otto prefissi, l'estensione, i checkpoint
e le 32 continuazioni per prefisso con la validazione API Metal attiva.
Sono registrate 26.526 attese, 26.527 aggiornamenti, zero barriere aggiuntive;
110 allocazioni tracked e altrettante untracked, 348.053.248 byte cumulati
per modalità. Sorgenti e binario restano invariati. Non viene effettuata
alcuna adozione prima delle misure del decode ordinario.

Il primo confronto ordinario prevede tre blocchi OFF/ON/OFF, tre
ripetizioni tg32 a profondità128 per blocco, sessione singola ctx512,
GPU64, KV F16, opzione `-t 4`, input fissi seed1 e warmup stock.
Pausa di 60 secondi prima di ogni blocco; contatori verificano sia
le modalità reali delle allocazioni sia l'attivazione delle fence.
I nove campioni sono conservati. OFF misura 8,943381 token/s e ON
8,937973 token/s: **−0,06047%, effetto neutro**. Le medie OFF sono
3578,84 e 3577,29 ms; dispersione 0,0433%. Sorgenti e binario invariati,
zero campioni scartati. Lo swap non resta invariato in tutti i processi;
i valori prima/dopo sono conservati nel manifest e questa prova non
viene usata per dimostrare un guadagno stabile.

Il controllo breve non mostra un segnale da approfondire con una serie
più ampia. La variante con fence rimane privata; la produzione mantiene
il tracking dei buffer di sessione. Solo il guard Managed della maschera
viene adottato, senza attribuirgli un guadagno di throughput.
[Controllo ordinario completo](measurements/hazard_fences_native_screen_2026-10-10/results.json).

Il runner dei candidati ora salva stdout/stderr e metadati prima di
sollevare un errore per exit code non nullo. Un eseguibile fittizio che
stampa sui due stream ed esce con 127 verifica il ramo senza inferenza
GPU. [Controllo del runner](measurements/candidate_failure_logging_2026-10-10/run/results.json).

```sh
make qw3-hazard-fences-test qw3-hazard-fences-native-bench
# ID 0: chiamata vecchia contro guard Managed; tutti i buffer tracked.
MTL_DEBUG_LAYER=0 QW3_METAL_KV_F16=1 QW3_METAL_NGL=64 ./qw3-hazard-fences-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf all 0
# ID 2: native tracked contro untracked con fence; validazione API attiva.
MTL_DEBUG_LAYER=1 QW3_METAL_KV_F16=1 QW3_METAL_NGL=64 ./qw3-hazard-fences-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf all 2
```
