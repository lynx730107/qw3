# M5: margine del decoding e vincolo di memoria

L'obiettivo resta eguagliare e possibilmente superare llama.cpp con lo
stesso modello e precisione. I risultati devono rimanere identici bit per
bit. Dal 2026-10-11 sono escluse anche ottimizzazioni che aumentano la
memoria dei pesi o dei buffer di sessione: nessuna copia preparata, cache
aggiuntiva o espansione dello scratch. Le copie IQ3_S Private da 3,53 GB e
il layout allineato da 3,59 GB restano soltanto esperimenti archiviati e non
devono essere riproposti come candidati adottabili sotto questo vincolo.
La verifica del modello dovrà evitare sessioni duplicate simultanee quando
può usare un oracolo su disco e una sola sessione alla volta.

Il [confronto diretto bilanciato](m5_native_reference_run_cooldown_2026-10-11.md)
misura qw3 a 8,86247 e llama a 9,13410 token/s, tg32 a profondità 128.
Servono +3,065% di throughput, oppure circa 3,36 ms in meno per token.
Entrambi i motori hanno dispersione delle medie intorno allo 0,60%.

L'[inventario dei tensori effettivamente associati al motore](measurements/projection_cost_2026-10-04/inventory.json)
conta 12.238.086.296 byte di pesi per token assumendo una lettura di ogni
tensore utilizzato e una sola riga dell'embedding. Il GGUF da 13,15 GB
contiene anche dati che non vengono letti integralmente a ogni token.
L'inventario non misura transazioni DRAM: cache e riletture cambiano il
traffico fisico.

[Apple dichiara 153 GB/s per M5](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/).
Con i soli pesi e la precedente assunzione, `153e9 / 12238086296` produce
un tetto nominale di 12,50 token/s. Le [precedenti misure locali di streaming](m5_projection_prefill_2026-10-04.md)
su buffer da 256–512 MiB hanno medie di 129,00–129,26 GB/s, con dispersioni
interne e fra processi inferiori al 5%. Divise per gli stessi pesi, danno
10,54–10,56 token/s. Sono evidenze archiviate del 2026-10-04, non nuove
misure effettuate dopo il vincolo di memoria.

Il prodotto dei pesi per la velocità qw3 del confronto diretto è circa
108,46 GB/s di banda equivalente. Non dimostra né quella banda DRAM reale
né una percentuale di saturazione. I tetti precedenti ignorano
dequantizzazione, calcolo, attivazioni, KV, stato ricorrente, riletture,
barriere e scheduling. Non costituiscono una previsione raggiungibile e
non dimostrano un guadagno disponibile del 19% o del 41%.

Questi dati consentono di cercare ancora un miglioramento di pochi punti
percentuali nel decoding; non provano che il vincolo bit per bit permetta
di ottenerlo. La [profilazione diagnostica](m5_native_decode_profile_2026-10-10.md)
individua le proiezioni IQ4_XS e IQ3_S come bersagli maggiori. Le prossime
prove dovranno conservare i buffer originali, ridurre lavoro o accessi nei
kernel e superare prima la verifica numerica. Ogni candidato positivo
richiede controlli ordinari e un nuovo confronto diretto; velocità qw3 e
llama prese da campagne diverse non dimostrano un sorpasso.
