# Voce locale con audio.cpp

Prima implementazione opzionale per macOS/Apple Silicon: microfono → Qwen3-ASR
0.6B → normale turno di `qw3-agent` → Kokoro 82M → altoparlanti. La lingua
predefinita è l'italiano, con voce `if_sara` (alternativa: `im_nicola`).

Lo stato del piano iniziale e i passaggi ancora aperti sono raccolti nella
[roadmap vocale](voice-roadmap.md).

## Installazione

Dalla radice del repository, con Xcode Command Line Tools, CMake, Git e Python 3:

```sh
make voice-setup
```

Il comando:

- compila `qw3-agent-voice` e l'utility diagnostica `qw3-voice`;
- scarica audio.cpp **v0.9.3**, commit
  `41c1f8ffa0d13a1b7e366e30ea5fe5ab8d169d48`, sotto `.deps/audio.cpp`;
- compila il servizio Metal con le sole famiglie `qwen3_asr` e `kokoro_tts`;
- compila e include i dati eSpeak-ng necessari a Kokoro in italiano, senza
  installare librerie di sistema;
- installa circa 1,3 GB di pesi audio Q8 tramite il model manager ufficiale;
- crea `config/audio-cpp.local.json` e `config/voice.local.json`, senza
  sovrascrivere configurazioni già presenti.

La versione del codice è fissata; i pacchetti GGUF vengono installati dal catalogo
upstream e il model manager ne registra revisione e metadati nei manifest sotto
`.deps/audio-models`. I pesi, i sorgenti esterni, le build e le configurazioni
locali sono esclusi da Git.

Passaggi separati: `make voice`, `make audio-server`,
`python3 tools/setup_voice_models.py`. La prima build scarica anche le dipendenze
del servizio. `QW3_AUDIO_BUILD_JOBS=2 make audio-server` limita la compilazione.

## Avvio

Primo terminale, dalla radice di `qw3`:

```sh
.deps/audio.cpp/build/qw3-metal/bin/audiocpp_server \
  --config config/audio-cpp.local.json --no-ui
```

Secondo terminale:

```sh
./qw3-agent-voice --voice-config config/voice.local.json --voice-check

./qw3-agent-voice \
  -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --ctx 8192 --kv-f16 --nothink \
  --voice-config config/voice.local.json
```

`--voice-check` controlla disponibilità e ID configurati sul servizio, senza
caricare il 35B o provare l'inferenza audio. Il servizio deve essere avviato
separatamente; l'agente non avvia processi in background.

Nella chat:

- **Invio** oppure `/voice` avvia la registrazione.
- Con le impostazioni predefinite, circa **1,2 secondi di silenzio** dopo aver
  parlato chiudono la registrazione e inviano la trascrizione come messaggio utente.
- Un altro **Invio** permette comunque di terminare la registrazione manualmente.
- **`/listen`** avvia turni vocali consecutivi: dopo la risposta parlata e una
  breve pausa, l'agente riapre il microfono. **Ctrl-C** ritorna alla tastiera.
- Se per 8 secondi non viene rilevata attività sufficiente, la registrazione
  viene scartata senza chiamare ASR o LLM e `/listen` torna alla tastiera.
- Il limite predefinito è 30 secondi, modificabile in `max_record_seconds`.
- Puoi anche scrivere normalmente e usare i comandi esistenti dell'agente.
- **`/speed 0.85`** rallenta la voce dalle prossime risposte; **`/speed 1`**
  seleziona la velocità normale. **`/speed`** mostra l'impostazione corrente e
  **`/speed default`** ripristina quella del servizio audio.cpp.
- **Ctrl-C** annulla registrazione, richiesta audio, generazione o riproduzione
  in corso; `/quit` chiude la chat.

Il microfono si apre soltanto quando avvii un turno o attivi `/listen`.
Ascolto, generazione e riproduzione avvengono in sequenza. In `/listen`, un
errore di registrazione/trascrizione o una risposta non riprodotta interrompono
la sequenza e riportano alla tastiera; il testo della conversazione resta disponibile.

Al primo avvio di una registrazione macOS chiede accesso al microfono. Se viene
negato, abilitalo in Impostazioni di Sistema → Privacy e Sicurezza → Microfono.
La build incorpora la descrizione del permesso nel proprio Info.plist.

La trascrizione viene mostrata prima della generazione e segue le normali regole
degli strumenti dell'agente. Usa `--read-only` o `--no-tools` quando desideri
quelle modalità. Un testo riconosciuto come `/quit` resta un messaggio utente:
i comandi della chat si eseguono soltanto da tastiera.

Anche un prompt singolo può ricevere una risposta vocale:

```sh
./qw3-agent-voice \
  -m ../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --ctx 8192 --kv-f16 --nothink --no-tools \
  --voice-config config/voice.local.json \
  -p 'Presentati brevemente in italiano.'
```

## Configurazione e memoria

`config/voice.example.json` descrive il client: URL del servizio, ID dei modelli,
lingue, voce, timeout delle richieste e durata massima della registrazione.
`config/audio-cpp.example.json` descrive il servizio e i percorsi dei pesi.
I due ID del client devono corrispondere agli `id` del servizio.

### Risposte pensate per l'ascolto

Con `--voice-config`, il prompt dell'agente richiede risposte in lingua naturale
che si capiscano senza guardare lo schermo. La regola è generale: presentare il
risultato utile per primo, raggruppare i dati, nominare gli elementi confrontati
ed esplicitare valori e unità, date, orari e percentuali. Tabelle, titoli grafici,
emoji e lunghi elenchi vengono evitati a meno che l'utente li chieda espressamente.
L'agente deve mantenere i valori esatti, l'incertezza e la completezza richiesta.
Le chiamate agli strumenti conservano la propria sintassi.

Per esempio, un confronto può diventare: «Il piano Base costa venti euro al mese
e offre cento megabit al secondo. Plus costa trenta euro e arriva a trecento
megabit al secondo.» Questa presentazione viene generata dal modello, senza
un secondo passaggio LLM dedicato alla riscrittura vocale.

Il filtro prima del TTS è una protezione aggiuntiva: per le tabelle Markdown
esplicite costruisce frasi con le intestazioni e i valori di ogni riga, senza
interpretare il dominio, cambiare quantità o eliminare righe. Aggiunge pause
ai titoli e agli elenchi e rimuove la decorazione dei titoli. La tabella scritta
resta disponibile. Questo ripiego conserva i dati ma può risultare più meccanico
di una risposta generata direttamente per l'ascolto.

I valori tra backtick vengono conservati: per esempio, `9,99 €`, `-1,5%`,
`21,5 °C` e `item_42` mantengono il proprio contenuto, mentre i blocchi di
codice restano esclusi. Le fonti con collegamenti vengono lette per nome;
gli URL HTTP(S) visibili vengono ridotti al dominio, senza leggere percorsi
e parametri. Sono gestiti anche i collegamenti Markdown per riferimento.
Questa pulizia è locale e non visita i siti. Il testo scritto conserva
collegamenti e valori completi.

Riavvia l'agente per caricare il nuovo profilo. Se riprendi una conversazione
salvata con il vecchio prompt, usa `/new` per applicarlo a una nuova conversazione;
il filtro di presentazione viene comunque applicato alle nuove risposte anche
quando la conversazione conserva il vecchio prompt.
I test del 35B verificano prosa e conservazione dei fatti su offerte e misure,
oltre ai casi del filtro su tabelle e liste di argomenti diversi.

### Sintesi progressiva della risposta validata

Una volta completato e accettato il turno dell'agente, la lettura usa segmenti
di testo: il primo viene riprodotto appena pronto, mentre si sintetizza il
successivo. Viene preparato al massimo un segmento in anticipo e si usano due
WAV temporanei a rotazione. Ctrl-C o un errore fermano il flusso, annullano e
attendono il lavoro pendente ed eliminano i file; non vengono saltati segmenti
per continuare con quelli successivi.
L'annullamento chiude le richieste del client; non garantisce che il servizio
interrompa immediatamente un calcolo audio già avviato.

`speech_chunk_chars` nel client controlla la dimensione indicativa dei segmenti:
il default è `320`, con valori interi fra `80` e `1000`. Si preferisce la fine
di una frase; per frasi lunghe si divide fra parole. Numeri decimali,
identificativi e caratteri Unicode non vengono tagliati all'interno di una
parola. Una singola parola più lunga del limite rimane intera. Il dizionario
viene applicato prima della suddivisione e rimane coerente per tutta la risposta;
le modifiche salvate durante la lettura valgono dalla risposta successiva.

Questa pipeline sovrappone **TTS e riproduzione**. La generazione di QW3 termina
prima dell'audio, conservando la validazione della risposta e il rilascio delle
viste Metal richiesto dalla configurazione da 24 GB. La sintesi durante la
generazione e l'interruzione automatica riprendendo a parlare restano nella
roadmap, insieme al collegamento diretto delle librerie audio.cpp.

### Velocità della voce

Puoi aggiungere `"tts_speed": 0.85` al file client `config/voice.local.json`
per rallentare la voce all'avvio. I valori accettati vanno da `0.5` a `2`;
`1` è la velocità normale di Kokoro. Se ometti la proprietà, viene mantenuta
la velocità configurata sul servizio audio.cpp. Il file di esempio usa `1`;
le configurazioni locali già presenti non vengono sovrascritte.

Durante la chat, `/speed 0.85` applica la modifica senza riavviare né agente
né servizio. Vale per la sessione e non riscrive il file. `/speed default`
rimuove l'impostazione del client per usare quella del servizio, anche se
la configurazione client iniziale specificava una velocità. Un valore non
valido viene segnalato e lascia intatta la velocità precedente.
Il comando si digita da tastiera; una frase riconosciuta dal microfono
rimane un messaggio per il modello.

### Stop automatico sul silenzio

Il rilevatore locale misura il livello audio RMS in dBFS. Usa una soglia di
attivazione, una soglia di rilascio inferiore di 6 dB e una durata minima di
attività per ignorare picchi molto brevi. Piccole interruzioni fino a 100 ms
durante l'attivazione sono tollerate; dopo l'attivazione, le pause più brevi
di `silence_seconds` mantengono aperto il turno.

| Impostazione del client | Default | Effetto |
| --- | --- | --- |
| `auto_stop` | `true` | Arresto automatico; `false` ripristina lo stop manuale con Invio. |
| `silence_seconds` | `1.2` | Silenzio richiesto dopo aver parlato; aumenta il valore per pause più lunghe. |
| `speech_threshold_db` | `-40` | Soglia RMS: un valore più negativo rileva voci più basse ma anche più rumore. |
| `min_speech_seconds` | `0.18` | Attività minima prima di considerare iniziato un turno. |
| `speech_start_timeout_seconds` | `8` | Attesa massima dell'inizio del parlato. |

`max_record_seconds` rimane un limite assoluto, anche nello stop manuale. Con
`auto_stop: true`, un turno senza attività sufficiente viene scartato anche se
premi Invio; usa la modalità manuale o adatta la soglia per registrazioni molto
deboli o brevi. Le impostazioni sono caricate all'avvio dell'agente.

Questa è una rilevazione energetica, senza un ulteriore modello neurale. Un
rumore abbastanza forte o continuo può essere rilevato come attività; soglia
e pause vanno provate sul microfono e nell'ambiente reali. La modalità `/listen`
si usa a turni e non supporta ancora l'interruzione automatica riprendendo a parlare.

Puoi controllare la decisione su un file senza aprire il microfono o contattare
audio.cpp:

```sh
./qw3-voice --config config/voice.local.json endpoint /tmp/input.wav
```

Il rapporto JSON riporta il motivo dello stop, il tempo, il massimo RMS dei
frame analizzati e l'attività rilevata (`speech_detected` è una decisione di
livello, non una classificazione del contenuto). I file usano frame da 20 ms;
il microfono usa il metering di AVAudioRecorder, il cui livellamento può dare
tempi leggermente diversi.

### Dizionario di pronuncia

`config/pronunciation.it.json` raccoglie le correzioni di accento verificate
all'ascolto. Inizialmente contiene un dizionario vuoto:

```json
{
  "version": 1,
  "replacements": {}
}
```

Il client lo abilita con `"pronunciation_dictionary": "pronunciation.it.json"`
in `config/voice.local.json`. Il percorso relativo viene risolto rispetto alla
cartella di quel file di configurazione, anche se l'agente cambia directory.
Omettendo l'opzione o impostandola a `""`, le sostituzioni sono disabilitate.

Una futura correzione verificata può avere questo formato:

```json
{
  "version": 1,
  "replacements": {
    "principi generali": "princìpi generali"
  }
}
```

Le chiavi sono parole o frasi letterali, senza espressioni regolari. Si cercano
parole intere, ignorando la distinzione tra maiuscole e minuscole; l'iniziale
maiuscola e le parole tutte maiuscole vengono mantenute. Le frasi più lunghe
hanno precedenza quando iniziano nello stesso punto. Ogni sostituzione avviene
una volta sul testo originale, senza applicare altre regole al suo risultato.
Per parole ambigue come `ancora` o `principi`, preferisci una frase che ne
identifichi il significato.

Il filtro agisce dopo la pulizia del testo, soltanto sulla copia inviata al TTS.
La risposta mostrata, il transcript e i messaggi del modello mantengono il testo
originale. Il file viene riletto prima di ogni sintesi: dopo il primo riavvio
dell'agente con questa versione, basta salvare il dizionario per applicare le
nuove voci. Non serve riavviare audio.cpp. Un dizionario non valido viene
segnalato e impedisce quella sintesi; la risposta scritta resta disponibile.

Per vedere il testo che verrà letto, senza avviare il servizio o generare audio:

```sh
./qw3-voice --config config/voice.local.json pronounce 'I principi generali.'
```

### Condivisione della memoria

L'URL predefinito è `http://127.0.0.1:8080`: con questi valori i dati audio
rimangono sulla macchina. Cambiando `server_url`, le registrazioni e il testo da
sintetizzare vengono inviati al server scelto. La registrazione è WAV PCM16 mono
a 16 kHz; il client la carica tramite multipart, senza richiedere un filesystem
condiviso con audio.cpp.

Sul Mac M5 da 24 GB, mantenere contemporaneamente i pesi Metal del 35B e avviare
CoreAudio ha causato un errore di riproduzione. Per questo la modalità vocale
alterna l'uso della memoria:

1. Scarica dalla GPU **solo i due modelli audio configurati** prima dell'LLM,
   usando `/v1/tasks/unload_models`; audio.cpp li ricarica alla richiesta seguente.
2. Riattiva le viste Metal dei pesi QW3 e genera la risposta.
3. Sincronizza la GPU e rilascia le viste dei pesi QW3 prima dell'audio.
   La mappatura CPU, la conversazione, la cache KV e DeltaNet restano presenti.
4. Attende, al massimo 5 secondi e solo quando la memoria wired supera il 75%
   della RAM, che il driver renda effettivo il rilascio prima di avviare l'audio.

Il servizio usa `lazy_load: true` e `max_loaded_models: 1`. Queste scelte
privilegiano la compatibilità con 24 GB e comportano ricaricamenti tra i turni.
Non sono ancora un'ottimizzazione della latenza. Inizia con contesti moderati.

## Diagnostica e verifiche

Le utility non caricano il modello linguistico:

```sh
./qw3-voice --config config/voice.local.json check
./qw3-voice --config config/voice.local.json record /tmp/input.wav
./qw3-voice --config config/voice.local.json transcribe /tmp/input.wav
./qw3-voice --config config/voice.local.json speak 'Ciao, sono QW3.' /tmp/reply.wav
./qw3-voice --config config/voice.local.json plan 'Prima frase. Seconda frase.'
./qw3-voice --config config/voice.local.json say 'Prima frase. Seconda frase.'
./qw3-voice play /tmp/reply.wav
```

`speak` scrive il WAV senza riprodurlo. Per file audio esterni, prepara WAV mono a
16 kHz prima di `transcribe`, per esempio con `ffmpeg -i input.wav -ar 16000 -ac 1
output.wav`. Il ciclo dell'agente elimina normalmente i propri WAV temporanei;
i file indicati esplicitamente alle utility restano disponibili.

`plan` mostra l'array JSON dei segmenti senza contattare audio.cpp o aprire
dispositivi audio. `say` sintetizza e riproduce con la stessa pipeline dell'agente,
poi riporta le misure JSON: segmenti previsti/riprodotti, tempo al primo segmento
pronto, tempi cumulativi di sintesi e riproduzione e tempo totale. Il primo
segmento pronto non misura l'inizio effettivo dell'audio sul dispositivo. I tempi
di sintesi e riproduzione si sovrappongono e non vanno sommati per stimare il
totale. La memoria wired è globale al sistema, campionata ai confini dei segmenti:
non rappresenta il picco continuo né la memoria dei soli processi audio/LLM.

Nell'agente, `--trace /percorso/trace.jsonl` aggiunge l'evento `voice_audio`
con le stesse misure e `generation_and_handoff_ms`, misurato dall'avvio del
turno LLM al passaggio all'audio. Registrazione e trascrizione precedenti non
sono incluse in quel tempo.

```sh
# Protocollo HTTP simulato, filtri, errori, timeout, annullamento; niente microfono.
# Include le fixture WAV per silenzio iniziale, pause, picchi brevi e isteresi.
make test-voice

# Include due turni reali del 35B con dispositivi simulati e ritorno alla
# tastiera dopo SIGINT; il test singolo riproduce solo un WAV silenzioso.
QW3_VOICE_TEST_MODEL=../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf make test-voice

# Logits identici dopo tre cicli di rilascio/ripristino dei pesi sul 35B reale.
make test-voice-residency

# Sintesi italiana reale -> conversione WAV con ffmpeg -> riconoscimento reale.
# Avvia un servizio temporaneo su una porta libera e lo chiude a fine test.
make test-voice-local

# Aggiunge la riproduzione progressiva reale di due segmenti Kokoro, senza microfono.
QW3_VOICE_TEST_PLAYBACK=1 make test-voice-local

# Facoltativo: include un turno del 35B e riproduce la risposta.
QW3_VOICE_TEST_MODEL=../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  make test-voice-local
```

Se il servizio non è disponibile all'avvio, l'agente vocale termina prima di
caricare il 35B. Un errore di sintesi o riproduzione dopo un turno viene mostrato
e lascia disponibile la risposta testuale. Una trascrizione vuota non avvia un
turno. Gli errori di configurazione, gli HTTP non riusciti e le risposte diverse
da WAV/JSON attesi vengono segnalati.

## Limiti della prima versione

Verificato il 29 ottobre 2026 sul MacBook Air M5 da 24 GB: protocollo simulato,
filtri ed errori; continuità numerica esatta dopo rilascio/ripristino dei pesi;
sintesi italiana reale ritrascritta da Qwen3-ASR; risposta del 35B sintetizzata
da Kokoro e riprodotta tramite AVFoundation. I test non attivano il microfono.

L'estensione del 30 ottobre aggiunge lo stop automatico sul silenzio e `/listen`.
Le prove su file coprono silenzio iniziale, picchi brevi, isteresi, pause,
durata massima e modalità manuale; il controller viene provato con due turni
del 35B, dispositivi simulati e SIGINT al turno successivo. La taratura e
l'acquisizione reali del microfono rimangono da verificare nell'ambiente utente.
La stessa estensione conserva i dati inline, semplifica le fonti nell'audio e
aggiunge la velocità per richiesta e `/speed`. I test controllano i dati inviati
al TTS, gli aggiornamenti nella stessa sessione e i valori non validi; la prova
locale confronta anche la durata di due WAV Kokoro a velocità diverse.

La pipeline progressiva aggiunge prove dell'ordine e della conservazione del
testo, della sovrapposizione fra TTS e riproduzione, della coda limitata, della
coerenza del dizionario e della pulizia su errore e annullamento. L'opzione
`QW3_VOICE_TEST_PLAYBACK` verifica la riproduzione progressiva nativa reale.

Registrazione, trascrizione e generazione rimangono in sequenza; dopo il rilascio
dei pesi QW3, sintesi e riproduzione possono sovrapporsi per segmenti consecutivi.
Il TTS riceve solo una risposta finale accettata dopo gli strumenti e le
verifiche; vengono rimossi ragionamento, blocchi di codice e parte della
formattazione Markdown. Una risposta composta solo da codice non viene letta.

Non sono implementati VAD neurale, interruzione automatica quando
riparli, cancellazione dell'eco o sintesi durante la generazione. Il microfono
va ancora verificato manualmente con il dispositivo e il permesso macOS
dell'utente. La build normale `make` mantiene la consueta interfaccia testuale;
le dipendenze audio sono collegate soltanto ai target opzionali.

Riferimenti upstream: [API del servizio](https://github.com/0xShug0/audio.cpp/blob/v0.9.3/app/server/api.md),
[Qwen3-ASR](https://github.com/0xShug0/audio.cpp/blob/v0.9.3/docs/models/qwen3_asr.md),
[Kokoro](https://github.com/0xShug0/audio.cpp/blob/v0.9.3/docs/models/kokoro_tts.md).
