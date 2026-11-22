# Piano dell'integrazione vocale

Questo documento riprende il piano iniziale: prima un ciclo premi-per-parlare,
poi integrazione audio nativa, silenzio, sintesi progressiva e interruzione
quando l'utente riprende a parlare. Gli interventi sulla presentazione dei
dati e sul dizionario sono complementari a questo percorso.

## 1. Ciclo iniziale — implementato

Registrazione → Qwen3-ASR 0.6B → normale turno dell'agente → Kokoro 82M →
riproduzione. Il TTS riceve il testo finale accettato dopo gli strumenti e
le verifiche; ragionamento e blocchi di codice vengono esclusi. Il testo
della conversazione rimane disponibile anche se l'audio fallisce.

## 2. Acquisizione e riproduzione native — implementate; motore separato

Il modulo C/Objective-C usa AVFoundation per microfono e altoparlanti e
NSURLSession per l'API locale. Non richiede Python o ffmpeg nel ciclo audio.
audio.cpp è ancora un processo separato: il collegamento diretto delle sue
librerie C++ nell'eseguibile resta da implementare e verificare, compresi
il ciclo di vita dei modelli e l'interazione con i backend Metal dei due runtime.

## 3. Stop sul silenzio e turni consecutivi — implementati

Rilevatore energetico con isteresi, silenzio finale, timeout iniziale e limite
assoluto; `/listen` riapre il microfono dopo la riproduzione. Le prove automatiche
non aprono il microfono. La taratura sul dispositivo e nell'ambiente dell'utente
resta una verifica manuale. Non si tratta di un VAD neurale.

## 4. Sintesi progressiva e memoria — prima parte implementata

La risposta finale validata viene suddivisa preferendo i confini di frase.
Il primo segmento viene riprodotto appena sintetizzato; durante la riproduzione
si prepara il successivo. La coda mantiene una sola sintesi in anticipo e usa
due file WAV temporanei a rotazione. Annullamento ed errori fermano il flusso,
attendono la richiesta pendente ed eliminano i file. Il dizionario viene letto
una volta per risposta, prima della suddivisione.

Sono disponibili il piano dei segmenti senza audio, le misure della pipeline
e l'evento JSONL `voice_audio` nell'agente. I campioni di memoria wired sono
globali al sistema e non rappresentano né il picco continuo né la memoria
attribuibile ai soli processi QW3/audio.cpp.

**Resta da implementare la sintesi mentre QW3 genera ancora.** Sul Mac da 24 GB
la convivenza dei pesi Metal del 35B con CoreAudio ha causato errori; per ora
si conserva il rilascio delle viste dei pesi prima dell'audio. Occorrono:

- una modalità di residenza/condivisione con margine di memoria misurato e
  nessuna regressione della sessione o della riproduzione;
- un punto esplicito in cui il testo sia impegnato come risposta all'utente,
  prima di inviarlo all'audio: non basta intercettare l'output dei token, che
  può ancora contenere strumenti o una risposta respinta dalle verifiche;
- prove con il 35B, strumenti attivi, cancellazione e risposte lunghe,
  misurando il tempo dal termine dell'input all'inizio effettivo dell'audio.

La pipeline attuale sovrappone TTS e riproduzione, non LLM e TTS.

## 5. Interruzione riprendendo a parlare — da implementare

Richiede acquisizione simultanea alla riproduzione, distinzione fra voce
dell'utente ed eco degli altoparlanti, cancellazione coordinata del TTS e
gestione del nuovo turno. Il rilevatore energetico attuale, usato da solo,
potrebbe interrompere l'agente sulla sua stessa voce. Prima si verifica il
percorso con cuffie; poi si integra e si misura la cancellazione dell'eco
per l'uso con altoparlanti. Ctrl-C è già disponibile per annullare manualmente.

## Verifiche di riferimento

`make test-voice` verifica filtri, confini dei segmenti, conservazione dei dati,
ordine, una sola sintesi in anticipo, sovrapposizione con la riproduzione,
istantanea del dizionario, errori e annullamento con dispositivi simulati.
Con `QW3_VOICE_TEST_MODEL`, include il 35B e il trace audio. Con
`QW3_VOICE_TEST_PLAYBACK=1 make test-voice-local`, verifica Kokoro e la
riproduzione progressiva AVFoundation reali, senza microfono.

La [misura del 30 ottobre](measurements/voice_pipeline_2026-10-30.json)
registra una prova funzionale reale di due segmenti: primo WAV pronto in
594 ms e riproduzione completa riuscita. Misura soltanto la pipeline audio,
con QW3 inattivo, e non costituisce un confronto statistico della latenza.
