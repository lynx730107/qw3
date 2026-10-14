# Concorrenza delle proiezioni DeltaNet e FFN

Il gate numerico sul 27B passa, con kernel, buffer dati e numero di encoder
originali. Lo screening dà un punto +2,757%, con controlli che variano
del 5,936%: risultato inconcludente, nessuna adozione e nessun nuovo
confronto con llama. L'obiettivo richiede ancora il +2,383% di
decoding documentato nel [riferimento](m5_q4_uint_reference_2026-10-13.md).

La precedente [eccezione per la sola coppia FFN](m5_decode_ffn_barriers_2026-10-14.md)
dava un segnale non uniforme e aggregato −0,743%, ed è scartata. Questa
prova estende il gruppo indipendente alle quattro proiezioni DeltaNet:
qkv, gate, alpha e beta. Leggono tutte x1 e scrivono regioni disgiunte di
scratch, rispettivamente a offset 0, n_qkv, n_qkv+inner_n e
n_qkv+inner_n+n_v_heads. Le dimensioni delle ultime due sono n_v_heads.
Convoluzione e ricorrenza rimangono dopo una barriera completa.

Resta un solo encoder concorrente nel token. Il callback aggiunge una
barriera prima di ciascun dispatch, eccetto quelli successivi al primo
nelle finestre indipendenti di due o quattro matvec. Ogni finestra deve
contenere esattamente il numero dichiarato di dispatch; ogni violazione
fa fallire il gate. Le barriere già presenti nel backend restano intatte.
Il generatore usa snapshot verificati del gate precedente; il verificatore
ricostruisce byte per byte qw3.c e qw3_metal.m originali rimuovendo i soli
hook, cleanup e inoltri dei dispatch. Non si cambia il codice Metal.

Non ci sono nuove copie dei pesi, buffer dati di sessione, pipeline o
encoder aggiuntivi. Entrambi i modi attraversano gli stessi callback CPU,
con inserimento delle barriere disattivato nel controllo seriale. La RAM
totale e di picco non sono misurate; non si deduce un beneficio dal solo
numero di barriere eliminate.

## Gate numerico

Build modello e timing senza warning, Metal API Validation attiva. Una
sola sessione ctx256 riutilizzata, oracolo temporaneo su disco, prefisso128
seed1 e seguito greedy32, KV F16 e 64 layer/head GPU. Coincidono 66
vocabolari completi finiti, token, history, posizione e stato canonico
prima/dopo: KV GQA vivo, DeltaNet e convoluzione, escluso il tail
temporaneo Flash Attention. Restano da eseguire otto prefissi, checkpoint
e regressioni dell'agente prima di un'eventuale adozione confermata dai tempi.

Il seguito del candidato conta 992 finestre FFN e 1536 DeltaNet, 2528
totali: per token 31/48/79. Su 31296 dispatch aggiunge 25696 barriere e
ne salta 5600, cioè per token 978/803/175. Baseline zero interventi;
cleanup e contatori verificati. Sorgenti, snapshot, binario, stat del
GGUF, binari di produzione e store personale restano invariati nel gate.

[Manifest](measurements/decode_projection_barriers_2026-10-14/manifest.json)
e [verifica](measurements/decode_projection_barriers_2026-10-14/verification.json)
conservano sorgenti, comandi e dati grezzi. Non si archiviano binari,
modelli o checkpoint. Lo screening usa OFF/ON/OFF/ON/OFF, tg32 depth128
ctx256, tre ripetizioni, warmup e 60 secondi prima di ogni processo,
API Validation disattiva. Si conservano tutti i campioni.

## Screening instabile

Medie dei cinque blocchi in ms: 3574,757 /3607,864 /3786,970 /3557,954
/3578,116. Primo ON +2,023% sugli OFF interpolati, −0,918%/+4,964% sui
singoli vicini. Secondo ON +3,502%, +6,437%/+0,567% sui singoli vicini.
L'aggregato +2,757% non costituisce un guadagno confermato: lo spread
max/min OFF 5,936% supera il limite prestabilito del 5%. In particolare
il primo ON perde sul primo OFF, mentre l'OFF centrale più lento pesa
su entrambi gli effetti interpolati.

Restano tutti i 15 campioni; nessuno è scartato. Le tre ripetizioni
dell'OFF centrale sono 3800,650 /3803,152 /3757,108 ms. Non si attribuisce
la causa a carichi o temperature non misurati. `pmset -g therm` eseguito
dopo la serie non riporta avvisi registrati; non misura frequenza o
temperatura GPU e non dimostra stabilità termica durante la serie.

Ogni ON conta 7663 finestre: 3007 FFN e 4656 DeltaNet, 94866 dispatch,
77891 barriere aggiunte e 16975 salti indipendenti. OFF zero interventi;
CSV 176.761.680 byte di sessione. Swap 941,31 MiB prima/dopo ciascun
processo; sorgenti, snapshot, binari e stat del GGUF invariati.
[Analisi](measurements/decode_projection_barriers_2026-10-14/native/analysis.json)
e [verifica](measurements/decode_projection_barriers_2026-10-14/native/verification.json)
conservano tutti i dati e il ricalcolo.

Il seguito verifica il segnale con ordine AB/BA nello stesso processo e
nella stessa sessione, usando i reset, prefill e timer originali. Questo
diagnostico riduce la distanza temporale tra le coppie, ma non sostituisce
una replica indipendente e il confronto finale con llama.
