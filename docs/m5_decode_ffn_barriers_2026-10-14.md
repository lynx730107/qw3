# Unico encoder concorrente con dipendenze esplicite

La nuova prova privata mantiene un solo encoder per token e tutti gli
shader originali. Passa il gate numerico sul 27B: logits completi finiti,
token e stato canonico coincidono bit per bit. Lo screening dei tempi dà
un segnale non uniforme e aggregato −0,743%: la variante viene scartata.
Produzione e obiettivo restano invariati.

La [prima finestra FFN concorrente](m5_decode_ffn_concurrent_2026-10-14.md)
perde −0,437% nello screening e viene scartata. Aggiungeva confini di
encoder; la nuova prova elimina quei confini senza attribuire a essi
una causa già dimostrata. Apre l'encoder concorrente per l'intero token e
inserisce una barriera sui buffer prima di ogni dispatch, salvo il secondo
matvec della coppia FFN indipendente. Gate e up leggono x1 e scrivono
regioni disgiunte di scratch, a offset 0 e n_ff. Le coppie già fuse e il
prefill restano originali.

Il bridge privato sostituisce le sole chiamate Objective-C di dispatch
con callback che inoltrano dimensioni e argomenti originali, inserendo
le barriere. Il generatore e la verifica ricostruiscono byte per byte
qw3.c e qw3_metal.m originali rimuovendo hook, cleanup e callback.
Non si modificano shader, espressioni FP32, pipeline, pesi, layout o
buffer di dati della sessione. Non si aggiungono encoder nella modalità
normale di decoding. Le callback CPU e la gestione delle barriere hanno
comunque un costo possibile; la RAM totale e di picco non sono misurate.

## Controlli

Build modello e timing senza warning. Il primo harness compilato emetteva
warning di protocollo e indentazione dei return generati: viene corretto
prima di qualunque esecuzione GPU/modello o misura. Il log iniziale è
conservato; nessun campione proviene da quel binario. La selezione della
funzione è limitata alla definizione, senza modificare i return tra il
prototipo e la definizione; i cast di protocollo riguardano solo i dispatch.

Metal API Validation attiva; una sola sessione ctx256 riutilizzata con
oracolo temporaneo su disco, prefisso128 seed1 e seguito greedy32,
KV F16 e 64 layer/head GPU. Baseline e candidato coincidono su 66
vocabolari completi finiti, token, history, posizione e stato canonico
prima/dopo: KV GQA vivo, DeltaNet e convoluzione. Il tail temporaneo Flash
Attention è escluso. Non si eseguono ancora otto prefissi, checkpoint o
regressioni dell'agente: restano necessari prima di un'eventuale adozione.

Durante i 32 passi del candidato si contano 31296 dispatch, 30304
barriere aggiunte e 992 salti: per token 978/947/31. Ogni finestra deve contenere
esattamente due dispatch; i contatori e il cleanup sono verificati.
Le barriere già presenti nel backend restano invariate. Baseline con le
stesse callback di inoltro ma zero interventi di sincronizzazione. Sorgenti, snapshot, binario, stat del GGUF,
binari di produzione e store personale restano invariati nel gate.

[Manifest](measurements/decode_ffn_barriers_2026-10-14/manifest.json) e
[verifica indipendente](measurements/decode_ffn_barriers_2026-10-14/verification.json)
conservano sorgenti e log. Non si archiviano binari, GGUF o checkpoint.
Lo screening usa OFF/ON/OFF/ON/OFF, tg32 depth128 ctx256, tre ripetizioni,
warmup e 60 secondi prima di ogni processo. Tutti i campioni vengono
conservati; non c'è ancora un nuovo confronto con llama.

## Screening completo

Tutti i 15 campioni dei cinque processi sono conservati. Medie in ms:
3612,107 /3575,642 /3556,624 /3632,502 /3583,855. Il primo ON dà +0,244%
sugli OFF interpolati, ma +1,020%/−0,532% sui singoli vicini. Il secondo
dà −1,714%, e −2,089%/−1,339% sui singoli vicini. L'effetto aggregato dei
due ON è −0,743%; non c'è un intervallo paired indipendente perché i
controlli sono condivisi. Spread max/min OFF 1,560%, entro il limite del
5% ma superiore al primo segnale positivo. Non si attribuisce la
variabilità a cause esterne non misurate e non si eliminano campioni.

Ogni ON conta 3007 finestre, 94866 dispatch, 91859 barriere aggiunte e
3007 salti indipendenti, cioè 97 forward. Gli OFF contano zero interventi.
Le righe CSV mantengono 176.761.680 byte di sessione. Swap 941,31 MiB
prima/dopo ciascun processo; sorgenti, snapshot, binari e stat del GGUF
invariati. Il verificatore rilegge comandi, log, CSV, campioni e contatori
e ricalcola gli effetti, compreso l'aggregato.

[Analisi](measurements/decode_ffn_barriers_2026-10-14/native/analysis.json)
e [verifica](measurements/decode_ffn_barriers_2026-10-14/native/verification.json).
Nessuna adozione o suite estesa/checkpoint per questa variante scartata.
La ricerca prosegue sulle quattro proiezioni DeltaNet indipendenti che
leggono x1 e scrivono intervalli distinti di scratch; la serie presente
non misura ancora tale estensione.
