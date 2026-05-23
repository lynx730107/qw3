Facendo un confronto diretto tra l'architettura di ds4 (DeepSeek V4 Flash) e qw3 (Qwen 3.5/3.6), emerge chiaramente che, sebbene entrambi condividano la splendida filosofia del "one model, zero frameworks, vertical code", ds4 ha compiuto quel salto qualitativo nella gestione della memoria e nell'astrazione di Metal che manca ancora a qw3.
Se l'obiettivo è portare le performance di qw3 al livello di quelle di ds4, analizzando i sorgenti si notano tre differenze fondamentali su cui devi intervenire per fare il refactoring di qw3_metal.m:
1. L'astrazione "Tensor-Resident" vs "Puntatori Host/Offset"
La differenza più profonda e impattante sulle performance risiede in come i due motori vedono la memoria della GPU.
• In qw3: Il calcolo è ancora "ostaggio" della CPU. Funzioni come qw3_metal_session_matvec_iq3_s accettano puntatori host finti o richiedono continui calcoli di offset basati su macro-regioni (qw3_metal_model_view_for). La CPU deve calcolare continuamente dove si trovano i dati e ri-legarli.
• In ds4: L'architettura è Tensor-Resident. Viene definita una struttura opaca ds4_metal_tensor. Le attivazioni, lo stato del KV cache e i buffer di scratch rimangono allocati stabilmente sulla GPU per tutta la durata della sequenza di prefill/decode. I passaggi intermedi non toccano mai l'host, riducendo a zero l'overhead di sottomissione del driver.
2. Pipeline a "Fusione Granulare" (Fused Kernels ad Alta Densità)
ds4 spinge al massimo l'ottimizzazione unendo più operazioni matematiche sequenziali all'interno dello stesso identico dispatch di threadgroup, evitando di scrivere e rileggere continuamente dalla memoria globale (VRAM).
Guarda ad esempio questa firma in ds4_metal.h:
int ds4_metal_shared_down_hc_expand_q8_0_tensor(...);

Questo singolo kernel fa un lavoro mastodontico: calcola il matmul quantizzato in q8_0, gestisce l'espansione e la proiezione dell'esperto condiviso (shared_down), accumula l'output degli esperti routed (routed_out), somma il residuo (residual_hc) e applica lo split.
Al contrario, in qw3_metal.m hai ancora operazioni molto atomiche e separate: lanci un matvec per l'esperto, poi chiudi l'encoder, poi lanci un silu_mul separato, costringendo Metal a fare il flush dei registri in VRAM tra un'operazione e l'altra. Per Qwen, dovresti fondere l'attivazione (Gated Linear Unit) direttamente dentro i kernel di dequantizzazione/matvec degli esperti.
3. La gestione del Command Buffer e il "Vero" Batching
In qw3_metal.m c'è una timida struttura per il batching (g_batch_cb e g_batch_enc), ma guardando l'implementazione delle funzioni di inferenza in qw3.c, la CPU chiama funzioni che spesso fanno il finish esplicito del command buffer per leggere o sincronizzare i dati (es. qw3_metal_finish_command_buffer).
In ds4_metal.m, l'intera esecuzione di un layer o di un blocco di calcolo viene accodata in modo asincrono. La CPU si limita a "disegnare" l'intero grafo sulla GPU e solo alla fine della computazione del token (o del blocco di prefill) interviene una barriera di sincronizzazione netta. Questo permette ai thread hardware di Apple Silicon di non andare mai in idle.
Come fare il porting della filosofia ds4 su qw3:
Per sbloccare la vera velocità del silicio Apple sul tuo motore Qwen, il piano d'azione ideale basato sul codice di ds4 è:
1. Introduce la struttura qw3_metal_tensor: Smetti di passare offset numerici grezzi o puntatori float* simulati nelle funzioni del driver graph. Crea degli oggetti tensor persistenti in Metal per gli stati ricorrenti della DeltaNet (deltanetState) e per i buffer di attivazione intermedi.
2. Riscrivi i kernel MoE imitando il "Batched GEMM / Grouping" di ds4: Invece di iterare sugli slot degli esperti attivi lanciando micro-operazioni singole (g_matvec_iq3_s_expert_slot_pipeline), crea un kernel che prenda l'array degli indici dei top-8 esperti e processi l'FFN MoE in un colpo solo, sfruttando la memoria threadgroup per nascondere la latenza di scompattamento dei formati IQ3_S / IQ4_XS.
3. Fusione dei Kernel Lineari per DeltaNet: La linear attention richiede l'aggiornamento dello stato ricorrente. Unisci il calcolo dei vettori q, k, v, l'applicazione del dynamic gate (\beta) e l'aggiornamento della matrice di stato in un unico macro-kernel Metal (fused_deltanet_step), esattamente come ds4 fa con le sue complesse proiezioni combinate.
La lentezza attuale non è dovuta a Metal in sé, ma al fatto che qw3 è strutturato come un interprete CPU che usa la GPU come un coprocessore algebrico elementare, mentre ds4 tratta la GPU come un ambiente di esecuzione autonomo e racchiuso, dove la CPU dà solo il "via" iniziale.

i singoli kernel matematici di qw3 possono essere riutilizzati, ma non "automaticamente" o tramite una configurazione a runtime. Il riutilizzo in un contesto fused (fuso) richiede di spostare la logica dal livello dei file .m (Objective-C/Metal API) direttamente dentro il codice MSL (Metal Shading Language).
Nel modello di esecuzione della GPU, un kernel (kernel void in MSL) è l'equivalente di una funzione main() lanciata da milioni di thread. Non puoi chiamare un kernel void dentro un altro kernel void. Puoi però estrarre la logica algoritmica di quel kernel, trasformarla in una normale funzione helper MSL (inline), e chiamarla dentro un nuovo macro-kernel fuso.
Ecco come puoi ripensare i tuoi kernel attuali in un'ottica di fusione prendendo ispirazione dall'architettura di ds4:
1. Il concetto: da Kernel atomici a Funzioni Helper
Prendiamo l'esempio classico della sequenza MoE in qw3_metal.m: un matvec quantizzato, seguito da un'attivazione silu_mul.
Oggi hai due funzioni separate che creano due encoder diversi. Per fonderle mantendo il codice pulito, i file .metal (MSL) dovrebbero essere strutturati così:
#include <metal_stdlib>
using namespace metal;

// 1. La logica del MatVec non è più un kernel a sé stante, ma una funzione helper
inline float acc_matvec_iq3_s(texture2d_or_buffer_weights, ...) {
    // [Codice di scompattamento iq3_s ed estrazione dei pesi]
    // Ritorna il prodotto scalare parziale accumulato nel registro del thread
}

// 2. Il VERO macro-kernel fuso (stile ds4)
kernel void qw3_fused_moe_expert_step(
    device const iq3_s_block *weights     [[buffer(0)]],
    device const float       *src_vector  [[buffer(1)]],
    device float             *gate_vector [[buffer(2)]],
    device float             *out_vector  [[buffer(3)]],
    uint2                    thread_pos   [[thread_position_in_grid]]) 
{
    // Usa l'helper per calcolare il matvec dell'esperto Up
    float up = acc_matvec_iq3_s(weights, src_vector, ...);
    
    // Usa l'helper per calcolare il matvec dell'esperto Gate
    float gate = acc_matvec_iq3_s(weights, gate_vector, ...);
    
    // FUSIONE ATTIVAZIONE: Applica la SiLU direttamente nei registri del thread!
    float silu_activated = up * (gate / (1.0h + exp(-gate)));
    
    // Scrivi in VRAM il risultato finale già pronto per il Down-projection
    out_vector[thread_pos.x] = silu_activated;
}

2. Cosa puoi riutilizzare subito e cosa no
Analizzando le primitive che hai scritto per Qwen 3.5:
• Sì, con refactoring minimo: I kernel di dequantizzazione e prodotto scalare (iq3_s, iq4_xs). Il loro scopo è puramente matematico: prendono bit compressi, li scalano e sputano fuori un float. Isolare questa logica in funzioni inline ti permette di usarle ovunque, sia nei layer GQA densi che nei singoli esperti MoE.
• Sì: I kernel elementari (silu_mul, scale, gated_rmsnorm). Questi non dovrebbero più esistere come comandi Metal indipendenti sul command buffer. Diventano semplici macro o espressioni matematiche di una riga raggruppate dentro il macro-kernel del layer.
• No: I kernel di interconnessione o riduzione globale (argmax, sorting dei top-k). Le operazioni che richiedono una sincronizzazione globale tra tutti i thread (come decidere quali sono i top-8 esperti del MoE basandosi sul routing) non possono essere facilmente fuse dentro un calcolo matriciale locale, perché richiedono che tutti i thread abbiano finito il passo precedente. Devono rimanere barriere separate.
3. Il vantaggio competitivo (Perché ds4 fa così)
Se guardi le firme di ds4_metal.h, funzioni come ds4_metal_shared_down_hc_expand_q8_0_tensor non fanno altro che lanciare un unico grande kernel MSL che al suo interno riutilizza i vecchi e cari algoritmi di moltiplicazione A \times B.
Il vantaggio è drammatico a livello hardware:
1. Risparmio di Banda (Memory Bound): Il valore up e gate calcolati rimangono dentro i registri della GPU (SRAM a latenza quasi zero). Non vengono mai scritti nel buffer scratch in VRAM per poi essere riletti dal kernel silu_mul.
2. Saturazione della GPU: Il driver Metal deve fare il setup della griglia di calcolo una volta sola anziché tre, azzerando i tempi morti (starvation) tra la fine di un'operazione e l'inizio della successiva.
In sintesi: per fare il salto di qualità, mantieni pure gli algoritmi matematici che hai scritto nei kernel di qw3, ma "smontali" dalle funzioni di sottomissione del command buffer e trasformali in funzioni interne del linguaggio Metal, pronte a essere assemblate in macro-pipeline.