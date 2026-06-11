L'ottimizzazione del prefill nel progetto qw3 (il motore custom ottimizzato per Apple Silicon per far girare modelli come Qwen3.6-35B-MoE) è una sfida complessa ed essenziale per ottenere performance vicine a quelle di motori più maturi come llama.cpp o DS4.
Ecco il riassunto completo di tutta la questione, lo stato attuale, le soluzioni tecniche implementate (come TILED4) e la risposta sulla fattibilità della KV cache quantizzata a 8-bit.
1. Il contesto e il problema originario
Inizialmente, la generazione di qw3 era legata a un path CPU reference. Il bring-up di Metal aveva validato la correttezza numerica, ma soffriva di un collo di bottiglia strutturale enorme: a ogni token generato veniva eseguito un re-prefill completo (un ricalcolo totale da zero del prompt). Di conseguenza, la velocità crollava drasticamente con l'aumentare dei token.
L'obiettivo primario è stato quindi eliminare il re-prefill e strutturare un prefill in batch nativo su GPU, muovendosi in direzione di llama.cpp (puntamento a oltre 600 tok/s sul prompt lungo).
2. Gli step dell'ottimizzazione (La Roadmap)
Fase A: Lo Scaffold della Sessione Persistente
Per evitare di ricalcolare il contesto a ogni step, è stata creata la struttura qw3_metal_session. Questa alloca in modo persistente sulla memoria della GPU (VRAM) tutti i buffer necessari affinché i dati non debbano continuamente fare la spola (readback) tra CPU e GPU.
I buffer persistenti allocati includono:
•	La GQA KV Cache.
•	Lo stato ricorrente di DeltaNet.
•	Lo stato di convoluzione (conv state).
•	I logits di output.
Attraverso dei test diagnostici progressivi (da te testati ed eseguiti con successo), sono stati collegati i kernel fondamentali direttamente a questi buffer residenti su GPU, stabilendo la catena:

‭$$\text{Token Embd} \rightarrow \text{session.x0} \rightarrow \text{RMSNorm} \rightarrow \text{session.x1} \rightarrow \text{MatVec QKV} \rightarrow \text{session.scratch (layout qkv + z)}$$‬‭‬‭‬‭‬‭‬‭‬
Fase B: L'introduzione di FlashAttention e il Padding
Per gestire i prompt reali (che spesso non hanno lunghezze multiple di 64), è stato introdotto come default il kernel di FlashAttention GQA accoppiato a un meccanismo di padding monodimensionale. Questo intervento ha sbloccato le prestazioni sui prompt non allineati, portando il prefill da ~200 tok/s a ~315 tok/s costanti.
Fase C: L'ottimizzazione del Mixture of Experts (MoE)
Il modello Qwen3.6-35B fa un uso intensivo di MoE. Dai profili è emerso che i blocchi di down-projection degli esperti pesavano per oltre la metà del tempo del prefill.
•	È stata implementata una tecnica di compattazione del dispatch (tramite calcolo indiretto su GPU), evitando di lanciare threadgroup vuoti per gli esperti non attivati dal router.
•	È stato introdotto il supporto ottimizzato TensorOps per i pesi quantizzati in Q6_K.
Fase D: L'algoritmo TILED4 (DeltaNet/GDN)
Mentre soluzioni come il cambio di precisione intermedia (MID_F16) o le modifiche alla tassellazione MoE non hanno portato benefici tangibili (risultando in rumore o lievi peggioramenti), la svolta sul ramo delle attention lineari (DeltaNet/GDN) è arrivata con l'opzione opt-in:
QW3_METAL_BATCH_GDN_TILED4=1.
•	Come funziona: Riduce drasticamente il numero di dispatch atomici frastagliati unendo i calcoli in tile parallelizzate.
•	Risultato: Il tempo di calcolo di deltanet_gdn è crollato da 40-50 ms/layer a 30-32 ms/layer. Grazie a TILED4, il throughput di prefill su un blocco pesante come pp4096 (4096 token di prompt) si è attestato sui 435.70 tok/s.