# Generazione M5 con due proiezioni in un dispatch Metal

Il diagnostico combina proiezioni indipendenti con lo stesso input e destinazioni disgiunte su Qwen3.8-27B-UD-Q3_K_XL. I kernel mantengono i dot product e le riduzioni IQ3_S e IQ4_XS native. Il passaggio a un dispatch per coppia riguarda **12 coppie dello stesso formato** e **25 coppie di formati diversi** per token. Il totale comprende 30 coppie QKV/gate DeltaNet e 7 coppie gate/up FFN. Si riusano pesi, buffer di sessione e scratch.

La prima serie usa processi distinti, 32 token generati a profondità 128, ctx256, KV F16, 64 layer Metal, quattro thread e seed 1. Tre campioni dopo il warmup per processo, con 6 secondi prima di ciascun processo. I confronti vicini prendono la media dei due controlli che circondano il candidato.

| Variante | Differenza nelle due finestre vicine |
|---|---:|
| Stesso formato | +1,23%, −17,14% |
| Formati diversi | −1,14%, +7,46% |
| Entrambe | −2,51%, −2,60% |

La dispersione delle medie dei controlli è 74,52%, calcolata come massimo/minimo meno uno. Le medie aggregate di questa serie sono quindi inadeguate per stabilire un piccolo guadagno. Tutti i 39 campioni sono conservati. [Processi distinti](screen.json) · [Analisi](screen-analysis.json)

Con un solo processo e una sessione riusata, il percorso dello stesso formato misura **4,771 contro 4,769 tok/s** per il controllo e il candidato, pari a **−0,05%**. Le quattro coppie adiacenti, in ordine bilanciato, misurano +3,91%, +0,28%, −6,95% e +3,16%. Le pipeline sono riscaldate prima dei campioni e il prefill è fuori dal timer. [Coppie dello stesso formato](paired.json)

Nel test con la sola Metal API Validation, il candidato completo ha riprodotto tutti i 33 vettori di logits, lo stato canonico e il checkpoint bit per bit. Le quattro combinazioni di proiezioni isolate sono esatte eseguendo il kernel senza Shader Validation. La strumentazione Shader Validation ha dato un esito diverso. Il test completo fallisce al primo token di continuazione e nel test isolato compaiono righe a zero anche nell'output del kernel originale IQ3_S. La causa non è stata isolata. [Confronto completo](quality-api.log) · [Proiezioni isolate](matrix.log) · [Con Shader Validation](matrix-validated-v1.log) · [Errore di continuazione](quality.log)

Con la stessa procedura, la variante per i formati diversi misura **4,707 contro 4,670 tok/s**, pari a **−0,78%**. Le quattro coppie sono +1,89%, +0,95%, −1,18% e −4,88%. I livelli assoluti delle due serie sono specifici di questa sessione di misura e i confronti usano il controllo della rispettiva serie. [Coppie di formati diversi](paired-mixed.json)

Il diagnostico conserva **55 campioni** tra le tre serie. La riduzione nel numero dei dispatch non ha dato un guadagno ripetibile di generazione, quindi nessuna variante viene adottata. [Sorgenti diagnostici](prepare.py) · [Hash della checkout](checkout_sha256.json)
