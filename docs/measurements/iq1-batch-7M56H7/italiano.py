"""Deterministic Italian smoke checks with locally verifiable answers."""
import os, sys, json, subprocess
cases=[
 ("conti","Hai 48 bulloni. Ne usi 15 al mattino e 9 nel pomeriggio. Ricevi poi altre 2 scatole da 6 bulloni. Quanti bulloni hai alla fine? Rispondi solo con un numero.", "36"),
 ("estrazione","Registro aggiornato: ordine A17 annullato, B42 in attesa, C09 spedito. Scrivi esclusivamente il codice dell’ordine che è ancora in attesa.", "B42"),
 ("dato_assente","Nel verbale si legge: Ada e Bruno hanno approvato il preventivo. Il verbale non riporta importi. Quanto vale il preventivo? Se il testo non lo dice rispondi esattamente NON INDICATO.", "NON INDICATO"),
 ("correzione","Inizialmente la riunione era prevista martedì alle 9. L’ultima email la sposta a giovedì alle 11, nella stessa sala. Restituisci solo il nuovo giorno e l’ora, nel formato giorno HH:MM.", "giovedì 11:00"),
]
model=sys.argv[2] if len(sys.argv)>2 else "../../models/Qwen3.8-27B-UD-IQ3_XXS.gguf"
modes=["1"] if len(sys.argv)>2 else ["0","1"]
for mode in modes:
    for name,prompt,expected in cases:
        proc=subprocess.run(["./qw3-cli","-m",model,"--ngl","64","--ctx","512","--kv-f16","--nothink","--temp","0","-n","40","-p",prompt],env={**os.environ,"QW3_METAL_IQ1_BATCH":mode},capture_output=True,text=True,check=True)
        answer=proc.stdout.strip()
        print(json.dumps(dict(mode=mode,case=name,expected=expected,answer=answer,ok=answer.strip().casefold().rstrip('.')==expected.casefold()),ensure_ascii=False),flush=True)

prompt = ("Scrivi una breve email a un cliente, in tono cortese. Il corriere aveva previsto "
          "il ritiro venerdì, ma lo ha spostato a lunedì. Non conosciamo ancora il giorno "
          "di consegna. Spiega cosa è cambiato senza promettere una data di consegna.")
for mode in modes:
    proc=subprocess.run(["./qw3-cli","-m",model,"--ngl","64","--ctx","512","--kv-f16","--nothink",
                         "--temp","0","-n","160","-p",prompt],env={**os.environ,"QW3_METAL_IQ1_BATCH":mode},
                        capture_output=True,text=True,check=True)
    print(json.dumps(dict(mode=mode,case="email",answer=proc.stdout.strip()),ensure_ascii=False),flush=True)
