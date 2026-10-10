# M5: condivisione della tabella IQ3_S nel decode (2026-10-10)

Il nuovo kernel [K64 prefill](m5_kstage_prefill_2026-10-10.md) avvicina il
prefill a llama.cpp, ma non modifica i matvec di decode. Questa prova privata
esamina la tabella IQ3_S FP32 da 8 KiB della coppia gate/up: quattro o otto
gruppi SIMD la condividono invece dei due nativi. Restano quattro righe per
gruppo SIMD, stessi pesi GGUF, input, prodotti e ordine delle somme FP32.
Cambiano soltanto inizializzazione della tabella e geometria del dispatch.

Dodici casi di matrice, sei per variante, passano il confronto finito bit per
bit contro il kernel nativo. Offset non nulli, guardie di output e input
immutato verificati. Le forme sono 256→1, 4352→33, 5120→4097/4111/17408,
17408→5120: alcune reinterpretano un prefisso limitato del tensore reale,
non sono tutte dimensioni effettive della coppia gate/up nel modello.

Due sessioni dello stesso motore partono da un identico prefisso sintetico
di 128 token, seed 1, contesto 512, 64 layer Metal, KV F16, quattro thread.
Sessanta secondi di pausa prima di ciascuna variante; 72 forward greedy
accoppiati AB/BA, primi otto esclusi dal timer. Ogni intero vocabolario
rimane finito ed esatto dopo prefisso e ciascun forward; history e stato
canonico K/V, DeltaNet e convoluzione finali sono esatti. Ogni variante
esegue 1368 dispatch candidati, cioè 72×19 coppie gate/up effettive.

| Gruppi SIMD | Baseline / candidato, tok/s | Variazione | Blocchi positivi su 8 |
| --- | ---: | ---: | ---: |
| 4 | 8,89054 / 8,87097 | −0,220% | 3 |
| 8 | 8,86440 / 8,86739 | +0,034% | 5 |

La variante a quattro gruppi peggiora il dato aggregato; quella a otto non
mostra un guadagno convincente. **Nessuna viene adottata.** Non si sommano
questi risultati a quelli di K64 e non si attribuisce alle due sessioni un
throughput assoluto di una normale sessione singola. Tutti i 128 campioni
temporali sono conservati, insieme a log, comandi, hash e snapshot;
sorgenti e binari rimangono immutati durante le prove.
[Archivio completo](measurements/iq3_groups_2026-10-10/results.json).

I target opzionali `qw3-iq3-groups-test` e `qw3-iq3-groups-bench` conservano
l'esperimento. Il test offre anche il controllo completo su otto prefissi
e checkpoint, ma questa campagna non lo ha eseguito: le varianti sono state
scartate dopo matrici e forward accoppiati. I controlli completi dei
checkpoint citati nel report K64 riguardano invece l'integrazione K64.

```sh
make qw3-iq3-groups-test qw3-iq3-groups-bench
QW3_METAL_KV_F16=1 ./qw3-iq3-groups-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 4
QW3_METAL_KV_F16=1 ./qw3-iq3-groups-test ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 8
QW3_METAL_KV_F16=1 caffeinate -disu ./qw3-iq3-groups-bench ../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf 4
# Per la seconda variante, sostituire 4 con 8 dopo la pausa.
```
