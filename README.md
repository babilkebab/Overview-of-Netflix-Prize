# README

Questo progetto, realizzato per la materia "Big Data" della magistrale in Informatica dell'Università di Catania (A.A. 2024/25), ha l'intento di riprodurre, in piccola scala, quanto avvenuto durante la challenge "Netflix Prize" indetta nel 2006, con tanto di versione ridotta del modello vincitore "BellKor's Pragmatic Chaos". 

I dettagli sono contenuti nel PDF "Overview of the Netflix Prize Challenge".


## Testare la pipeline


Per riprodurre fedelmente l'intera pipeline come descritto nel pdf della relazione, eseguire i seguenti passi:

* Installare le librerie nel requirements.txt
* Estrarre i dati del Netflix prize scaricati da [Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data/data)
* Inserire i vari file "combined_dataX.txt" e il file "probe.txt" in "txt_data"
* Eseguire "structures_building/extract_data_mappings_sparse.py" per ottenere la matrice sparsa CSR
* Eseguire "structures_building/similarities.py" e "structures_building/normalized_similarities.py" per ottenere le matrici con le similarità tra gli item
* Eseguire "structures_building/compute_biases.py" per calcolare i bias
* Eseguire tutti gli script in "src_naive_cf" per provare gli algoritmi di collaborative filtering basati su KNN
* Eseguire gli script in "global_ngbr" per addestrare e testare il GlobalNgbr
* Eseguire "structures_building/extract_data_to_df" per avere i dati del training set e del probe in formato .csv
* A questo punto è possibile addestrare e testare i modelli SVD, nella cartella "svd"
* Eseguire "autorec/autorec.py" per addestrare e testare l'AutoRec
* Eseguire "structures_building/compute_dicts_svdpp.py" per poter testare poi il modello SVD++ e MiniTimeSVD++
* Eseguire "structures_building/ensemble_probe_splitting.py" per dividere il probe set ed ottenere training e test set per l'ensemble
* Eseguire "ensemble/globalngbr.py" per ottenere le predizioni del GlobalNgbr
* Eseguire "ensemble/svd.py" seguito dal nome del modello SENZA ESTENSIONE per come è stato salvato nella cartella svd/models, per ottenere le predizioni per ogni modello SVD
* A questo punto eseguire "ensemble/ensemble.py" per testare l'ensemble.