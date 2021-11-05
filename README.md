# music_accompainment_python
Python / Sagemath scripts to do music accompainment on AKAI LPK 25


I wanted to share some sagemath / python scripts to "produce" music like the pieces in the folder midi.

Description:

    compute_knn_model.py will create a knn-model and dump it in the folder knn_models.
    
    scamp-piano.py will need a midi-keyboard as input and a configuration file.
    
    Probably you will need to install a few libraries with “pip install”.
    
    It runs on ubuntu with python 3.6 and sagemath 9.0 (sudo apt-get install sagemath)
    
    To run the scripts run:
    
    sage scamp-piano.py scamp-piano-configurations/cello-conf-loop-3.yaml
    
    You could however want to change the weights and produce a new knn-model-file. In this case:
    
    sage compute_knn_model.py
    
    The meaning of the weights is described in the file Measuring_note_similarity_with_positive_definite_kernels.pdf
