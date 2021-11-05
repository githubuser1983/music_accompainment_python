# music_accompainment_python
Python / Sagemath scripts to do music accompainment on AKAI LPK 25


I wanted to share some sagemath / python scripts to "produce" music like these pieces:
The name of pieces are in no particular order:
Ram Bam Dam, Light, Lively, Live Music Piano, Live Music Fantasia

Here are the links in no particular order:

https://drive.google.com/file/d/10zAd2YMn_Jg-5ORgasYI1GrU1qY7BBPO/view?usp=sharing

https://drive.google.com/file/d/19L9ZRwzCRtISzRfMwEG9XJXfCqUmaBOu/view?usp=sharing

https://drive.google.com/file/d/1H-q9bs3jxRtrg8YOJRxt4AibPDyRNXxo/view?usp=sharing

https://drive.google.com/file/d/1JIUTMb10qJi-wjWlgl-WkU0I4TsKsOO0/view?usp=sharing

https://drive.google.com/file/d/1KZJ88Wi1eK1VWqUCkA2weRO30MqrmNp0/view?usp=sharing

https://drive.google.com/file/d/1WoEhBKigax0nY19qQwt4sH5YP-6fTf1o/view?usp=sharing

https://drive.google.com/file/d/1c6cSOUQhc6MZRbHm7FVfI73L62wGshAG/view?usp=sharing

https://drive.google.com/file/d/1hg8HQe5maJ1v06g5yHhaeB0x84ofLYF2/view?usp=sharing

https://drive.google.com/file/d/1mnBcbAFUMyIYurNXLLklPnUFMp9lKW9b/view?usp=sharing

https://drive.google.com/file/d/1vWxKTdOt4RvD7h39JYb6Pm35pMyF_L7O/view?usp=sharing


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
