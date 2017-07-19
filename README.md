# predict-word-stress
COMP9318 data mining &amp; machine learning project

Classifier:
Use bagging technique, random forest as classifier to predict word stress, reduce variance, and amplify the weak classification.

Feature selection:
focus on more physical properties of the word, such as part of speech, ending character, number of consonant, positional information of soundmarks. Positional information is constructed on the trigrams. 
For example:
RECONSTRUCTIONS
R IY2 K AH0 N S T R AH1 K SH AH0 N Z
Firstly, record the part of speech, number of consonant, ending character of the word as part of the features.
Secondly, for every trigram in this word, build index of the trigram as another feature, the trigram is defined as the formation of consonant-vowel-consonant. 
In this example:
R IY K, K AH N, R AH K, SH AH N as four trigrams to record, each trigram is represented as the combination of index in the CV list where CV = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']

Result:
CV can be 85%, but when uploading the system, only 70% more. Rank 4th place in the course.
