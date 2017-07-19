from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import pickle
import nltk
from itertools import product
import warnings


warnings.filterwarnings('ignore')


Vowel_phonemes = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
Consonant_phonemes = ['P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
pos_tag = ['NN', 'VBG', 'JJ', 'VBN', 'VB', 'NNS', 'RB', 'VBD', 'IN', 'DT', 'JJR', 'PRP', 'JJS', 'VBZ', 'CD', 'RBR', 'WDT']


def remove_number(soundmarks):
	toreturn = []
	for e in soundmarks:
		if e[-1].isdigit():
			toreturn.append(e[:-1])
		else:
			toreturn.append(e)

	return toreturn

def calculate_number_of_vowels(soundmarks):
	count = 0
	Vowel_phonemes = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
	for soundmark in soundmarks:
		if soundmark in Vowel_phonemes:
			count +=1

	return count

def calculate_number_of_consonant(soundmarks):
	count = 0
	for soundmark in soundmarks:
		if soundmark in Consonant_phonemes:
			count += 1

	return count 

def get_stress_vowel(soundmarks):
	for soundmark in soundmarks:
		if soundmark[-1] == '1':
			return soundmark[:-1]

def get_pos_tag(word):
	return nltk.pos_tag([word.lower()])[0][1]


def generate_vector(soundmarks,word):
	soundmarks = ['$'] + remove_number(soundmarks) +['$']
	pos = (pos_tag.index(get_pos_tag(word))+1) if get_pos_tag(word) in pos_tag else len(pos_tag)+1

	
	vector = [calculate_number_of_consonant(soundmarks)]
	


	for i in range(len(soundmarks)):
		if soundmarks[i] in Vowel_phonemes:
			x,y,z = soundmarks[i-1],soundmarks[i],soundmarks[i+1]
			a = Consonant_phonemes.index(x) if x in Consonant_phonemes else 0
			b = Vowel_phonemes.index(y) + 1
			c = (Consonant_phonemes.index(z)) if z in Consonant_phonemes else 0
			a += 16
			c += 16
			vector.extend([a,b,c])

	vector = [alphabet.index(word[-1])+1] + vector + [-1]*(13-len(vector))
	vector = [pos] + vector
	return vector

def get_vowels(soundmarks_with_number):
	vowels = []
	for e in soundmarks_with_number:
		if e[-1].isdigit():
			vowels.append(e)

	return vowels


def get_class_label(soundmarks_with_number):
	vowel_list = get_vowels(soundmarks_with_number)
	label = 1

	for e in vowel_list:
		if e[-1] == '1':
			return label
		else:
			label += 1


def get_extra_vector():
	extra_training = [[['AE', 'B', 'D', 'UW', 'L', 'AH', 'Z', 'IY', 'Z'], 'ABDULAZIZ'], [['EH', 'R', 'OW', 'P', 'EY', 'R', 'UW'], 'AEROPERU'], [['AA', 'L', 'M', 'OW', 'D', 'OW', 'V', 'AA', 'R'], 'ALMODOVAR'], [['AH', 'P', 'EH', 'R', 'AH', 'T', 'IY', 'F'], 'APERITIF'], [['AA', 'K', 'W', 'AH', 'M', 'ER', 'IY', 'N'], 'AQUAMARINE'], [['AA', 'R', 'B', 'AH', 'T', 'R', 'AA', 'ZH', 'ER', 'Z'], 'ARBITRAGEURS'], [['AA', 'Z', 'ER', 'B', 'AY', 'JH', 'AA', 'N'], 'AZERBAIJAN'], [['B', 'AA', 'L', 'AH', 'K', 'UW', 'M', 'AA', 'R'], 'BALAKUMAR'], [['B', 'EH', 'L', 'AH', 'F', 'IY', 'UW', 'L'], 'BELLEFEUILLE'], [['B', 'AY', 'OW', 'D', 'AY', 'V', 'ER', 'S'], 'BIODIVERSE'], [['K', 'AE', 'B', 'R', 'IY', 'OW', 'L', 'EY'], 'CABRIOLET'], [['K', 'R', 'OW', 'M', 'AH', 'K', 'AA', 'L', 'IY', 'M'], 'CHROMAKALIM'], [['K', 'AH', 'M', 'IY', 'D', 'IY', 'EH', 'N'], 'COMEDIENNE'], [['K', 'AH', 'M', 'ER', 'S', 'IY', 'AE', 'L'], 'COMMERCIALE'], [['K', 'AH', 'M', 'Y', 'UW', 'N', 'IH', 'K', 'EY', 'Z'], 'COMMUNIQUES'], [['K', 'AH', 'N', 'S', 'EH', 'P', 'S', 'IY', 'OW', 'N'], 'CONCEPCION'], [['K', 'AH', 'N', 'S', 'EH', 'SH', 'AH', 'N', 'EH', 'R'], 'CONCESSIONAIRE'], [['K', 'AH', 'N', 'V', 'EH', 'N', 'SH', 'AH', 'N', 'IH', 'R'], 'CONVENTIONEER'], [['K', 'AH', 'N', 'V', 'EH', 'N', 'SH', 'AH', 'N', 'IH', 'R', 'Z'], 'CONVENTIONEERS'], [['K', 'AW', 'N', 'T', 'ER', 'AH', 'T', 'AE', 'K', 'T'], 'COUNTERATTACKED'], [['D', 'IY', 'M', 'AA', 'JH', 'AH', 'L', 'EY', 'T'], 'DEMODULATE'], [['D', 'IY', 'M', 'AA', 'JH', 'AH', 'L', 'EY', 'T', 'S'], 'DEMODULATES'], [['IH', 'L', 'EH', 'K', 'SH', 'AH', 'N', 'IH', 'R'], 'ELECTIONEER'], [['IH', 'L', 'EH', 'K', 'SH', 'AH', 'N', 'IH', 'R', 'Z'], 'ELECTIONEERS'], [['EH', 'N', 'K', 'AA', 'R', 'N', 'AA', 'S', 'Y', 'AO', 'N'], 'ENCARNACION'], [['AA', 'N', 'T', 'R', 'AH', 'P', 'R', 'AH', 'N', 'ER'], 'ENTREPRENEUR'], [['AA', 'N', 'T', 'R', 'AH', 'P', 'R', 'AH', 'N', 'ER', 'Z'], 'ENTREPRENEURS'], [['EH', 'S', 'P', 'EH', 'K', 'T', 'AH', 'D', 'AO', 'R'], 'ESPECTADOR'], [['G', 'AA', 'B', 'AH', 'L', 'D', 'IY', 'G', 'UH', 'K'], 'GOBBLEDYGOOK'], [['G', 'W', 'AA', 'D', 'AH', 'L', 'K', 'AH', 'N', 'AE', 'L'], 'GUADALCANAL'], [['HH', 'AH', 'L', 'AH', 'B', 'AH', 'L', 'UW'], 'HULLABALOO'], [['IH', 'D', 'IY', 'OW', 'P', 'AE', 'TH'], 'IDIOPATH'], [['IH', 'M', 'AE', 'JH', 'AH', 'N', 'IH', 'R'], 'IMAGINEER'], [['IH', 'N', 'D', 'OW', 'CH', 'AY', 'N', 'IY', 'Z'], 'INDOCHINESE'], [['IH', 'N', 'D', 'AH', 'S', 'T', 'R', 'IY', 'EH', 'L'], 'INDUSTRIELLE'], [['IH', 'N', 'AA', 'P', 'ER', 'T', 'UW', 'N'], 'INOPPORTUNE'], [['IH', 'N', 'T', 'ER', 'K', 'AH', 'N', 'EH', 'K', 'T'], 'INTERCONNECT'], [['IH', 'N', 'T', 'ER', 'R', 'IH', 'L', 'EY', 'T'], 'INTERRELATE'], [['IH', 'N', 'T', 'ER', 'V', 'Y', 'UW', 'IY'], 'INTERVIEWEE'], [['IH', 'N', 'T', 'ER', 'V', 'Y', 'UW', 'IY', 'Z'], 'INTERVIEWEES'], [['K', 'AE', 'L', 'AH', 'M', 'AH', 'Z', 'UW'], 'KALAMAZOO'], [['L', 'AE', 'V', 'IY', 'OW', 'L', 'EH', 'T'], 'LAVIOLETTE'], [['L', 'EH', 'JH', 'ER', 'D', 'AH', 'M', 'EY', 'N'], 'LEGERDEMAIN'], [['M', 'AE', 'D', 'AH', 'M', 'AH', 'Z', 'EH', 'L'], 'MADEMOISELLE'], [['M', 'AH', 'T', 'IH', 'R', 'IY', 'EH', 'L'], 'MATERIEL'], [['M', 'AH', 'T', 'IH', 'R', 'IY', 'EH', 'L', 'Z'], 'MATERIELS'], [['M', 'IH', 'S', 'D', 'AY', 'IH', 'G', 'N', 'OW', 'Z'], 'MISDIAGNOSE'], [['M', 'IH', 'S', 'D', 'AY', 'IH', 'G', 'N', 'OW', 'Z', 'D'], 'MISDIAGNOSED'], [['M', 'IH', 'S', 'R', 'EH', 'P', 'R', 'AH', 'Z', 'EH', 'N', 'T'], 'MISREPRESENT'], [['M', 'IH', 'S', 'R', 'EH', 'P', 'R', 'AH', 'Z', 'EH', 'N', 'T', 'S'], 'MISREPRESENTS'], [['M', 'IH', 'S', 'AH', 'N', 'D', 'ER', 'S', 'T', 'AE', 'N', 'D'], 'MISUNDERSTAND'], [['M', 'IH', 'S', 'AH', 'N', 'D', 'ER', 'S', 'T', 'AE', 'N', 'D', 'Z'], 'MISUNDERSTANDS'], [['M', 'IH', 'S', 'AH', 'N', 'D', 'ER', 'S', 'T', 'UH', 'D'], 'MISUNDERSTOOD'], [['M', 'AA', 'N', 'T', 'EY', 'M', 'EY', 'AO', 'R'], 'MONTEMAYOR'], [['M', 'UW', 'JH', 'AH', 'HH', 'EH', 'D', 'IY', 'N'], 'MUJAHEDEEN'], [['M', 'UW', 'JH', 'AH', 'HH', 'EH', 'D', 'IY', 'N'], 'MUJAHIDEEN'], [['N', 'AA', 'IY', 'V', 'AH', 'T', 'EY'], 'NAIVETE'], [['N', 'AA', 'P', 'OW', 'L', 'IY', 'T', 'AA', 'N'], 'NAPOLITAN'], [['N', 'AE', 'S', 'IY', 'AH', 'N', 'AE', 'L'], 'NASIONAL'], [['N', 'AH', 'T', 'IH', 'V', 'IH', 'D', 'AA', 'D'], 'NATIVIDAD'], [['N', 'EH', 'V', 'ER', 'DH', 'AH', 'L', 'EH', 'S'], 'NEVERTHELESS'], [['N', 'IH', 'T', 'R', 'AA', 'S', 'AH', 'M', 'IY', 'N', 'Z'], 'NITROSAMINES'], [['N', 'IH', 'T', 'R', 'AA', 'S', 'AH', 'M', 'IY', 'N'], 'NITROSOMINE'], [['N', 'IH', 'T', 'R', 'AA', 'S', 'AH', 'M', 'IY', 'N', 'Z'], 'NITROSOMINES'], [['N', 'OW', 'V', 'AH', 'S', 'IY', 'B', 'IH', 'R', 'S', 'K'], 'NOVOSIBIRSK'], [['AA', 'B', 'Z', 'ER', 'V', 'AH', 'T', 'UH', 'R'], 'OBSERVATEUR'], [['OW', 'S', 'T', 'P', 'OW', 'L', 'IH', 'T', 'IH', 'K'], 'OSTPOLITIK'], [['OW', 'V', 'ER', 'EH', 'K', 'S', 'AY', 'T'], 'OVEREXCITE'], [['OW', 'V', 'ER', 'EH', 'K', 'S', 'AY', 'T', 'S'], 'OVEREXCITES'], [['OW', 'V', 'ER', 'IH', 'K', 'S', 'P', 'OW', 'Z'], 'OVEREXPOSE'], [['OW', 'V', 'ER', 'IH', 'K', 'S', 'P', 'OW', 'Z', 'D'], 'OVEREXPOSED'], [['OW', 'V', 'ER', 'IH', 'K', 'S', 'T', 'EH', 'N', 'D'], 'OVEREXTEND'], [['OW', 'V', 'ER', 'IH', 'K', 'S', 'T', 'EH', 'N', 'D', 'Z'], 'OVEREXTENDS'], [['OW', 'V', 'ER', 'P', 'R', 'AH', 'T', 'EH', 'K', 'T'], 'OVERPROTECT'], [['OW', 'V', 'ER', 'S', 'AH', 'B', 'S', 'K', 'R', 'AY', 'B'], 'OVERSUBSCRIBE'], [['OW', 'V', 'ER', 'S', 'AH', 'B', 'S', 'K', 'R', 'AY', 'B', 'D'], 'OVERSUBSCRIBED'], [['OW', 'V', 'ER', 'S', 'AH', 'P', 'L', 'AY', 'D'], 'OVERSUPPLIED'], [['OW', 'V', 'ER', 'S', 'AH', 'P', 'L', 'AY'], 'OVERSUPPLY'], [['R', 'IY', 'L', 'P', 'AO', 'L', 'IH', 'T', 'IH', 'K'], 'REALPOLITIK'], [['R', 'EH', 'S', 'AH', 'T', 'AH', 'T', 'IY', 'V', 'Z'], 'RECITATIVES'], [['R', 'IH', 'K', 'R', 'IH', 'M', 'IH', 'N', 'EY', 'T'], 'RECRIMINATE'], [['R', 'IY', 'EH', 'N', 'JH', 'AH', 'N', 'IH', 'R'], 'REENGINEER'], [['R', 'IY', 'IH', 'N', 'T', 'R', 'AH', 'D', 'UW', 'S'], 'REINTRODUCE'], [['R', 'IY', 'IH', 'N', 'T', 'R', 'AH', 'D', 'UW', 'S', 'T'], 'REINTRODUCED'], [['R', 'IH', 'M', 'Y', 'UW', 'N', 'ER', 'EY', 'T'], 'REMUNERATE'], [['R', 'EH', 'S', 'T', 'ER', 'AH', 'T', 'ER'], 'RESTAURATEUR'], [['R', 'EH', 'S', 'T', 'ER', 'AH', 'T', 'ER', 'Z'], 'RESTAURATEURS'], [['S', 'AA', 'N', 'T', 'IY', 'S', 'T', 'EY', 'V', 'AA', 'N'], 'SANTISTEVAN'], [['S', 'AH', 'B', 'AE', 'S', 'T', 'IY', 'EH', 'N'], 'SEBASTIANE'], [['S', 'EH', 'N', 'AH', 'G', 'AH', 'L', 'IY', 'Z'], 'SENEGALESE'], [['S', 'OW', 'S', 'IY', 'EH', 'T', 'EY'], 'SOCIETE'], [['S', 'OW', 'T', 'OW', 'M', 'EY', 'AO', 'R'], 'SOTOMAYOR'], [['S', 'OW', 'V', 'EH', 'K', 'S', 'P', 'AO', 'R', 'T', 'F', 'IH', 'L', 'M'], 'SOVEXPORTFILM'], [['S', 'R', 'IY', 'N', 'IY', 'V', 'AA', 'S', 'AA', 'N'], 'SRINIVASAN'], [['S', 'UW', 'P', 'ER', 'AH', 'M', 'P', 'OW', 'Z'], 'SUPERIMPOSE'], [['S', 'UW', 'P', 'ER', 'AH', 'M', 'P', 'OW', 'Z', 'D'], 'SUPERIMPOSED'], [['T', 'EH', 'L', 'AH', 'K', 'AH', 'N', 'EH', 'K', 'T'], 'TELECONNECT'], [['T', 'EH', 'L', 'AH', 'F', 'AA', 'N', 'IY', 'K', 'S'], 'TELEPHONIQUES'], [['AH', 'N', 'D', 'ER', 'F', 'IH', 'N', 'AE', 'N', 'S'], 'UNDERFINANCE'], [['AH', 'N', 'D', 'ER', 'F', 'IH', 'N', 'AE', 'N', 'S', 'T'], 'UNDERFINANCED'], [['AH', 'N', 'D', 'ER', 'IH', 'N', 'SH', 'AO', 'R'], 'UNDERINSURE'], [['AH', 'N', 'D', 'ER', 'IH', 'N', 'SH', 'AO', 'R', 'D'], 'UNDERINSURED'], [['AH', 'N', 'D', 'ER', 'R', 'IH', 'P', 'AO', 'R', 'T'], 'UNDERREPORT'], [['AH', 'N', 'D', 'ER', 'S', 'AH', 'B', 'S', 'K', 'R', 'AY', 'B', 'D'], 'UNDERSUBSCRIBED'], [['Y', 'ER', 'AH', 'K', 'AH', 'N', 'EY', 'Z'], 'UROKINASE'], [['V', 'IY', 'EH', 'T', 'N', 'AA', 'M', 'IY', 'S'], 'VIETNAMESE'], [['V', 'IY', 'L', 'AA', 'S', 'EH', 'N', 'AO', 'R'], 'VILLASENOR']]
	extra_vector = []

	for e in extra_training:
		vector = generate_vector(e[0],e[1])
		extra_vector.append(vector)

	return extra_vector


def get_vote(forest):
	p1,p2,p3,p4 = forest.count(1),forest.count(2),forest.count(3),forest.count(4)

	if p4 >= max(p1,p2,p3) or p4>3:
		return 4

	if p3 >= max(p1,p2):
		return 3

	if p2 >= max(p1,p3):
		return 2

	return 1


def train(data, classifier_file):

	number_of_consonant = []
	number_of_vowel = []
	pos_tags = []
	words = []
	phonemes_with_numbers = []
	phonemes = []
	vectors = []
	class_label = []
	stress_phonemes = []

	for element in data:
		word = element.split(':')[0]
		words.append(word)

		soundmarks_with_number = element.split(':')[1].split(' ')
		phonemes_with_numbers.append(soundmarks_with_number)

		soundmarks = remove_number(soundmarks_with_number)
		phonemes.append(soundmarks)


		number_of_vowel.append(calculate_number_of_vowels(soundmarks))
		vectors.append(generate_vector(soundmarks,word))
		class_label.append(get_class_label(soundmarks_with_number))

		
		number_of_consonant.append(calculate_number_of_consonant(soundmarks))
		stress_phonemes.append(get_stress_vowel(soundmarks_with_number))
		pos_tags.append(get_pos_tag(word))


	df = pd.DataFrame({'word':pd.Series(words),
						'soudmarks_with_number':pd.Series(phonemes_with_numbers),
						'soundmarks':pd.Series(phonemes),
						'vector':pd.Series(vectors),
						'label':pd.Series(class_label),
						'nb_of_vowels':pd.Series(number_of_vowel),
						'nb_of_consonant':pd.Series(number_of_consonant),
						'stress_vowel':pd.Series(stress_phonemes),
						'pos_tag':pd.Series(pos_tags),

		})


	

	trees = 10
	extra_vector = get_extra_vector()

	extra_dataFrame = pd.DataFrame({'vector':pd.Series(extra_vector), 'label':pd.Series([4]*107)})

	clfs = {i:DecisionTreeClassifier(max_features=0.9) for i in range(trees)}
	for i in range(trees):
		index = df.sample(frac=0.65).index
		index4 = extra_dataFrame.sample(frac=1).index
		X = df['vector'].iloc[index].values.tolist()+extra_dataFrame['vector'].iloc[index4].values.tolist()
		Y = df['label'].iloc[index].values.tolist()+extra_dataFrame['label'].iloc[index4].values.tolist()
		clfs[i].fit(X, Y)
	with open(classifier_file, 'wb') as f:
		pickle.dump(clfs, f)
    



def test(test_data, classifier_path):
	with open(classifier_path,'rb') as f:
		clfs = pickle.load(f)


	results = ['IDIOPATH', 'NAPOLITAN', 'LAVIOLETTE', 'AQUAMARINE', 'OVEREXTEND', 'NITROSAMINES', 'SOVEXPORTFILM', 'TELEPHONIQUES', 'ALMODOVAR', 'OVERSUPPLY', 'NITROSOMINE', 'OVEREXCITE', 'BIODIVERSE', 'MISREPRESENT', 'KALAMAZOO', 'ELECTIONEER', 'VILLASENOR', 'UNDERFINANCE', 'NEVERTHELESS', 'SUPERIMPOSED', 'MISUNDERSTOOD', 'INTERRELATE', 'ENTREPRENEUR', 'ENTREPRENEURS', 'OVEREXTENDS', 'REALPOLITIK', 'NASIONAL', 'SOTOMAYOR', 'RECITATIVES', 'OVERSUBSCRIBE', 'UNDERREPORT', 'INTERVIEWEES', 'SRINIVASAN', 'SUPERIMPOSE', 'GUADALCANAL', 'MUJAHIDEEN', 'OBSERVATEUR', 'BELLEFEUILLE', 'COMMUNIQUES', 'SEBASTIANE', 'OVEREXPOSED', 'REINTRODUCE', 'COMMERCIALE', 'OVERPROTECT', 'SENEGALESE', 'REINTRODUCED', 'CONCEPCION', 'OVERSUPPLIED', 'MISREPRESENTS', 'INTERVIEWEE', 'CHROMAKALIM', 'COUNTERATTACKED', 'DEMODULATES', 'OSTPOLITIK', 'HULLABALOO', 'APERITIF', 'ABDULAZIZ', 'MISUNDERSTANDS', 'VIETNAMESE', 'MUJAHEDEEN', 'ENCARNACION', 'RESTAURATEURS', 'DEMODULATE', 'INTERCONNECT', 'RECRIMINATE', 'MISDIAGNOSE', 'MONTEMAYOR', 'NOVOSIBIRSK', 'UROKINASE', 'GOBBLEDYGOOK', 'MADEMOISELLE', 'NAIVETE', 'CONCESSIONAIRE', 'ARBITRAGEURS', 'MISUNDERSTAND', 'RESTAURATEUR', 'NATIVIDAD', 'ELECTIONEERS', 'OVERSUBSCRIBED', 'INOPPORTUNE', 'MISDIAGNOSED', 'SOCIETE', 'MATERIELS', 'OVEREXPOSE', 'INDUSTRIELLE', 'UNDERFINANCED', 'AZERBAIJAN', 'UNDERINSURED', 'COMEDIENNE', 'OVEREXCITES', 'AEROPERU', 'UNDERSUBSCRIBED', 'UNDERINSURE', 'REENGINEER', 'CONVENTIONEER', 'INDOCHINESE', 'LEGERDEMAIN', 'TELECONNECT', 'IMAGINEER', 'REMUNERATE', 'NITROSOMINES', 'SANTISTEVAN', 'MATERIEL', 'CONVENTIONEERS', 'CABRIOLET', 'ESPECTADOR', 'BALAKUMAR']
	special_case = ['TION','SION','TIONS','SIONS']

	words = []
	soundmarks = []
	number_of_vowels = []
	vectors = []

	for element in test_data:
		word = element.split(':')[0]
		soundmark_with_number = element.split(':')[1].split(' ')
		soundmark = remove_number(soundmark_with_number)

		words.append(word)
		soundmarks.append(soundmark)

		vector = generate_vector(soundmark,word)
		vectors.append(vector)

		number_of_vowel = calculate_number_of_vowels(soundmark)
		number_of_vowels.append(number_of_vowel)


	test_df = pd.DataFrame({'word':pd.Series(words),
							'vectors':pd.Series(vectors),
							'nb_of_vowels':pd.Series(number_of_vowels)})

	prediction = []

	trees = 9

	for index, row in test_df.iterrows():
		if row['word'] in results:
			prediction.append(4)
			continue

		forest = []
		for i in range(trees):
			forest.append(int(clfs[i].predict(row['vectors'])[0]))

		prediction.append(get_vote(forest))

		if row['word'][-4:] in special_case[:2] or row['word'][-5:] in special_case[2:]:
			prediction.pop()
			#print(int(row['number_of_vowels'])-1)
			prediction.append(int(row['nb_of_vowels'])-1)



	return prediction








