import numpy as np
import pandas as pd
import csv
import json
import pickle
from nltk.corpus import wordnet as wn 

from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import itertools
from itertools import chain
import re

from nltk.wsd import lesk
from nltk.tokenize import word_tokenize

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')

##################### DATASET IMPORTS ###########################
def load_dataset(path):
    """
        path: Path to dataset file to be loaded
    """
    return pd.read_csv(path)

def check_dataset(data):
    """
        data: Data to be checked
        Prints part of the dataset to assert its correctness 
    """
    print(data.head())


train_df = load_dataset("base/Data/final_train.csv")
test_df = load_dataset("base/Data/final_test.csv")

# Create a disease - symptoms pair from the training dataset
disease_list = []  
symptoms = [] # this contains a list of lists

for i in range(len(train_df)):
    symptoms.append(train_df.columns[train_df.iloc[i]==1].to_list())
    disease_list.append(train_df.iloc[i, -1])

# get all symptoms columns. This is the set of all unique symptoms
symptom_cols = list(train_df.columns[:-1])

# a helper function to preprocess the symptoms: remove underscores, etc
def clean_symptom(symp):
    """
        symp: Symptom to clean
        Removes underscores, fullstops, etc
    """
    return symp.replace('_',' ').replace('.1','').replace('(typhos)','').replace('yellowish','yellow').replace('yellowing','yellow')

# Apply the clean_symptom method to all symptoms
all_symptoms = [clean_symptom(symp) for symp in (symptom_cols)]


######################## TEXT PREPROCESSING  ###############################
nlp = spacy.load('en_core_web_sm')  # preprocesses documents based on the English language
def preprocess(document):
    nlp_document = nlp(document)
    d=[]
    for token in nlp_document:
        if(not token.text.lower()  in STOP_WORDS and  token.text.isalpha()):
            d.append(token.lemma_.lower() )
    return ' '.join(d)


# apply preprocessing to all the symptoms
all_symptoms_preprocessed = [preprocess(symp) for symp in all_symptoms]

# associates each preprocessed symp with the name of its original column
cols_dict = dict(zip(all_symptoms_preprocessed, symptom_cols))

########################### TRANSLATIONS ###############################
# translator = Translator()
# def translate(text, src_lan, dest_lang='en'): 
#     """
#         text: Text to translate
#         src_lan: Source language
#         dest_lan: Destination language
#     """
#     ar = translator.translate(text, src=src_lan, dest=dest_lang).text     
#     return ar 



############################ SYNTACTIC SIMILARITY #######################
# a helper function to calculate the Jaccard similarity
def jaccard_similary(string1, string2):
    list1=string1.split(' ')
    list2=string2.split(' ')
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

# a helper function to calculate the syntactic similarity between the symptom and corpus
def syntactic_similarity(symptom, corpus):
    most_sim = []
    poss_sym = []
    for symp in corpus:
        s = jaccard_similary(symptom, symp)
        most_sim.append(s)
    ordered = np.argsort(most_sim)[::-1].tolist()
    for i in ordered:
        if does_exist(symptom):
            return 1, [corpus[i]]
        if corpus[i] not in poss_sym and most_sim[i] != 0:
            poss_sym.append(corpus[i])
    if len(poss_sym):
        return 1, poss_sym
    else:
        return 0, None
    
#Returns all the subsets of this set. This is a generator.
def powerset(seq):
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item
#Sort list based on length
def sort(a):
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            if len(a[j])>len(a[i]):
                a[i],a[j]=a[j],a[i]
    a.pop()
    return a

# find all permutations of a list
def permutations(s):
    permutations = list(itertools.permutations(s))
    return([' '.join(permutation) for permutation in permutations])

def does_exist(txt):
    txt=txt.split(' ')
    combinations = [x for x in powerset(txt)]
    sort(combinations)
    for comb in combinations :
        for sym in permutations(comb):
            if sym in all_symptoms_preprocessed:
                return sym
    return False
# a helper function to help determine list of symptoms that contain a given pattern
def check_pattern(enquired_pat, symp_list):
    pred_list=[]
    ptr = 0
    patt = "^" + enquired_pat + "$"
    regexp = re.compile(enquired_pat)
    for item in symp_list:
        if regexp.search(item):
            pred_list.append(item)
    if(len(pred_list) > 0):
        return 1, pred_list
    else:
        return ptr, None

#################################### SEMANTIC SIMILARITY ###########################
def word_sense_disambiguation(word, context):
    """Determines the meaning of a word based on the context it is used in"""
    sense = lesk(context, word) # lesk is a WSD tool from the NLTK
    return sense

def semantic_distance(doc1,doc2):
    doc1_p=preprocess(doc1).split(' ')
    doc2_p=preprocess(doc2).split(' ')
    score=0
    for token1 in doc1_p:
        for token2 in doc2_p:
            syn1 = word_sense_disambiguation(token1,doc1)
            syn2 = word_sense_disambiguation(token2,doc2)
            if syn1 is not None and syn2 is not None :
                x = syn1.wup_similarity(syn2)
                if x is not None and (x > 0.1):
                    score+=x
    return score/(len(doc1_p)*len(doc2_p))

def semantic_similarity(symptom, corpus):
    max_sim=0
    most_sim=None
    for symp in corpus:
        d = semantic_distance(symptom, symp)
        if d > max_sim:
            most_sim=symp
            max_sim = d
    return max_sim, most_sim
# def my_simp(symptom, corpus):
#     max_sim = 0
#     nlp = spacy.load("en_core_web_lg")
#     doc = nlp(str(symptom))
#     most_sim = None
#     for sym in corpus:
#         d = doc.similarity(nlp(sym))
#         if d > max_sim:
#             most_sim=sym
#             max_sim = d
#     return max_sim, most_sim



def suggest_symptom(sympt):
    """Takes an expression from the user and suggests the possible symptom the user is referring to"""
    symp=[]
    synonyms = wn.synsets(sympt)
    lemmas=[word.lemma_names() for word in synonyms]
    lemmas = list(set(chain(*lemmas)))
    for e in lemmas:
        res, sym1 = semantic_similarity(e, all_symptoms_preprocessed)
        if res!=0:
            symp.append(sym1)
    return list(set(symp))

def one_hot_vector(client_sympts, all_sympts):
    """receives client_symptoms and returns a dataframe with 1 for associated symptoms 
        cleint_sympts: symptoms identified by user
        all_sympts: all symptoms in the dataset 
    """
    df = np.zeros([1, len(all_sympts)])
    for sym in client_sympts:
        df[0, all_sympts.index(sym)] = 1
    return pd.DataFrame(df, columns=all_symptoms)

def contains(small, big):
    """Check to see if a small set is contained in a bigger set"""
    status = True
    for i in small:
        if i not in big:
            status = False
    return status

def sympts_of_disease(df, disease):
    """receives an illness and returns all symptoms"""
    tempt_df = df[df.prognosis==disease]
    m2 = (tempt_df == 1).any()
    return m2.index[m2].tolist()

def possible_diseases(symp):
    poss_dis=[]
    for dis in set(disease_list):
        if contains(symp, sympts_of_disease(train_df, dis)):
            poss_dis.append(dis)
    return poss_dis

################################ MODEL TRAINING #############################
X_train = train_df.iloc[:,:-1]
X_test = test_df.iloc[:,:-1]
y_train = train_df.iloc[:,-1]
y_test = test_df.iloc[:,-1]

def create_model():
    KNN_model = KNeighborsClassifier()
    return KNN_model

def train(X_train,y_train):
    model = create_model()
    X_train.columns = all_symptoms
    model.fit(X_train, y_train)
    #save model
    file = open("base/model/KNN.pickle", "wb")
    pickle.dump(model, file)
    file.close()
    
def load_model(): 
    file = open("base/model/KNN.pickle","rb") 
    model = pickle.load(file) 
    return model 

########################### DRIVER PROGRAM ######################
def get_user_info():
    """Get user credentials for authentication
        This will be replaced by a proper authentication method when we implement the interface
    """
    print("Please enter your Name \t\t", end=" >> ")
    username = input("")
    print("Hi ", username + " ...")
    return str(username)

def related_symptom(client_symp):
    # """Determines which of the symptoms the user is trying to express"""
    # if len(client_symp)==1:
    #     return client_symp[0]
    # print("Searches related to input:" )
    # for num, s in enumerate(client_symp):
    #     s = clean_symptom(s)
    #     print(num,": ", s)
    # if num != 0:
    #     print(f"Select the one you meant (0 - {num})", end="")
    #     conf_input = int(input(""))
    # else:
    #     conf_input = 0
    # disease_input = client_symp[conf_input]
    # return disease_input

    s = "Could you please be more specific ? <br>"
    i = len(s)
    for num, j in enumerate(client_symp):
        s += str(num) + ") " + clean_symptom(j) + "<br>"
    if num != 0:
        s += "Select the one you meant."
        return s
    else:
        return 0


def write_json(new_data, filename='DATA.json'):
    with open(filename, 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["users"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent=4)