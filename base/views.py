from django.contrib.auth.mixins import UserPassesTestMixin
from django.urls import reverse_lazy
from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.views.generic.list import ListView


################## USER UTILS  ###########################
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import createUserForm

################ MODELS IMPOERTS ##########################
from .models import Patient, Disease

import numpy as np
import pandas as pd
import csv
import json
import pickle
from nltk.corpus import wordnet as wn

from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import itertools
import re

from nltk.wsd import lesk
from nltk.tokenize import word_tokenize

from sklearn.neighbors import KNeighborsClassifier

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


train_d = load_dataset("base/Data/final_train.csv")
train_df = train_df = train_d.drop('department', axis=1)
test_df = load_dataset("base/Data/final_test.csv")

# Create a disease - symptoms pair from the training dataset
disease_list = []
symptoms = []  # this contains a list of lists

for i in range(len(train_df)):
    symptoms.append(train_df.columns[train_df.iloc[i] == 1].to_list())
    disease_list.append(train_df.iloc[i, -1])

# get all symptoms columns. This is the set of all unique symptoms
symptom_cols = list(train_df.columns[:-1])

# a helper function to preprocess the symptoms: remove underscores, etc


def clean_symptom(symp):
    """
        symp: Symptom to clean
        Removes underscores, fullstops, etc
    """
    return symp.replace('_', ' ').replace('.1', '').replace('(typhos)', '').replace('yellowish', 'yellow').replace('yellowing', 'yellow')


# Apply the clean_symptom method to all symptoms
all_symptoms = [clean_symptom(symp) for symp in (symptom_cols)]


######################## TEXT PREPROCESSING  ###############################
# preprocesses documents based on the English language
nlp = spacy.load('en_core_web_sm')


def preprocess(document):
    nlp_document = nlp(document)
    d = []
    for token in nlp_document:
        if (not token.text.lower() in STOP_WORDS and token.text.isalpha()):
            d.append(token.lemma_.lower())
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
    list1 = string1.split(' ')
    list2 = string2.split(' ')
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

# Returns all the subsets of this set. This is a generator.


def powerset(seq):
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item
# Sort list based on length


def sort(a):
    for i in range(len(a)):
        for j in range(i+1, len(a)):
            if len(a[j]) > len(a[i]):
                a[i], a[j] = a[j], a[i]
    a.pop()
    return a

# find all permutations of a list


def permutations(s):
    permutations = list(itertools.permutations(s))
    return ([' '.join(permutation) for permutation in permutations])


def does_exist(txt):
    txt = txt.split(' ')
    combinations = [x for x in powerset(txt)]
    sort(combinations)
    for comb in combinations:
        for sym in permutations(comb):
            if sym in all_symptoms_preprocessed:
                return sym
    return False
# a helper function to help determine list of symptoms that contain a given pattern


def check_pattern(enquired_pat, symp_list):
    pred_list = []
    ptr = 0
    patt = "^" + enquired_pat + "$"
    regexp = re.compile(enquired_pat)
    for item in symp_list:
        if regexp.search(item):
            pred_list.append(item)
    if (len(pred_list) > 0):
        return 1, pred_list
    else:
        return ptr, None

#################################### SEMANTIC SIMILARITY ###########################


def word_sense_disambiguation(word, context):
    """Determines the meaning of a word based on the context it is used in"""
    sense = lesk(context, word)  # lesk is a WSD tool from the NLTK
    return sense


def semantic_distance(doc1, doc2):
    doc1_p = preprocess(doc1).split(' ')
    doc2_p = preprocess(doc2).split(' ')
    score = 0
    for token1 in doc1_p:
        for token2 in doc2_p:
            syn1 = word_sense_disambiguation(token1, doc1)
            syn2 = word_sense_disambiguation(token2, doc2)
            if syn1 is not None and syn2 is not None:
                x = syn1.wup_similarity(syn2)
                if x is not None and (x > 0.25):
                    score += x
    return score/(len(doc1_p)*len(doc2_p))


def semantic_similarity(symptom, corpus):
    max_sim = 0
    most_sim = None
    for symp in corpus:
        d = semantic_distance(symptom, symp)
        if d > max_sim:
            most_sim = symp
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
    symp = []
    synonyms = wn.synsets(sympt)
    lemmas = [word.lemma_names() for word in synonyms]
    lemmas = list(set(itertools.chain(*lemmas)))
    for e in lemmas:
        res, sym1 = semantic_similarity(e, all_symptoms_preprocessed)
        if res != 0:
            symp.append(sym1)
    return list(set(symp))


def one_hot_vector(client_sympts, all_sympts):
    """receives client_symptoms and returns a dataframe with 1 for associated symptoms 
        cleint_sympts: symptoms identified by user
        all_sympts: all symptoms in the dataset 
    """
    df = np.zeros([1, len(all_sympts)])
    for sym in client_sympts:
        df[0, all_sympts.index(clean_symptom(sym))] = 1
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
    tempt_df = df[df.prognosis == disease]
    m2 = (tempt_df == 1).any()
    return m2.index[m2].tolist()


def possible_diseases(symp):
    poss_dis = []
    for dis in set(disease_list):
        if contains(symp, sympts_of_disease(train_df, dis)):
            poss_dis.append(dis)
    return poss_dis


################################ MODEL TRAINING #############################
X_train = train_df.iloc[:, :-1]
X_test = test_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
y_test = test_df.iloc[:, -1]


def create_model():
    KNN_model = KNeighborsClassifier()
    return KNN_model


def train(X_train, y_train):
    model = create_model()
    X_train.columns = all_symptoms
    model.fit(X_train, y_train)
    # save model
    file = open("base/model/KNN.pickle", "wb")
    pickle.dump(model, file)
    file.close()


def load_model():
    file = open("base/model/KNN.pickle", "rb")
    model = pickle.load(file)
    return model


########################### DRIVER PROGRAM ######################
# Global variables
current_user = None
USER = None


def related_symptom(client_symp):
    sentence = "Could you please be more specific ? <br>"
    # i = len(s)
    for num, j in enumerate(client_symp):
        sentence += str(num) + ") " + clean_symptom(j) + "<br>"
    if num != 0:
        sentence += "Select the one you meant."
        return sentence
    else:
        return 0


def signUpPage(request):
    form = createUserForm()

    if request.method == 'POST':
        form = createUserForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            first_name = request.POST.get('first_name')
            last_name = request.POST.get('last_name')
            gender = request.POST.get('gender')
            age = request.POST.get('age')
            email = form.cleaned_data.get('email')

            # save patient
            patient, _ = Patient.objects.get_or_create(
                username=username, first_name=first_name, last_name=last_name, email=email, gender=gender, age=age)
            patient.first_name = first_name
            patient.last_name = last_name
            patient.email = email
            patient.gender = gender
            patient.age = age
            patient.save()
            messages.success(
                request, f'Successfully created an account for {username}!')
            return redirect('login')
    context = {
        'form': form
    }
    return render(request, 'base/register.html', context)


def logInPage(request):

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        # email = request.POST.get('email')
        user = authenticate(request, username=username, password=password)

        if user is not None:
            patient, _ = Patient.objects.get_or_create(username=username)
            global USER
            USER = user
            # patient.email = email
            patient.user = user
            patient.save()
            global current_user
            current_user = patient
            login(request, user)
            if user.is_staff:
                return redirect('admin-page')
            return redirect('chat')
        else:
            messages.info(request, 'Username Or Password incorrect!')

    context = {}
    return render(request, 'base/login.html', context)


def logoutUser(request):
    global USER, current_user
    # Reset current user to none upon logging out
    USER = None
    current_user = None
    logout(request)
    return redirect('login')


def admin_page(request):
    user = request.user
    if user.is_staff:
        # get all patients, starting from recently added
        patients = Patient.objects.all().order_by('-id')
        # Get 5 most recently registered patients
        latest_patients = patients[:5]
        # get all diseases, starting from recently diagnosed
        diseases = Disease.objects.all().order_by('-id')
        # Get 5 most recently diagnosed diseases
        lastest_diseases = diseases[:5]
        context = {
            "patients": patients,
            "latest_patients": latest_patients,
            "diseases": diseases,
            "lastest_diseases": lastest_diseases
        }
        return render(request, "base/admin-panel.html", context)
    else:
        return redirect('login')


class PatientList(UserPassesTestMixin, ListView):
    model = Patient
    context_object_name = 'patients'
    template_name = 'base/patient-list.html'

    def get_queryset(self, *args, **kwargs):
        # reverse the list to get the lastest first
        return Patient.objects.all().order_by('-id')

    # Guide against unathorized access
    def test_func(self):
        return self.request.user.is_staff

    def handle_no_permission(self):
        return redirect('login')


class DiseaseList(UserPassesTestMixin, ListView):
    model = Disease
    context_object_name = 'diseases'
    template_name = 'base/disease-list.html'

    def get_queryset(self):
        # reverse the list to get the lastest first
        return Disease.objects.all().order_by('-id')

    # Guide against unathorized access
    def test_func(self):
        return self.request.user.is_staff

    def handle_no_permission(self):
        return redirect('login')


def patient_profile(request):
    if current_user is not None:
        diseases = current_user.disease_set.all()
        context = {
            "user": current_user,
            "diseases": diseases
        }
        return render(request, "base/patient-profile.html", context)
    else:
        return redirect('login')


def chat_page(request):
    request.session.clear()
    if USER is not None:
        if USER.is_authenticated:
            # clear the current session when the user visits the bot page
            request.session.clear()
            return render(request, "base/chatbot.html")
        else:
            return redirect('login')
    else:
        return redirect('login')


def model_response(request):
    resp = request.GET.get('msg')  # resp is the response from the user
    session = request.session

    if "step" in session:
        if session["step"] == "QUIT":
            name = session["name"]
            age = session["age"]
            gender = session["gender"]
            session.clear()
            if resp.upper() == "Q":
                return HttpResponse(f'Thank you for using IntelliS, {name}.')
            else:
                session["step"] = "first_symp"
                session["name"] = name
                session["age"] = age
                session["gender"] = gender

    if "step" not in session:
        # return HttpResponse("What is your name ?")
        if resp.lower() == "ok":
            session["name"] = current_user.username
            session["gender"] = current_user.gender
            session["age"] = current_user.age
            session["step"] = "start"
            return HttpResponse(f'Hello {session["name"]}! I will ask few questions about your situation in order to help \
                                identify what you are sufferint from. Type start to continue with the diagnosis process')
        else:
            # Ask the user to reply "OK" if she fails to type ok upon the start of the program.
            return HttpResponse(f'Sorry {current_user.username}, I didn\'t understand that. Please reply OK to proceed with the conversation')

    if session['step'] == "start":
        if resp.lower() == "start":
            session['step'] = "GFS"   # GFS => Get First Symptom
            if session['gender'] == "MALE":
                session["gender_p"] = "Mr"     # gender_p => gender prefix
            elif session['gender'] == "FEMALE":
                session["gender_p"] = "Ms"
            else:
                session["gender_p"] = ""
        else:
            return HttpResponse(f'Sorry {session["name"]}! I didn\'t understand that. Please type start to continue.')

    if session['step'] == "GFS":
        session['step'] = "first_symp"  # first symptom
        return HttpResponse(f'What is the major symptom you are experiencing, {session["name"]}?')

    if session['step'] == "first_symp":
        sym1 = resp.lower()  # first symptom reported by user
        sym1 = preprocess(sym1)
        sim1, psym1 = syntactic_similarity(sym1, all_symptoms_preprocessed)
        temp = [sym1, sim1, psym1]
        session['F_SYM'] = temp  # information on the first symptom
        # now set 'step' to 'second_symp to get second symptom from user
        session['step'] = "second_symp"
        if sim1 == 1:  # sim1 == 1 if input does not match any exact symptom, so we find the related symptom the user is trying to express from all the possible symptoms
            # related symptom  (of first symptom)
            session['step'] = "related_symp1"
            resp = related_symptom(psym1)
            print(resp)
            if resp != 0:
                return HttpResponse(resp)
        else:
            return HttpResponse("What other symptom are you experiencing?")

    if session['step'] == "related_symp1":
        temp = session['F_SYM']
        psym1 = temp[2]
        psym1 = psym1[int(resp)]
        temp[2] = psym1
        session['F_SYM'] = temp
        # # now set 'step' to 'second_symp to get second symptom from user
        session['step'] = "second_symp"
        return HttpResponse("What other symptom are you experiencing?")
    if session['step'] == "second_symp":
        sym2 = resp.lower()
        sym2 = preprocess(sym2)
        sim2 = 0
        psym2 = []
        if len(sym2) != 0:
            sim2, psym2 = syntactic_similarity(sym2, all_symptoms_preprocessed)
        temp = [sym2, sim2, psym2]
        session['S_SYM'] = temp  # information on the second symptom
        session['step'] = "semantic"  # semantic similarity
        if sim2 == 1:
            # related symptom of the user's reported (second symptom)
            session['step'] = "related_symp2"
            # take a look at the related symptom method to get an idea of what is returned.
            resp = related_symptom(psym2)
            if resp != 0:
                return HttpResponse(resp)
    # only runs if the symptom entered by the user can not directly be identified from the list of symptoms available to the model.
    if session['step'] == "related_symp2":
        temp = session['S_SYM']
        psym2 = temp[2]
        psym2 = psym2[int(resp)]
        temp[2] = psym2
        session['S_SYM'] = temp
        session['step'] = "semantic"
    if session['step'] == "semantic":
        temp = session["F_SYM"]  # retrieve info from the first symptom
        sym1 = temp[0]
        sim1 = temp[1]
        temp = session["S_SYM"]  # retrieve ithe second symptom's info
        sym2 = temp[0]
        sim2 = temp[1]
        if sim1 == 0 or sim2 == 0:
            session['step'] = "BFsim1=0"
        else:
            # next step, check for the possible_diseases
            session['step'] = "PD"
    if session['step'] == "BFsim1=0":
        if sim1 == 0 and len(sym1) != 0:
            sim1, psym1 = semantic_similarity(sym1, all_symptoms_preprocessed)
            temp = []
            temp.append(sym1)
            temp.append(sim1)
            temp.append(psym1)
            session['F_SYM'] = temp
            # process of semantic similarity=1 for the first symptom.
            session['step'] = "sim1=0"
        else:
            session['step'] = "BFsim2=0"
    if session['step'] == "sim1=0":  # semantic no => suggestion
        temp = session["F_SYM"]
        sym1 = temp[0]
        sim1 = temp[1]
        if sim1 == 0:
            if "suggested" in session:
                sugg = session["suggested"]
                if resp.lower() == "yes":
                    psym1 = sugg[0]
                    sim1 = 1
                    temp = session["F_SYM"]
                    temp[1] = sim1
                    temp[2] = psym1
                    session["F_SYM"] = temp
                    sugg = []
                else:
                    del sugg[0]
            if "suggested" not in session:
                session["suggested"] = suggest_symptom(sym1)
                sugg = session["suggested"]
            if len(sugg) > 0:
                msg = "Do you experience any " + sugg[0] + "?"
                return HttpResponse(msg)
        if "suggested" in session:
            del session["suggested"]
        session['step'] = "BFsim2=0"
    if session['step'] == "BFsim2=0":
        temp = session["S_SYM"]  # retrieve info from the second symptom
        sym2 = temp[0]
        sim2 = temp[1]
        if sim2 == 0 and len(sym2) != 0:
            sim2, psym2 = semantic_similarity(sym2, all_symptoms_preprocessed)
            temp = []
            temp.append(sym2)
            temp.append(sim2)
            temp.append(psym2)
            session['S_SYM'] = temp
            session['step'] = "sim2=0"
        else:
            session['step'] = "TEST"
    if session['step'] == "sim2=0":
        temp = session["S_SYM"]
        sym2 = temp[0]
        sim2 = temp[1]
        if sim2 == 0:
            if "suggested_2" in session:
                sugg = session["suggested_2"]
                if resp.lower() == "yes":
                    psym2 = sugg[0]
                    sim2 = 1
                    temp = session["S_SYM"]
                    temp[1] = sim2
                    temp[2] = psym2
                    session["S_SYM"] = temp
                    sugg = []
                else:
                    del sugg[0]
            if "suggested_2" not in session:
                session["suggested_2"] = suggest_symptom(sym2)
                sugg = session["suggested_2"]
            if len(sugg) > 0:
                msg = f"Do you experience any {sugg[0]}?"
                session["suggested_2"] = sugg
                return HttpResponse(msg)
        if "suggested_2" in session:
            del session["suggested_2"]
        # test if semantic and syntactic  suggestions were not found
        session['step'] = "TEST"
    if session['step'] == "TEST":
        temp = session["F_SYM"]
        sim1 = temp[1]
        psym1 = temp[2]
        temp = session["S_SYM"]
        sim2 = temp[1]
        psym2 = temp[2]
        if sim1 == 0 and sim2 == 0:
            # GO TO THE END
            result = None
            session['step'] = "END"
        else:
            if sim1 == 0:
                psym1 = psym2
                temp = session["F_SYM"]
                temp[2] = psym2
                session["F_SYM"] = temp
            if sim2 == 0:
                psym2 = psym1
                temp = session["S_SYM"]
                temp[2] = psym1
                session["S_SYM"] = temp
            session['step'] = 'PD'  # proceed to possible_diseases
    if session['step'] == 'PD':

        # create patient symp list
        temp = session["F_SYM"]
        sim1 = temp[1]
        psym1 = temp[2]
        temp = session["S_SYM"]
        sim2 = temp[1]
        psym2 = temp[2]
        if "all" not in session:
            session["asked"] = []
            session["all"] = [cols_dict[psym1], cols_dict[psym2]]
            print(f'All: {session["all"]}')
        session["diseases"] = possible_diseases(session["all"])
        print(f'session diseases: {session["diseases"]}')
        all_sym = session["all"]
        diseases = session["diseases"]
        dis = diseases[0]
        session["dis"] = dis
        session['step'] = "for_dis"
    if session['step'] == "DIS":
        if "symv" in session:
            if len(resp) > 0 and len(session["symv"]) > 0:
                symts = session["symv"]
                all_sym = session["all"]
                if resp.upper() == "YES":
                    all_sym.append(symts[0])
                    # session["all"] = [clean_symptom(sym) for sym in all_sym]
                    session["all"] = all_sym
                    print(
                        f'possible disease {possible_diseases(session["all"])}')
                del symts[0]
                session["symv"] = symts
        if "symv" not in session:
            session["symv"] = sympts_of_disease(train_df, session["dis"])
        if len(session["symv"]) > 0:
            if symts[0] not in session["all"] and symts[0] not in session["asked"]:
                asked = session["asked"]
                asked.append(clean_symptom(symts[0]))
                session["asked"] = asked
                symts = session["symv"]
                msg = f"Do you experience any {clean_symptom(symts[0])}?"
                return HttpResponse(msg)
            else:
                del symts[0]
                session["symv"] = symts
                resp = ""
                return model_response(request)
        else:
            PD = possible_diseases(session["all"])
            print(
                f"PD of possible diseases: {possible_diseases(session['all'])}")
            diseases = session["diseases"]
            if diseases[0] in PD:
                session["test_prediction"] = diseases[0]
                print(f'testpred: {session["test_prediction"]}')
                PD.remove(diseases[0])
            session["diseases"] = PD
            session['step'] = "for_dis"
    if session['step'] == "for_dis":
        diseases = session["diseases"]
        if len(diseases) <= 0:

            session["step"] = "PREDICT"
        else:
            session["dis"] = diseases[0]
            session["step"] = "DIS"
            session["symv"] = sympts_of_disease(train_df, session["dis"])
            return model_response(request)
        # Let model predict the diseases
    if session['step'] == "PREDICT":
        KNN_model = load_model()
        result = KNN_model.predict(
            one_hot_vector(session["all"], all_symptoms))
        print(f"The result is {result[0]}")
        session['step'] = "END"
    if session['step'] == "END":
        if result is not None:
            if result[0] != session["test_prediction"]:
                session['step'] = "QUIT"
                return HttpResponse("According to the symptoms specified, I am sorry I cannot predict your "
                                    "disease with confidence. <br> Please consult your doctor.Re-enter your major symptom or Tap Q to "
                                    "stop the conversation ")
            session['step'] = "GUIDELINES"
            session["disease"] = result[0]
            # Save diagnosed disease
            department = train_d.loc[train_d['prognosis']
                                     == session["disease"], 'department'].iloc[0]
            session["department"] = department
            d = Disease.objects.create(
                patient=current_user, title=result[0], department=department)
            d.save()
            return HttpResponse(f'Dear {session["name"]} , you may have {session["disease"]}. Tap G to get guidelines on what to do.')
        else:
            # Ask user if they want to continue with the conversation or not
            session['step'] = "QUIT"
            return HttpResponse("Do you want to provide more on what you feeling? Tap Q to stop the conversation")
    if session["step"] == "GUIDELINES":
        session["step"] = "FINAL"
        if resp.upper() == "G":
            # direct patient to the appropriate unit for further diagnosis and interaction with a doctor
            department = train_d.loc[train_d['prognosis']
                                     == session["disease"], 'department'].iloc[0]
            return HttpResponse(
                f'Kindly see a doctor at the "{department.upper()}" unit. Tap any key to continue...')
    if session['step'] == "FINAL":
        session['step'] = "BYE"
        return HttpResponse("Diagnosis completed. Do you need another medical consultation (yes or no)?")
    if session['step'] == "BYE":
        name = session["name"]
        age = session["age"]
        gender = session["gender"]
        session.clear()
        if resp.lower() == "yes":
            session["gender"] = gender
            session["name"] = name
            session["age"] = age
            session['step'] = "first_symp"
            return HttpResponse(f'What\'s your main symptom, {session["name"]}?')
        else:
            session['step'] = "start"
            session["gender"] = gender
            session["name"] = name
            session["age"] = age
            return HttpResponse(f' Thank you for using IntelliS for your diagnosis, {name}. Type "start" to start a new conversation')
