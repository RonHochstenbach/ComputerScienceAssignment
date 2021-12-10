#!/usr/bin/env python
# coding: utf-8

# General imports
# 

# In[270]:


import json

import pandas as pd
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold=50)

import re
import random
from sklearn.utils import shuffle
from collections import defaultdict
from itertools import combinations

#!pip install hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from sklearn.cluster import AgglomerativeClustering
from scipy.special import comb
from scipy import interpolate

#pip install jupyterthemes
#!jt -r


# Parameters

# In[336]:


hyper_parameters = {
    'seed' : 1234,
    'hashes': 200,    #originally 100
    'bands': 50,      #originally 20
    'bootstraps': 5,
    'gamma':0.756,
    'plot_q_grams': False, 
    'remove_same_shop': True
}


# Auxilliary functions

# In[238]:


# creating train- and test set
def train_test_split(data, selected):
  
  #create train set
  train_set = dict()
  counter = 0
  for j in selected:
    index = 0
    for model in data:
      for item in data[model]:
        if (j == index) and (item not in train_set.values()):
          train_set.update({counter: [item]})
          counter+=1
        index+=1   
  
  #create test set
  test_set = dict()
  count = 0
  for j in range(hyper_parameters['number_of_items']):
    index = 0
    if not j in selected:
      for model in data:
        for item in data[model]:
          if index == j:
            test_set.update({count: [item]})
            count+=1
          index+=1
  
  return train_set, test_set

#----------------------------------------------------------------------

#calculating dice coefficients for q-grams
def dice_coefficient(a, b, q):
  """dice coefficient 2nt/(na + nb)."""
  if not len(a) or not len(b): return 0.0
  #if shorter than 3(q), append something that doesnt occur after cleaning
  if len(a) == 1:  a=a+u'@' 
  if len(b) == 1:  b=b+u'$'
  if len(a) == 2:  a=a+u'(' 
  if len(b) == 2:  b=b+u')'
      
  a_bigram_list=[]
  for i in range(len(a)-(q-1)):
    a_bigram_list.append(a[i:i+q])
    
  b_bigram_list=[]
  for i in range(len(b)-(q-1)):
    b_bigram_list.append(b[i:i+q])

  a_bigrams = set(a_bigram_list)
  b_bigrams = set(b_bigram_list)
  overlap = len(a_bigrams & b_bigrams)

  dice_coeff = overlap * 2.0/(len(a_bigrams) + len(b_bigrams))
  return dice_coeff

#----------------------------------------------------------------------
#q-gram similarity measure between two candidates
def similarity_qgram(features_a,features_b,title_a, title_b, q,gamma,alpha): 
  
  sim = 0
  avgSim = 0
  m = 0 #number of matching keys
  w = 0 #weight of the matches
  nmk_a = features_a.keys()
  nmk_b = features_b.keys()
  for key_a in features_a.keys():
    for key_b in features_b.keys():
      #.replace(" ","") as we dont want to take whitespace into account when calculating simularity
      key_sim = dice_coefficient(str(key_a).replace(" ",""), str(key_b).replace(" ",""), q)
      if key_sim > gamma:
        value_sim = dice_coefficient(str(features_a[key_a]).replace(" ",""), str(features_b[key_b]).replace(" ",""), q)
        sim = sim + key_sim*value_sim
        m+=1
        w = w + key_sim
  
  if w > 0:
    avgSim = sim/w
  
  #theta = m/min(len(features_a),len(features_b))

  titleSim = dice_coefficient(str(title_a).replace(" ",""),str(title_b).replace(" ",""),q)

  final_similarity = alpha*titleSim + (1-alpha)*avgSim

  return final_similarity
    
#----------------------------------------------------------------------


# Test sim
# 

# In[151]:


data_cleaned = data_cleaner_extra(data)

alpha = 0.9
beta=0.35
gamma=0.85
q=3
#a = data_cleaned['42LS5700'][0]
#b = data_cleaned['24SL410U'][0]
#c = data_cleaned['24SL410U'][1]

#print(similarity_qgram(a['featuresMap'],b['featuresMap'],a['title'], b['title'], 3,0.85,0.9))
#print("next")
#print(similarity_qgram(a['featuresMap'],c['featuresMap'],a['title'], c['title'], 3,0.85,0.9))

duplicates = []
for mod in data_cleaned:
    if len(data_cleaned[mod])>1:
        for pair in combinations(data_cleaned[mod],2):
            assert pair[0]['modelID'] == pair[1]['modelID']
            duplicates.append([pair[0],pair[1]])

non_dupl = []
for i in range(400):
    first_key = random.choice(list(data_cleaned.keys()))
    second_key = random.choice(list(data_cleaned.keys()))
    if not data_cleaned[first_key][0]['modelID'] == data_cleaned[second_key][0]['modelID']:
        non_dupl.append([data_cleaned[first_key][0],data_cleaned[second_key][0]])
        
items = np.concatenate((duplicates,non_dupl))

for pair in items.copy():
    brand_1 = ''
    brand_1_found = False   #if we dont find a brand for a pair, dont delete
    brand_2 = ''
    brand_2_found = False

    #retrieving brands of the index
    for brand in brands:
      if brand in pair[0]['title']:
        brand_1 = brand
        brand_1_found = True
      if brand in pair[1]['title']:
        brand_2 = brand
        brand_2_found = True
            
    if ((not brand_1 == brand_2) and brand_1_found and brand_2_found):
      np.delete(items,np.where(items == pair))
    elif pair[0]['shop'] == pair[1]['shop']: #assuming the same shop doesn't list the same item twice, remove if same shop
      if hyper_parameters['remove_same_shop']:
        np.delete(items,np.where(items == pair))

print(len(items))
sim_dupl = []
sim_non_dupl = []


for pair in items:
    for i in pair:
        if any(word in brands for word in i['title']):
            i['title'].replace(word,"")
        for k,v in i['featuresMap'].items():
            if any(word in brands for word in v):
                i['featuresMap'][k] = i['featuresMap'][k].replace(word,"")

t = time.time()
k = 0
for pair in items.copy():
    similarity = similarity_qgram(pair[0]['featuresMap'],pair[1]['featuresMap'],pair[0]['title'],pair[1]['title'], q,gamma, alpha)
    k+=1
    if pair[0]['modelID'] == pair[1]['modelID']:
      sim_dupl.append(similarity)
      #if similarity < 0.15:
        #print(pair[0])
        #print("-------")
        #print(pair[1])
        #print("~~~.    ~~~.    ~~~.    ~~~.    ")
    else:
      sim_non_dupl.append(similarity)
    
print("Time per pair: " + str((time.time()-t)/k))  

bins = np.linspace(0, 1, 200)
plt.hist(sim_dupl, bins, alpha=0.5, label='sim_dupl')
plt.hist(sim_non_dupl, bins, alpha=0.5, label='sim_non_dupl')
plt.legend(loc='upper right')
plt.show()




# Importing & Analyzing the Data

# In[338]:


#Opening json data as python dictionary
with open('/Users/ronhochstenbach/Downloads/TVs-all-merged.json') as json_file:
  data = json.load(json_file)

print("The number of keys = " + str(len(data)))
hyper_parameters.update({'number_of_models': len(data)})

amazon = 0
newegg = 0
bestbuy = 0
thenerds = 0

#example of a double: UN46ES6580

for i in data:
  for j in data[i]:
    if 'newegg.com' in j.values():
      newegg+=1
    elif 'bestbuy.com' in j.values():
      bestbuy+=1
    elif 'amazon.com'in j.values():
      amazon+=1
    elif 'thenerds.net'in j.values():
      thenerds+=1

print("Number of TVs from NewEgg: " + str(newegg))
print("Number of TVs from Amazon: " + str(amazon))
print("Number of TVs from BestBuy: " + str(bestbuy))
print("Number of TVs from TheNerds: " + str(thenerds))
print("The number of values = " + str(newegg+amazon+bestbuy+thenerds))

hyper_parameters.update({'number_of_items': newegg+amazon+bestbuy+thenerds})

#making a list of all brands in the data
brands = []
for i in data:
  for j in data[i]:
    if ('Brand' in j['featuresMap'].keys()) and (j['featuresMap']['Brand'].lower() not in brands):
      brands.append(j['featuresMap']['Brand'].lower())
    elif ('Brand Name' in j.keys()) and (j['featuresMap']['Brand Name'].lower() not in brands):
      brands.append(j['featuresMap']['Brand Name'].lower())

print(brands)


# Cleaning the data

# In[350]:


#performing the same data-cleaning steps as in section 3.1 of Hartveld et al. (2018)
def data_cleaner(data):
  data_cleaned = data
  index = 0
  for i in data_cleaned:
    for j in data_cleaned[i]:

      #cleaning title
      j['title'] = j['title'].lower()

      j['title'] = j['title'].replace("inches", "inch")
      j['title'] = j['title'].replace("\"", "inch")
      j['title'] = j['title'].replace("-inch", "inch")
      j['title'] = j['title'].replace(" inch", "inch")
        
      j['title'] = j['title'].replace("hertz", "hz")
      j['title'] = j['title'].replace("-hz", "hz")
      j['title'] = j['title'].replace(" hz", "hz")

      #cleaning features
      for key in j['featuresMap'].keys():
        j['featuresMap'][key] = j['featuresMap'][key].lower()

        j['featuresMap'][key] = j['featuresMap'][key].replace("inches", "inch")
        j['featuresMap'][key] = j['featuresMap'][key].replace("\"", "inch")
        j['featuresMap'][key] = j['featuresMap'][key].replace("-inch", "inch")
        j['featuresMap'][key] = j['featuresMap'][key].replace(" inch", "inch")
        
        j['featuresMap'][key] = j['featuresMap'][key].replace("hertz", "hz")
        j['featuresMap'][key] = j['featuresMap'][key].replace("-hz", "hz")
        j['featuresMap'][key] = j['featuresMap'][key].replace(" hz", "hz")

  return data_cleaned


# Data cleaner ++

# In[226]:


#performing the same data-cleaning steps as in section 3.1 of Hartveld et al. (2018)
def data_cleaner_extra(data):
  data_cleaned = data
  index = 0
  for i in data_cleaned:
    for j in data_cleaned[i]:

        #cleaning title
        j['title'] = j['title'].lower()

        j['title'] = j['title'].replace("inches", "inch")
        j['title'] = j['title'].replace("\"", "inch")
        j['title'] = j['title'].replace("-inch", "inch")
        j['title'] = j['title'].replace(" inch", "inch")
        
        j['title'] = j['title'].replace("hertz", "hz")
        j['title'] = j['title'].replace("-hz", "hz")
        j['title'] = j['title'].replace(" hz", "hz")

        j['title'] = j['title'].replace(" pounds", "lb")
        j['title'] = j['title'].replace("pounds", "lb")
        j['title'] = j['title'].replace(" lbs", "lb")
        j['title'] = j['title'].replace("lbs", "lb")
        j['title'] = j['title'].replace(" lb", "lb")
        j['title'] = j['title'].replace(" lbs.", "lb")
        j['title'] = j['title'].replace("lbs.", "lb")
        j['title'] = j['title'].replace(" lb.", "lb")
        j['title'] = j['title'].replace("lb.", "lb")

        j['title'] = j['title'].replace("(", "")
        j['title'] = j['title'].replace(")", "")
        
        for string in j['title'].split():
            if any(char == "-" for char in string):
                stripe_loc = string.find("-")
                if stripe_loc > 0 and stripe_loc + 3 <= len(string):
                    if string[stripe_loc+2] == "/" and string[stripe_loc+3]:
                        if string[stripe_loc+1].isdigit() and string[stripe_loc+3].isdigit():
                        #calculate fractional value, round off 1 decimal
                            fract = round(int(string[stripe_loc+1])/int(string[stripe_loc+3]),1)
                            if stripe_loc > 1:
                                #fixing eg 12-1/2
                                if string[stripe_loc-1].isdigit() and string[stripe_loc-2].isdigit():
                                    numb = float(string[stripe_loc-2:stripe_loc-1])
                                    j['title'] = j['title'].replace(string[stripe_loc-2:stripe_loc+3], str(round(fract+numb,1)))
                                #fixing eg 8-1/2
                                else:
                                    numb = float(string[stripe_loc-1])
                                    j['title'] = j['title'].replace(string[stripe_loc-1:stripe_loc+3], str(round(fract+numb,1)))
                            elif stripe_loc ==1:
                                numb = float(string[stripe_loc-1])
                                j['title'] = j['title'].replace(string[stripe_loc-1:stripe_loc+3], str(round(fract+numb,1)))

            if any(char == "/" for char in string):
                dash_loc = string.find("/")
                if string[dash_loc-1].isdigit() and string[dash_loc+1].isdigit():
                    number = round(float(string[dash_loc-1])/float(string[dash_loc+1]),1)
                    j['title'] = j['title'].replace(string[dash_loc-1:dash_loc+1], str(number))
      
      #cleaning features
        for key in j['featuresMap'].keys():
            j['featuresMap'][key] = j['featuresMap'][key].lower()

            j['featuresMap'][key] = j['featuresMap'][key].replace("inches", "inch")
            j['featuresMap'][key] = j['featuresMap'][key].replace("\"", "inch")
            j['featuresMap'][key] = j['featuresMap'][key].replace("-inch", "inch")
            j['featuresMap'][key] = j['featuresMap'][key].replace(" inch", "inch")

            j['featuresMap'][key] = j['featuresMap'][key].replace("hertz", "hz")
            j['featuresMap'][key] = j['featuresMap'][key].replace("-hz", "hz")
            j['featuresMap'][key] = j['featuresMap'][key].replace(" hz", "hz")

            j['featuresMap'][key] = j['featuresMap'][key].replace(" pounds", "lb")
            j['featuresMap'][key] = j['featuresMap'][key].replace("pounds", "lb")
            j['featuresMap'][key] = j['featuresMap'][key].replace(" lbs", "lb")
            j['featuresMap'][key] = j['featuresMap'][key].replace("lbs", "lb")
            j['featuresMap'][key] = j['featuresMap'][key].replace(" lb", "lb")
            j['featuresMap'][key] = j['featuresMap'][key].replace(" lbs.", "lb")
            j['featuresMap'][key] = j['featuresMap'][key].replace("lbs.", "lb")
            j['featuresMap'][key] = j['featuresMap'][key].replace(" lb.", "lb")
            j['featuresMap'][key] = j['featuresMap'][key].replace("lb.", "lb")

            j['featuresMap'][key] = j['featuresMap'][key].replace("(", "")
            j['featuresMap'][key] = j['featuresMap'][key].replace(")", "")
            
            for string in j['featuresMap'][key].split():
                if any(char == "-" for char in string):
                    stripe_loc = string.find("-")
                    if stripe_loc > 0 and stripe_loc+3 <= len(string):
                        if string[stripe_loc+2] == "/":
                            if string[stripe_loc+1].isdigit() and string[stripe_loc+3].isdigit():
                            #calculate fractional value, round off 1 decimal
                                fract = round(int(string[stripe_loc+1])/int(string[stripe_loc+3]),1)
                                if stripe_loc > 1:
                                    #fixing eg 12-1/2
                                    if string[stripe_loc-1].isdigit() and string[stripe_loc-2].isdigit():
                                        numb = int(string[stripe_loc-2:stripe_loc])
                                        j['featuresMap'][key] = j['featuresMap'][key].replace(string[stripe_loc-2:stripe_loc+4], str(round(fract+numb,1)))
                                    #fixing eg 8-1/2
                                    else:
                                        numb = int(string[stripe_loc-1])
                                        j['featuresMap'][key] = j['featuresMap'][key].replace(string[stripe_loc-1:stripe_loc+4], str(round(fract+numb,1)))
                                elif stripe_loc ==1:
                                    numb = int(string[stripe_loc-1])
                                    j['featuresMap'][key] = j['featuresMap'][key].replace(string[stripe_loc-1:stripe_loc+4], str(round(fract+numb,1)))
                if any(char == "/" for char in string):
                    dash_loc = string.find("/")
                    if string[dash_loc-1].isdigit() and string[dash_loc+1].isdigit():
                        if not float(string[dash_loc+1]) == 0:
                            number = round(float(string[dash_loc-1])/float(string[dash_loc+1]),1)
                            j['featuresMap'][key] = j['featuresMap'][key].replace(string[dash_loc-1:dash_loc+2], str(number))
                           
  return data_cleaned


# Making lists of model words

# In[227]:


def model_words(input_data):
  MW_title = np.array([])
  MW_value = np.array([])

  not_model_words = ['newegg.com', 'bestbuy.com', 'amazon.com', 'thenerds.com']
  for i in input_data:
    for j in input_data[i]:
      
      #creating title model words
      for word_in_title in j['title'].split():
        
        if any(letter.isalpha() for letter in word_in_title):           #combi of alphanumeric with either digit or special, or both
          if any(not letter.isalpha() for letter in word_in_title):
            if (word_in_title not in MW_title) and (word_in_title not in not_model_words):
              MW_title = np.append(MW_title,word_in_title)
        
        elif any(letter.isdigit() for letter in word_in_title):         #combi of digit and special
            if any(not letter.isdigit() for letter in word_in_title):
              if (word_in_title not in MW_title) and (word_in_title not in not_model_words):
                MW_title = np.append(MW_title,word_in_title)
      
      #creating value model words
      for value in j['featuresMap'].values():
        for word_in_value in value.split():                             #check whether any word in the value contains both a digit and a '.'
          if any(letter == '.' for letter in word_in_value):            
            if any(letter.isdigit() for letter in word_in_value):
              if re.sub("[^\d\.]", "", word_in_value) not in MW_value:
                #if present, we use the word that contains both digit and '.' as model word after removing the alphanumeric part (if present) 
                MW_value = np.append(MW_value,re.sub("[^\d\.]", "", word_in_value))

  #print("The number of model words in title = " + str(len(MW_title)))
  #print("The number of model words in values = " + str(len(MW_value)))
  return MW_title, MW_value  

  #np.savetxt('/content/drive/MyDrive/model_words_ron.csv', MW_title,delimiter=',',fmt='% s')


# Creating representations

# In[228]:


def add_representation_base(input_data, MW_title, MW_value):
  data_representation = input_data

  for i in data_representation:
    for j in data_representation[i]:
      #for now we do binary vector representation
      
      #title part
      rep_title = np.zeros(len(MW_title),dtype=int)
      k=0
      for MW in MW_title:
        if MW in j['title'] or MW in j['featuresMap'].values():
          rep_title[k] = 1
        k+=1

      #value part
      rep_value = np.zeros(len(MW_value),dtype=int)
      k=0
      for MW in MW_value:
        for value in j['featuresMap'].values():
          if MW in value:
            rep_value[k]=1
        k+=1
      
      #combine into one value
      rep = np.concatenate((rep_title,rep_value))

      #adding the representation value as a key-value pair to the dictionary
      j.update({'representation': rep})

  return data_representation


# Functions 1: finding candidate pairs by performing minhashing and LSH
# 
# 
# 
# 

# In[229]:


def LSH(considered_items, model_wrds):

  #create signature matrix
  signature_matrix = np.zeros((hyper_parameters['hashes'],len(considered_items)))

  index = np.array([])  #array of indices of MW's, that we can shuffle
  for i in range(len(model_wrds)):
    index = np.append(index,i).astype(int)

  #Min-Hashing
  for i in range(hyper_parameters['hashes']):
    permutation = shuffle(index, random_state = i)
    item = 0      #cycling over all tv's
    for entry in considered_items:
      for tv in considered_items[entry]:
        for k in range(len(model_wrds)):    #cycling over all model words
          if tv['representation'][permutation[k]] == 1:
            signature_matrix[i][item] = k
            item+=1
            break

  signature_matrix = signature_matrix.astype(int)
    
  #Locality-Sensitive Hashing
  buckets = defaultdict(set)
  bands = np.array_split(signature_matrix, hyper_parameters['bands'], axis=0)

  for i, band in enumerate(bands):
    for j in range(signature_matrix.shape[1]):
      band_id = tuple(list(band[:,j])+[str(i)])
      buckets[band_id].add(j)
  
  candidates_duplicates = set()
  #print("no buckets: "+ str(len(buckets)))


  for bucket in buckets.values():
    if len(bucket) > 1:
      for pair in combinations(bucket,2):
        candidates_duplicates.add(pair)

  dissimilarities = np.full((signature_matrix.shape[1],signature_matrix.shape[1]), -1*np.inf)
  unique_candidates = candidates_duplicates
  

  remove_set = set()
  for i in unique_candidates:
    item_1 = i[0]
    item_2 = i[1]
    if dissimilarities[item_1][item_2] == 1:
      remove_set.add(i)
    dissimilarities[item_1][item_2] = 1
    dissimilarities[item_2][item_1] = 1

  #print("candidates: " +str(len(unique_candidates)))
  #remove duplicates
  for i in remove_set:
    unique_candidates.remove(i)

  candidate_pairs = []

  for i in unique_candidates:
    item_1 = considered_items[i[0]][0]
    item_2 = considered_items[i[1]][0]
    candidate_pairs.append([item_1,item_2])

  return candidate_pairs


# Final classification or clustering with only filtering out when brand not same
# 

# In[230]:


def final_classification_base(candidate_pairs, considered_items):

  duplicates = candidate_pairs

  for duplicate in duplicates.copy():
    #remove candidate pairs of the same brand
    brand_1 = ''
    brand_1_found = False   #if we dont find a brand for a pair, dont delete
    brand_2 = ''
    brand_2_found = False

    #retrieving brands of the index
    for brand in brands:
      if brand in duplicate[0]['title']:
        brand_1 = brand
        brand_1_found = True
      if brand in duplicate[1]['title']:
        brand_2 = brand
        brand_2_found = True
            
    if ((not brand_1 == brand_2) and brand_1_found and brand_2_found):
      duplicates.remove(duplicate)
    
    #assuming the same shop doesn't list the same item twice, remove if same shop
    #elif duplicate[0]['shop'] == duplicate[1]['shop']:
    #  duplicates.remove(duplicate)

  return duplicates


# Final Classification with q-grams

# In[231]:


def final_classification_qgram(candidate_pairs, considered_items, q, alpha, beta, gamma):

  duplicates = candidate_pairs

  for duplicate in duplicates.copy():
    #remove candidate pairs of different brand
    brand_1 = ''
    brand_1_found = False   #if we dont find a brand for a pair, dont delete
    brand_2 = ''
    brand_2_found = False

    #retrieving brands of the index
    for brand in brands:
      if brand in duplicate[0]['title']:
        brand_1 = brand
        brand_1_found = True
      if brand in duplicate[1]['title']:
        brand_2 = brand
        brand_2_found = True
            
    if ((not brand_1 == brand_2) and brand_1_found and brand_2_found):
      duplicates.remove(duplicate)
    elif duplicate[0]['shop'] == duplicate[1]['shop']: #assuming the same shop doesn't list the same item twice, remove if same shop
      if hyper_parameters['remove_same_shop']:
        duplicates.remove(duplicate)

  #as now all remaining candidates and have same brand, dont pay attention to this for similarity
  for pair in duplicates:
        for i in pair:
           if any(word in brands for word in i['title']):
              i['title'].replace(word,"")
           for k,v in i['featuresMap'].items():
                  if any(word in brands for word in v):
                    i['featuresMap'][k] = i['featuresMap'][k].replace(word,"")

  #duplicate detection using q-grams in MSM
  sim_dupl = []
  sim_non_dupl = []
  #print("Remaining pre q-gram: " + str(len(duplicates)))
  count = 0

  for pair in duplicates.copy():
    similarity = similarity_qgram(pair[0]['featuresMap'],pair[1]['featuresMap'],pair[0]['title'],pair[1]['title'], q,gamma, alpha)
    if similarity < beta:
      duplicates.remove(pair)
      count+=1

    if pair[0]['modelID'] == pair[1]['modelID']:
      sim_dupl.append(similarity)
    else:
      sim_non_dupl.append(similarity)
  
  if hyper_parameters['plot_q_grams']:
    bins = np.linspace(0, 1, 20)
    plt.hist(sim_dupl, bins, alpha=0.5, label='sim_dupl')
    plt.legend(loc='upper right')
    plt.show()
    bins = np.linspace(0, 1, 100)
    plt.hist(sim_non_dupl, bins, alpha=0.5, label='sim_non_dupl')
    plt.legend(loc='upper right')
    plt.show()
 
  #print("Removed by q-gram: " + str(count))

  return duplicates


# Final classification set up to do hyperparameter optimization

# In[232]:


def class_qgram_hyperopt(space):
    
    alpha = space['alpha']
    beta = space['beta']
    gamma = space['gamma']

    q=3
    duplicates = candidate_pairs_after_LSH_hyperopt

    for duplicate in duplicates.copy():
        #remove candidate pairs of the same brand
        brand_1 = ''
        brand_1_found = False   #if we dont find a brand for a pair, dont delete
        brand_2 = ''
        brand_2_found = False

        #retrieving brands of the index
        for brand in brands:
          if brand in duplicate[0]['title']:
            brand_1 = brand
            brand_1_found = True
          if brand in duplicate[1]['title']:
            brand_2 = brand
            brand_2_found = True

        if ((not brand_1 == brand_2) and brand_1_found and brand_2_found):
          duplicates.remove(duplicate)
        elif duplicate[0]['shop'] == duplicate[1]['shop']: #assuming the same shop doesn't list the same item twice, remove if same shop
          if hyper_parameters['remove_same_shop']:
            duplicates.remove(duplicate)

      #as now all remaining candidates and have same brand, dont pay attention to this for similarity
    for pair in duplicates:
        for i in pair:
            if any(word in brands for word in i['title']):
                i['title'].replace(word,"")
            for k,v in i['featuresMap'].items():
                if any(word in brands for word in v):
                    i['featuresMap'][k] = i['featuresMap'][k].replace(word,"")
                    
    #duplicate detection using q-grams in MSM
    for pair in duplicates.copy():
        similarity = similarity_qgram(pair[0]['featuresMap'],pair[1]['featuresMap'],pair[0]['title'],pair[1]['title'], q,gamma, alpha)
        if similarity < beta:
          duplicates.remove(pair)


    #count the number of correctly predicted duplicates
    D_f = 0

    for i in duplicates:
        modelIDs_found = dict()
        for j in i:
          if j['modelID'] in modelIDs_found.keys():
            modelIDs_found[j['modelID']] += 1
          else:
            modelIDs_found.update({j['modelID']: 1})
      
        for num_found in modelIDs_found.values():
          if num_found > 1:
            D_f = D_f + comb(num_found,2)   #not sure if legit  
        modelIDs_found.clear()
  
    #count the number of duplicates present
    D_n = 0
    model_IDs_present = dict()
    for i in train_set.values():
        for j in i:
          if j['modelID'] in model_IDs_present:
            model_IDs_present[j['modelID']] +=1
          else:
            model_IDs_present.update({j['modelID']:1})

    for num_present in model_IDs_present.values():
        if num_present > 1:
          D_n = D_n + comb(num_present,2)

    #print("duplicates present: " + str(D_n))
    N_c_final = len(duplicates)
    
    if not N_c_final == 0:
        Pair_Quality = D_f/N_c_final
    else: Pair_Quality = 0
        
    if not D_n ==0:
        Pair_Completeness = D_f/D_n
    else: Pair_Completeness = 0
        
    if not Pair_Quality+Pair_Completeness == 0:
        F1 = (2*Pair_Quality*Pair_Completeness)/(Pair_Quality+Pair_Completeness)
    else:
        F1=0
        
    if not comb(len(test_set.values()),2) == 0:
        frac_comp = N_c_LSH/comb(len(test_set.values()),2)
    else: frac_comp = 0

    return -1*F1


# Final Classification clustering

# In[337]:


def final_classcluster(candidate_pairs, considered_items, q, alpha, beta, gamma, delta):

  duplicates = candidate_pairs

  for duplicate in duplicates.copy():
    #remove candidate pairs of different brand
    brand_1 = ''
    brand_1_found = False   #if we dont find a brand for a pair, dont delete
    brand_2 = ''
    brand_2_found = False

    #retrieving brands of the index
    for brand in brands:
      if brand in duplicate[0]['title']:
        brand_1 = brand
        brand_1_found = True
      if brand in duplicate[1]['title']:
        brand_2 = brand
        brand_2_found = True
            
    if ((not brand_1 == brand_2) and brand_1_found and brand_2_found):
      duplicates.remove(duplicate)
    elif duplicate[0]['shop'] == duplicate[1]['shop']: #assuming the same shop doesn't list the same item twice, remove if same shop
      if hyper_parameters['remove_same_shop']:
        duplicates.remove(duplicate)

  #as now all remaining candidates and have same brand, dont pay attention to this for similarity
  for pair in duplicates:
        for i in pair:
           if any(word in brands for word in i['title']):
              i['title'].replace(word,"")
           for k,v in i['featuresMap'].items():
                  if any(word in brands for word in v):
                    i['featuresMap'][k] = i['featuresMap'][k].replace(word,"")

  #duplicate detection using q-grams in MSM
  sim_dupl = []
  sim_non_dupl = []
  #print("Remaining pre q-gram: " + str(len(duplicates)))
  count = 0

  for pair in duplicates.copy():
    similarity = similarity_qgram(pair[0]['featuresMap'],pair[1]['featuresMap'],pair[0]['title'],pair[1]['title'], q,gamma, alpha)
    if similarity < beta:
      duplicates.remove(pair)
      count+=1

  #now all these are removed, we cluster
  
  #count unique items still around
  unique_items = dict()
  count = 0
  for pair in duplicates:
        for i in pair:
            if i not in unique_items.values():
                unique_items.update({count: i})
                count+=1

                
  dist_mat = np.zeros((len(unique_items),len(unique_items)))
  for i in range(len(unique_items)):
    for j in range(len(unique_items)):
      if i<j:
        distance = 1-similarity_qgram(unique_items[i]['featuresMap'],unique_items[j]['featuresMap'],unique_items[i]['title'],unique_items[j]['title'], q,gamma, alpha)
        dist_mat[i,j] = distance
        dist_mat[j,i] = distance
  
  clustering = AgglomerativeClustering(n_clusters = None, distance_threshold = delta, affinity='precomputed', linkage = 'average').fit(dist_mat)
    
  labels = clustering.labels_
  assert len(labels) == len(unique_items)
  
  clusters = dict()
  
  for i in range(len(labels)):
    if labels[i] in clusters.keys():
        clusters[labels[i]] = np.append(clusters[labels[i]],unique_items[i])
    else: 
        clusters.update({labels[i]:[unique_items[i]]})

  final_predictions = []
  for group in clusters.values():
        if len(group)>1:
            for pair in combinations(group,2):
                final_predictions.append([pair[0],pair[1]])
                

  return final_predictions


# Class cluster set up to do hyperparameter optim

# In[351]:


def clusterclass_hyperopt(space):
    
    alpha = space['alpha']
    beta = space['beta']
    gamma = space['gamma']
    delta = space['delta']

    q=3
    duplicates = candidate_pairs_after_LSH_hyperopt

    for duplicate in duplicates.copy():
        #remove candidate pairs of different brand
        brand_1 = ''
        brand_1_found = False   #if we dont find a brand for a pair, dont delete
        brand_2 = ''
        brand_2_found = False

        #retrieving brands of the index
        for brand in brands:
          if brand in duplicate[0]['title']:
            brand_1 = brand
            brand_1_found = True
          if brand in duplicate[1]['title']:
            brand_2 = brand
            brand_2_found = True

        if ((not brand_1 == brand_2) and brand_1_found and brand_2_found):
          duplicates.remove(duplicate)
        elif duplicate[0]['shop'] == duplicate[1]['shop']: #assuming the same shop doesn't list the same item twice, remove if same shop
          if hyper_parameters['remove_same_shop']:
            duplicates.remove(duplicate)

    #as now all remaining candidates and have same brand, dont pay attention to this for similarity
    for pair in duplicates:
        for i in pair:
           if any(word in brands for word in i['title']):
              i['title'].replace(word,"")
           for k,v in i['featuresMap'].items():
                  if any(word in brands for word in v):
                    i['featuresMap'][k] = i['featuresMap'][k].replace(word,"")

    #duplicate detection using q-grams in MSM
    sim_dupl = []
    sim_non_dupl = []
    #print("Remaining pre q-gram: " + str(len(duplicates)))
    count = 0

    for pair in duplicates.copy():
      similarity = similarity_qgram(pair[0]['featuresMap'],pair[1]['featuresMap'],pair[0]['title'],pair[1]['title'], q,gamma, alpha)
      if similarity < beta:
        duplicates.remove(pair)
        count+=1
    #print(duplicates)
    #now all these are removed, we cluster
  
    #count unique items still around
    unique_items = dict()
    count = 0
    for pair in duplicates:
        for i in pair:
            if i not in unique_items.values():
                unique_items.update({count: i})
                count+=1

    print("size dissim mat:")
    print(len(unique_items))          
    dist_mat = np.zeros((len(unique_items),len(unique_items)))
    for i in range(len(unique_items)):
      for j in range(len(unique_items)):
        if i<j:
            distance = 1-similarity_qgram(unique_items[i]['featuresMap'],unique_items[j]['featuresMap'],unique_items[i]['title'],unique_items[j]['title'], q,gamma, alpha)
            dist_mat[i,j] = distance
            dist_mat[j,i] = distance


    #np.savetxt("/Users/ronhochstenbach/Desktop/distmat.csv", dist_mat , delimiter=",")
    clustering = AgglomerativeClustering(n_clusters = None, distance_threshold = delta, affinity='precomputed', linkage = 'average').fit(dist_mat)
    
    labels = clustering.labels_
    assert len(labels) == len(unique_items)
  
    clusters = dict()
  
    for i in range(len(labels)):
        if labels[i] in clusters.keys():
            clusters[labels[i]] = np.append(clusters[labels[i]],unique_items[i])
        else: 
            clusters.update({labels[i]:[unique_items[i]]})

    final_predictions = []
    for group in clusters.values():
        if len(group)>1:
            for pair in combinations(group,2):
                final_predictions.append([pair[0],pair[1]])
                
    #count the number of correctly predicted duplicates
    D_f = 0

    for i in final_predictions:
        modelIDs_found = dict()
        for j in i:
          if j['modelID'] in modelIDs_found.keys():
            modelIDs_found[j['modelID']] += 1
          else:
            modelIDs_found.update({j['modelID']: 1})
      
        for num_found in modelIDs_found.values():
          if num_found > 1:
            D_f = D_f + comb(num_found,2)   #not sure if legit  
        modelIDs_found.clear()
  
    #count the number of duplicates present
    D_n = 0
    model_IDs_present = dict()
    for i in train_set.values():
        for j in i:
          if j['modelID'] in model_IDs_present:
            model_IDs_present[j['modelID']] +=1
          else:
            model_IDs_present.update({j['modelID']:1})

    for num_present in model_IDs_present.values():
        if num_present > 1:
          D_n = D_n + comb(num_present,2)

    #print("duplicates present: " + str(D_n))
    N_c_final = len(final_predictions)
    
    if not N_c_final == 0:
        Pair_Quality = D_f/N_c_final
    else: Pair_Quality = 0
        
    if not D_n ==0:
        Pair_Completeness = D_f/D_n
    else: Pair_Completeness = 0
        
    if not Pair_Quality+Pair_Completeness == 0:
        F1 = (2*Pair_Quality*Pair_Completeness)/(Pair_Quality+Pair_Completeness)
    else:
        F1=0
        
    if not comb(len(test_set.values()),2) == 0:
        frac_comp = N_c_LSH/comb(len(test_set.values()),2)
    else: frac_comp = 0

    return -1*F1


# Run Normal
# 

# In[129]:


data_cleaned = data_cleaner_extra(data)

hyper_params = {
    'alpha': 0.9,
    'beta': 0.5,
    'gamma': 0.85
}

indices = np.array([])
for i in range(hyper_parameters['number_of_items']):  
  indices = np.append(indices,i).astype(int)

PQ_output = []
PC_output = []
F1_output = []
FC_output = []

t = time.time()

for iteration in range(hyper_parameters['bootstraps']):

  t_iter = time.time()
  global train_representation_normal
  global candidate_pairs_normal
  global predicted_duplicates_normal


  np.random.seed(iteration)
  selected = np.random.choice(a=indices, replace = True, size =len(indices))
  selected = np.unique(selected)
  selected = np.sort(selected)

 
  train_set, test_set = train_test_split(data_cleaned, selected)

  #create modelwords from the train_set
  MW_title, MW_value = model_words(train_set)

  #adding representation to the train set
  train_representation_normal = add_representation_base(train_set, MW_title, MW_value)
  #perform LSH
  candidate_pairs_normal = LSH(train_representation_normal, np.concatenate((MW_title, MW_value)))
  N_c_LSH = len(candidate_pairs_normal)

  print("number of pairs after LSH: " + str(N_c_LSH))
  #do final predictions
  predicted_duplicates_normal = final_classification_qgram(candidate_pairs_normal, train_representation_normal,3,hyper_params['gamma'],hyper_params['beta'],hyper_params['alpha'])
  N_c_final = len(predicted_duplicates_normal)
  print("number of pairs final: " + str(N_c_final))


  #count the number of correctly predicted duplicates
  D_f = 0

  for i in predicted_duplicates_normal:
    modelIDs_found = dict()
    for j in i:
      if j['modelID'] in modelIDs_found.keys():
        modelIDs_found[j['modelID']] += 1
      else:
        modelIDs_found.update({j['modelID']: 1})
      
    for num_found in modelIDs_found.values():
      if num_found > 1:
        D_f = D_f + comb(num_found,2)   #not sure if legit  
    modelIDs_found.clear()
  

  #print("duplicates found: " + str(D_f))
  #count the number of duplicates present
  D_n = 0
  model_IDs_present = dict()
  for i in train_set.values():
    for j in i:
      if j['modelID'] in model_IDs_present:
        model_IDs_present[j['modelID']] +=1
      else:
        model_IDs_present.update({j['modelID']:1})
  
  for num_present in model_IDs_present.values():
    if num_present > 1:
      D_n = D_n + comb(num_present,2)
  
  #print("duplicates present: " + str(D_n))

  Pair_Quality = D_f/N_c_final
  Pair_Completeness = D_f/D_n
  F1 = (2*Pair_Quality*Pair_Completeness)/(Pair_Quality+Pair_Completeness)
  Fraction_Comparisons = len(candidate_pairs_normal)/comb(len(train_set.values()),2)
  time_iter = time.time()-t_iter

  PQ_output = np.append(PQ_output, Pair_Quality)
  PC_output = np.append(PC_output, Pair_Completeness)
  F1_output = np.append(F1_output, F1)
  FC_output = np.append(FC_output, Fraction_Comparisons)

  print("Pair Quality iteration " + str(iteration) + ": " + str(Pair_Quality))
  print("Pair Completeness iteration " + str(iteration)+ ": " + str(Pair_Completeness))
  print("F1 measure iteration "+ str(iteration) + ": " + str(F1))
  print("Fraction of Comparisons "+ str(iteration) + ": " + str(Fraction_Comparisons))
  print("Time elapsed this iteration: " + str(time_iter) + "seconds")

print("---------------    ITERATING FINISHED    ----------------")
print("Average Pair Quality: " + str(np.mean(PQ_output)))
print("Average Pair Completeness: " + str(np.mean(PC_output)))
print("Average F1 measure: " + str(np.mean(F1_output)))
print("Average Fraction of Comparisons: " + str(np.mean(FC_output)))
print("Total time elapsed: " + str(time.time()-t)+ "seconds") 


# Run Hyperopt

# In[344]:


#reload data when changing data cleaner type!!!
data_cleaned = data_cleaner_extra(data)


hyper_params_space = {
    'alpha': hp.uniform('alpha',0.8,1),
    'beta': hp.uniform('beta',0.2,0.5),
    'gamma': hp.uniform('gamma',0.6,0.9),
    'delta': hp.uniform('delta',0,1)
}

indices = np.array([])
for i in range(hyper_parameters['number_of_items']):  
  indices = np.append(indices,i).astype(int)

F1_output = []
opthyper_output = []

t = time.time()

for iteration in range(hyper_parameters['bootstraps']):
    t_iter = time.time()
    print("Starting iteration " + str(iteration))
    global train_representation_hyperopt
    global test_representation_hyperopt
    global candidate_pairs_after_LSH_hyperopt
    global predicted_duplicates_hyperopt
    
    np.random.seed(iteration)
    selected = np.random.choice(a=indices, replace = True, size =len(indices))
    selected = np.unique(selected)
    selected = np.sort(selected)
 
    train_set, test_set = train_test_split(data_cleaned, selected)
    print(len(train_set))
    #create modelwords from the train_set
    MW_title, MW_value = model_words(train_set)
    print(len(MW_title))
    print(len(MW_value))
    #adding representation to the train and test set
    train_representation_hyperopt = add_representation_base(train_set, MW_title, MW_value)
    #run train_set through LSH
    candidate_pairs_after_LSH_hyperopt = LSH(train_representation_hyperopt, np.concatenate((MW_title, MW_value)))
    print(len(candidate_pairs_after_LSH_hyperopt))
    print("LSH iteration " + str(iteration) + " done!")
    #run hyperparameter optimization using the train set
    trials = Trials()
    optimal_hyperparams = fmin(class_qgram_hyperopt, 
                              space=hyper_params_space, 
                              algo=tpe.suggest, 
                              max_evals=20,
                              trials=trials,
                              )
    print("Hyperopt iteration " + str(iteration) + " done!")
    test_representation_hyperopt = add_representation_base(test_set, MW_title, MW_value)
   
    #run LSH on test set
    test_candidates = LSH(test_representation_hyperopt, np.concatenate((MW_title, MW_value)))
    #final predictions on test set with optimal hyperparameters
    predicted_test_duplicates = final_classcluster(test_candidates, test_representation_hyperopt,3 ,optimal_hyperparams['alpha'],optimal_hyperparams['beta'],optimal_hyperparams['gamma'],optimal_hyperparams['delta'])
    N_c_final = len(predicted_test_duplicates)
    
    #count the number of correctly predicted duplicates
    D_f = 0
    for i in predicted_test_duplicates:
        modelIDs_found = dict()
        for j in i:
          if j['modelID'] in modelIDs_found.keys():
            modelIDs_found[j['modelID']] += 1
          else:
            modelIDs_found.update({j['modelID']: 1})
        for num_found in modelIDs_found.values():
          if num_found > 1:
            D_f = D_f + comb(num_found,2)   #not sure if legit
        modelIDs_found.clear()
  

    #print("duplicates found: " + str(D_f))
    #count the number of duplicates present
    D_n = 0
    model_IDs_present = dict()
    for i in test_set.values():
        for j in i:
          if j['modelID'] in model_IDs_present:
            model_IDs_present[j['modelID']] +=1
          else:
            model_IDs_present.update({j['modelID']:1})

    for num_present in model_IDs_present.values():
        if num_present > 1:
          D_n = D_n + comb(num_present,2)

      #print("duplicates present: " + str(D_n))

    Pair_Quality = D_f/N_c_final
    Pair_Completeness = D_f/D_n
    F1 = (2*Pair_Quality*Pair_Completeness)/(Pair_Quality+Pair_Completeness)

    F1_output = np.append(F1_output, F1)
    opthyper_output = np.append(opthyper_output, optimal_hyperparams)

    print("F1 iteration " + str(iteration) + ": " + str(F1))
    print("Optimal hyperparams " + str(iteration)+ ": " + str(optimal_hyperparams))
    print("Time elapsed this iteration: " + str(time.time()-t_iter) + "seconds")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

print("---------------    ITERATING FINISHED    ----------------")
best_iter = np.argmax(F1_output)
print("Best F1 achieved in iteration: " + str(best_iter))
print("Best F1 achieved: " + str(F1_output[best_iter]))
print("Optimal set of hyperparams: " + str(opthyper_output[best_iter]))
print("Time elapsed: " + str(time.time()-t)+ "seconds")


# Run graphs with clustering

# In[240]:


#reload data when changing data cleaner type!!!
data_cleaned = data_cleaner_extra(data)

hyper_params_space = {
    'alpha': hp.uniform('alpha',0.7,1),
    'beta': hp.uniform('beta',0.3,0.7),
    'gamma': hp.uniform('gamma',0.6,0.9),
    'delta': hp.uniform('delta',0,1)
}

indices = np.array([])
for i in range(hyper_parameters['number_of_items']):  
  indices = np.append(indices,i).astype(int)

try_bands = [1,10,20,40,50,70,85,100,120,140,175,199,80,90,110,130,75, 5, 30, 60, 150]

output = np.zeros((8,len(try_bands)))
# by row: bands, f1_pre, pq_pre, pc_pre, f1, pq, pc, frac_comp

band_iter = 0
for num_band in try_bands:
    
    hyper_parameters['bands'] = num_band
    assert hyper_parameters['bands'] == num_band
    print("Starting iteration for " + str(hyper_parameters['bands']) + " bands!")
    
    PQ_output = []
    PC_output = []
    F1_output = []
    FC_output = []
    PQ_pre_output = []
    PC_pre_output = []
    F1_pre_output = []
    
    t = time.time()
    for iteration in range(hyper_parameters['bootstraps']):
        print("Starting bootstrap " + str(iteration) + " with " +str(hyper_parameters['bands'])+ " bands.")
        t_iter = time.time()
        global train_representation_hyperopt
        global test_representation_hyperopt
        global candidate_pairs_after_LSH_hyperopt
        global predicted_duplicates_hyperopt

        selected = np.random.choice(a=indices, replace = True, size =len(indices))
        selected = np.unique(selected)
        selected = np.sort(selected)

        train_set, test_set = train_test_split(data_cleaned, selected)
        
        #create modelwords from the train_set
        MW_title, MW_value = model_words(train_set)

        #adding representation to the train and test set
        train_representation_hyperopt = add_representation_base(train_set, MW_title, MW_value)
        #run train_set through LSH
        candidate_pairs_after_LSH_hyperopt = LSH(train_representation_hyperopt, np.concatenate((MW_title, MW_value)))
        N_c_pre = len(candidate_pairs_after_LSH_hyperopt)
        
        #count the number of correctly predicted duplicates after pre clustering
        D_f_pre = 0
        for i in candidate_pairs_after_LSH_hyperopt:
            modelIDs_found = dict()
            for j in i:
              if j['modelID'] in modelIDs_found.keys():
                modelIDs_found[j['modelID']] += 1
              else:
                modelIDs_found.update({j['modelID']: 1})
            for num_found in modelIDs_found.values():
              if num_found > 1:
                D_f_pre = D_f_pre + comb(num_found,2)
            modelIDs_found.clear()
        
        #count the number of duplicates present
        D_n = 0
        model_IDs_present = dict()
        for i in test_set.values():
            for j in i:
              if j['modelID'] in model_IDs_present:
                model_IDs_present[j['modelID']] +=1
              else:
                model_IDs_present.update({j['modelID']:1})

        for num_present in model_IDs_present.values():
            if num_present > 1:
              D_n = D_n + comb(num_present,2)

        if not N_c_pre == 0:
            Pair_Quality_pre = D_f_pre/N_c_pre
        else: Pair_Quality_pre = 0
        
        if not D_n ==0:
            Pair_Completeness_pre = D_f_pre/D_n
        else: Pair_Completeness_pre = 0
        
        if not Pair_Quality_pre+Pair_Completeness_pre == 0:
            F1_pre = (2*Pair_Quality_pre*Pair_Completeness_pre)/(Pair_Quality_pre+Pair_Completeness_pre)
        else:
            F1_pre=0
            
        PQ_pre_output = np.append(PQ_pre_output, Pair_Quality_pre)
        PC_pre_output = np.append(PC_pre_output, Pair_Completeness_pre)
        F1_pre_output = np.append(F1_pre_output, F1_pre)
        
        #run hyperparameter optimization using the train set
        trials = Trials()
        optimal_hyperparams = fmin(class_qgram_hyperopt, 
                                  space=hyper_params_space, 
                                  algo=tpe.suggest, 
                                  max_evals=5,
                                  trials=trials,
                                  )
        test_representation_hyperopt = add_representation_base(test_set, MW_title, MW_value)

        #run LSH on test set
        test_candidates = LSH(test_representation_hyperopt, np.concatenate((MW_title, MW_value)))
        N_c_LSH = len(test_candidates)
        #final predictions on test set with optimal hyperparameters
        predicted_test_duplicates = final_classification_qgram(test_candidates, test_representation_hyperopt,3 ,optimal_hyperparams['alpha'],optimal_hyperparams['beta'],optimal_hyperparams['gamma'])
        N_c_final = len(predicted_test_duplicates)

        #count the number of correctly predicted duplicates
        D_f = 0
        for i in predicted_test_duplicates:
            modelIDs_found = dict()
            for j in i:
              if j['modelID'] in modelIDs_found.keys():
                modelIDs_found[j['modelID']] += 1
              else:
                modelIDs_found.update({j['modelID']: 1})
            for num_found in modelIDs_found.values():
              if num_found > 1:
                D_f = D_f + comb(num_found,2)   #not sure if legit
            modelIDs_found.clear()
        
        if not N_c_final == 0:
            Pair_Quality = D_f/N_c_final
        else: Pair_Quality = 0
        
        if not D_n ==0:
            Pair_Completeness = D_f/D_n
        else: Pair_Completeness = 0
        
        if not Pair_Quality+Pair_Completeness == 0:
            F1 = (2*Pair_Quality*Pair_Completeness)/(Pair_Quality+Pair_Completeness)
        else:
            F1=0
        
        if not comb(len(test_set.values()),2) == 0:
            frac_comp = N_c_LSH/comb(len(test_set.values()),2)
        else: frac_comp = 0
        
        PQ_output = np.append(PQ_output, Pair_Quality)
        PC_output = np.append(PC_output, Pair_Completeness)
        F1_output = np.append(F1_output, F1)
        FC_output = np.append(FC_output, frac_comp)
        
        

        print("~~~~~~~ Output iter " + str(iteration) + " with " + str(hyper_parameters['bands']) + " bands. ~~~~~~~~~~~")
        print("F1_pre: "+ str(F1_pre))
        print("PQ_pre: "+ str(Pair_Quality_pre))
        print("PC_pre: "+ str(Pair_Completeness_pre))
        print("F1: "+ str(F1))
        print("PQ: "+ str(Pair_Quality))
        print("PC: "+ str(Pair_Completeness))
        print("Frac. Comp. : "+ str(frac_comp))
        print("Time elapsed this iteration: " + str(time.time()-t_iter) + "seconds")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print("---------------    ITERATING FINISHED FOR " + str(hyper_parameters['bands']) + " BANDS    ----------------")
    
    print("Average Pre Pair Quality: " + str(np.mean(PQ_pre_output)))
    print("Average Pre Pair Completeness: " + str(np.mean(PC_pre_output)))
    print("Average Pre F1 measure: " + str(np.mean(F1_pre_output)))
    
    
    print("Average Pair Quality: " + str(np.mean(PQ_output)))
    print("Average Pair Completeness: " + str(np.mean(PC_output)))
    print("Average F1 measure: " + str(np.mean(F1_output)))
    print("Average Fraction of Comparisons: " + str(np.mean(FC_output)))
    print("Total time elapsed: " + str(time.time()-t)+ "seconds") 
    
    output[0,band_iter] = hyper_parameters['bands']
    
    output[1,band_iter] = np.mean(F1_pre_output)
    output[2,band_iter] = np.mean(PQ_pre_output)
    output[3,band_iter] = np.mean(PC_pre_output)
    
    output[4,band_iter] = np.mean(F1_output)
    output[5,band_iter] = np.mean(PQ_output)
    output[6,band_iter] = np.mean(PC_output)
    output[7,band_iter] = np.mean(FC_output)
    
    band_iter+=1
    #np.savetxt("/Users/ronhochstenbach/Desktop/ResultsFull.csv", output, delimiter=",")

print("All done! Hooray!")
    


# Make graphs

# In[349]:


results = pd.read_csv('/Users/ronhochstenbach/Desktop/ResultsFull.csv', sep = ',', header = None).to_numpy()
#1: F1_pre, 2: PQ_pre, 3: PC_pre, 4: F1, 5: PQ, 6: PC, 7: FC
print(results[7,5])
x = results[3,0:11]
y = results[1,0:11]
plt.plot(x, y)

plt.xlabel("Fraction of Comparisons")
plt.ylabel("Pair Completeness after LSH")


# In[ ]:




