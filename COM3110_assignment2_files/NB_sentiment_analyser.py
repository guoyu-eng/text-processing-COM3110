# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse
import sys, re, getopt, csv, nltk
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import textblob            #to import
from textblob import TextBlob
"""
IMPORTANT, modify this part with your details
"""
USER_ID = "aca20gl" #your unique student ID, i.e. the IDs starting with "acp", "mm" etc that you use to login into MUSE 
nltk.download('stopwords')

nltk.download('punkt')
def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args



def preprocessing_real(data):
    df_training = pd.read_csv(data,sep='\t')
    df_training['Phrase'] = df_training['Phrase'].str.lower()

    return df_training


def preprocessing(data,number_class):
    df_training = pd.read_csv(data,sep='\t')
    df_training['Phrase'] = df_training['Phrase'].str.lower()
        
        
    if number_class == 3 :
            
        df_training['Sentiment'].replace(4,2,inplace = True)   
        df_training['Sentiment'].replace(1,0,inplace =True)
        df_training['Sentiment'].replace(2,1,inplace =True)
        df_training['Sentiment'].replace(3,2,inplace =True)
            # df_training = df_training['Sentiment'].replace(4,2,inplace =True)
        return df_training
        
    else:
        return df_training
    
    # 
def move_comma_s(text,number_class,features):
    ''' move the Phrase comma '''
    if features == 'all_words':
        
        move = preprocessing(text,number_class)
        stop = stopwords.words('english')
        stemmer = nltk.SnowballStemmer("english")

        move['Phrase']  =  move['Phrase'].apply(lambda x : re.sub(r'[\.;:,\?\"\'\/]','',x))
        move['Phrase'] = move['Phrase'].apply(lambda x:' '.join([word for word in x.split() if word not in (stop)]))
  
        move['Phrase'] = move['Phrase'].apply(stemmer.stem)
  
        return move
    elif features == 'features':
         
        move = preprocessing(text,number_class)
        stop = stopwords.words('english')
        # move['Phrase']  =  move['Phrase'].str.replace("'s", "is")
        move['Phrase']  =  move['Phrase'].apply(lambda x : re.sub(r'[\.;:,\?\"\'\/]','',x))
        move['Phrase'] = move['Phrase'].apply(lambda x:' '.join([word for word in x.split() if word not in (stop)]))
     
        # ['Phrase'] = move['Phrase'].apply(lambda stop_remove: [word.lower() for word in stop_remove.split() if word not in stopwords])
        return move
        

   
def process_test(text,number_class):
    df_training = pd.read_csv(text,sep='\t')
    # df_training['Phrase'] = df_training['Phrase'].str.lower()
    
    # df_training['Phrase']  =  df_training['Phrase'].str.replace("'s", "is")
    df_training['Phrase']  =  df_training['Phrase'].apply(lambda x : re.sub(r'[\.;:,\?\"\'\/]','',x))
    return df_training
        
        
def prio_probability(text,number_class,features):
    data1 = dict()
       
    n = 0
    data = move_comma_s(text,number_class,features)
    # return the prior probability of 3 class
    a = data['Sentiment'].value_counts(normalize = True).to_frame()
    # return the frequency for all of class 
    # a = data['Sentiment'].value_counts().to_frame()
    for b in a['Sentiment']:
        data1[n] = b
        n = n+1
    return data1

   

 
def likelihood_molecular (text, num_cla,features):
    word_data_0 = dict()
    word_data_1 = dict()
    word_data_2 = dict()
    
    data = move_comma_s(text, num_cla,features)
    if num_cla == 3:   
        data_0 = data[data['Sentiment'] == 0]
        data_1 = data[data['Sentiment'] == 1]
        data_2 = data[data['Sentiment'] == 2]
        # get the likelihood for all word, 
        out_data_0 = data_0['Phrase'].tolist()
        out_data_1 = data_1['Phrase'].tolist()
        out_data_2 = data_2['Phrase'].tolist()
        
        for ser_word in out_data_0:
            ser_word_1 = ser_word.split()
            for word in ser_word_1:
                if word in word_data_0:
                    word_data_0[word] = word_data_0[word]+1   
                else:
                    word_data_0[word] = 1       
        
        for ser_word in out_data_1:
            ser_word_1 = ser_word.split()
            for word in ser_word_1:
                if word in word_data_1:
                    word_data_1[word] = word_data_1[word]+1 
                else:
                    word_data_1[word] = 1 
        
        for ser_word in out_data_2:
            ser_word_1 = ser_word.split()
            for word in ser_word_1:
                if word in word_data_2:
                    word_data_2[word] = word_data_2[word]+1             
                else:
                    word_data_2[word] = 1 
                    
        return word_data_0,word_data_1,word_data_2
    else:  
        word_data_3 = dict()
        word_data_4 = dict()
        data_0 = data[data['Sentiment'] == 0]
        data_1 = data[data['Sentiment'] == 1]
        data_2 = data[data['Sentiment'] == 2]
        data_3 = data[data['Sentiment'] == 3]
        data_4 = data[data['Sentiment'] == 4]
        # get the likelihood for all word, 
        out_data_0 = data_0['Phrase'].tolist()
        for ser_word in out_data_0:
            ser_word_1 = ser_word.split()
            for word in ser_word_1:
                if word in word_data_0:
                    word_data_0[word] = word_data_0[word]+1   
                else:
                        word_data_0[word] = 1       
        out_data_1 = data_1['Phrase'].tolist()
        for ser_word in out_data_1:
            ser_word_1 = ser_word.split()
            for word in ser_word_1:
                if word in word_data_1:
                    word_data_1[word] = word_data_1[word]+1 
                else:
                    word_data_1[word] = 1 
        out_data_2 = data_2['Phrase'].tolist()
        for ser_word in out_data_2:
            ser_word_1 = ser_word.split()
            for word in ser_word_1:
                if word in word_data_2:
                    word_data_2[word] = word_data_2[word]+1             
                else:
                    word_data_2[word] = 1 
        out_data_3 = data_3['Phrase'].tolist()
        for ser_word in out_data_3:
            ser_word_1 = ser_word.split()
            for word in ser_word_1:
                if word in word_data_3:
                    word_data_3[word] = word_data_3[word]+1             
                else:
                    word_data_3[word] = 1 
        out_data_4 = data_4['Phrase'].tolist()
        for ser_word in out_data_4:
            ser_word_1 = ser_word.split()
            for word in ser_word_1:
                if word in word_data_4:
                    word_data_4[word] = word_data_4[word]+1             
                else:
                    word_data_4[word] = 1                                   
        return word_data_0,word_data_1,word_data_2,word_data_3,word_data_4
    # 分母出现特征词总数和除去重复的特征值
def likelihood_denominator(text,num_cla,features):
    number = dict()
    data = move_comma_s(text, num_cla,features)
    data_totle = data['Phrase'].tolist()
    for ser_word in data_totle:
        ser_word_1 = ser_word.split()
        for word in ser_word_1:
            if word in number:
                number[word] = number[word]+1   
            else:
                number[word] = 1 
    if num_cla ==3:
      
        data_0,data_1,data_2 = likelihood_molecular(text,num_cla,features)
        # train posive and negative and nutral total keys number
        # feature_total_number = len(data_0.keys()) +len(data_1.keys())+len(data_2.keys())
        feature_total_number = len(number.keys())
        # positve feature total words
        total_0 = sum(data_0.values())+feature_total_number
        total_1 = sum(data_1.values())+feature_total_number
        total_2 = sum(data_2.values())+feature_total_number
            
        return total_0,total_1,total_2
    else:
            data_0,data_1,data_2,data_3,data_4 = likelihood_molecular(text,num_cla,features)
            # train posive and negative and nutral total keys number
            # feature_total_number = len(data_0.keys()) +len(data_1.keys())+len(data_2.keys())+len(data_3.keys())+len(data_4.keys())
            
            feature_total_number = len(number.keys())
           
            total_0 = sum(data_0.values())+feature_total_number
            total_1 = sum(data_1.values())+feature_total_number
            total_2 = sum(data_2.values())+feature_total_number
            total_3 = sum(data_3.values())+feature_total_number
            total_4 = sum(data_4.values())+feature_total_number
        
        
            return total_0,total_1,total_2,total_3,total_4


def likelihood_claculate (word,text,num_cla,features):
    posi = dict()
    neutr = dict()
    negat = dict()
    s_po = dict()
    s_neg = dict()
    if num_cla == 3: 
        
        
        nega,neu,po = likelihood_molecular(text,num_cla,features)
        # prio_data = prio_probability(text,num_cla)
        nega_down,neu_down,po_down =likelihood_denominator(text,num_cla,features)
        
        if word in po:  
            posi[word] = 2   
        else:
            posi[word] = 1
        
        
        if word in neu:  
            neutr[word] = 2
        else:
            neutr[word] = 1
            
        
        if word in nega:  
            negat[word] = 2 
        else:
            negat[word] = 1
            
            
        number2 = posi[word]
        number1 = neutr[word]
        number0 = negat[word]
    
        return number0,number1,number2
    else:
        nega,some_nega,neu,some_posi,po = likelihood_molecular(text,num_cla,features)
        # prio_data = prio_probability(text,num_cla)
        nega_down,some_nega_down,neu_down,some_po_down,po_down= likelihood_denominator(text,num_cla,features)
        
        if word in po:  
            posi[word] = 2   
        else:
            posi[word] = 1
        
        
        if word in neu:  
            neutr[word] = 2
        else:
            neutr[word] = 1
            
        
        if word in nega:  
            negat[word] = 2 
        else:
            negat[word] = 1
            
        if word in some_nega:
            s_neg[word] = 2
        else:
            s_neg[word] =1
        
        if word in some_posi:
            s_po[word] = 2
        else:
            s_po[word] =1
        result_4 = posi[word] 
        result_3 = s_po[word]
        result_2 = neutr[word]
        result_1 = s_neg[word]
        result_0 = negat[word]
            
        

        return result_0,result_1,result_2,result_3,result_4
    # the final naive Bayes classifier
    # def posteriors(test,train,num_cla,output_files):
def posteriors(test,train,num_cla,features,confusion_matrix,output_files):
    # feature_0,feature_1,feature_2 = likelihood_molecular(train,num_cla)
    test_data1 = process_test(test,num_cla)
    test_data = test_data1['Phrase'].tolist() 
    test_data2 = test_data1.insert(loc = 2 , column = 'Sentiment1',value = 0)
    row_number = len(test_data1['SentenceId'])
    large = dict()
    if num_cla == 3:
        nega_down,neu_down,po_down =likelihood_denominator(train,num_cla,features)

        prior = prio_probability(train,num_cla,features) 
     
        n = 0       
        nega,neu,po = likelihood_molecular(train,num_cla,features)
        for n, key in enumerate(test_data) :
            key_1 = key.split()
            positive = 1
            negative = 1
            neutral = 1 
          
            for key_2 in key_1: 
                if key_2 not in po:
                    positive = 1/po_down
                else:
                    positive = positive*(2/po_down)
                    
                if key_2 not in neu:
         
                    neutral = 1/neu_down
                else:
                    neutral = neutral*(2/neu_down)
                    
                if key_2 not in nega:
                    negative = 1/nega_down
                else:
                    negative = negative*(2/nega_down)

           
            large['2'] = positive*prior[2]
            large['1'] = neutral*prior[1]
            large['0'] = negative*prior[0]
            
            max_number = max(large.values())
            
            # result = max(large.items(),key = lambda x:x[1])
  
            for key,value in large.items():
                if value == max_number:
                    test_data1.loc[n,'Sentiment1'] = key
                    test_data1.loc[n,'Sentiment1'] = key
        if output_files:
            
            # replace the txt file name in there
            with open("text.txt",mode = 'a',encoding='utf-8') as ff:
                sys.stdout = open("text.txt","wt")
                aa = test_data1[["SentenceId","Sentiment1"]]
                print(aa.to_string())
                return aa
        else:
        
            return test_data1[["SentenceId","Sentiment1"]]
    # test_data1[["SentenceId","Sentiment"]]       
    else:
       
        down_0,down_1,down_2,down_3,down_4 =likelihood_denominator(train,num_cla,features)
        large_one = dict()
        
        # some_posi = ('some_positive')
        # neud = ("nertral")
        # some_nega = ('some_negative')
        # nega = ("negative")
            
        prior = prio_probability(train,num_cla,features)  
        nega,some_nega,neu,some_posi,po = likelihood_molecular(train,num_cla,features)
        for n, key in enumerate(test_data) :
            positive = 1
            somewhat_positive = 1
            negative = 1
            somewhat_negative = 1
            neutral = 1 
            key_1 = key.split()
            for key_2 in key_1: 
                if key_2 not in po:
                    positive = 1/down_4
                else:
                    positive = positive*(2/down_4)
                    
                if key_2 not in some_posi:
                    somewhat_positive = 1/down_3
                else:
                    somewhat_positive = somewhat_positive*(2/down_3)
                    
                if key_2 not in neu:
         
                    neutral = 1/down_2
                else:
                    neutral = neutral*(2/down_2)
                
                if key_2 not in some_nega:
         
                    somewhat_negative = 1/down_1
                else:
                    somewhat_negative = somewhat_negative*(2/down_1)
                
                
                    
                if key_2 not in nega:
                    negative = 1/down_0
                else:
                    negative = negative*(2/down_0) 
                
    
            # post_positive = positive*prior[4]
            # post_some_positive = somewhat_positive*prior[3]
            # post_neutral = neutral*prior[2]
            # post_some_negative = somewhat_negative*prior[1]
            # post_negative = negative*prior[0]
            
            
            large_one['4'] = positive*prior[4]
            large_one['3'] = somewhat_positive*prior[3]
            large_one['2'] = neutral*prior[2]
            large_one['1'] = somewhat_negative*prior[1]
            large_one['0'] = negative*prior[0]
            
            max_number = max(large_one.values())
            for key,value in large_one.items():
                if value == max_number:
                    test_data1.loc[n,'Sentiment1'] = key
              
   
    
      
        if output_files:
            # replace the txt file name in there
            with open("text.txt",mode = 'a',encoding='utf-8') as ff:
                sys.stdout = open("text.txt","wt")
                aa = test_data1
                print(aa.to_string())
                return aa
        else:
  
            return test_data1[["SentenceId","Sentiment1"]]
    # test_data1[["SentenceId","Sentiment1"]]
def f1(dev,train,classes,features,confusion_matrix,output_files):
    real_set = preprocessing_real(dev)
    # aa = dev['Sentiment']
    
    act_positive = real_set[real_set['Sentiment'] == 4]
    act_positive_1 = real_set[real_set['Sentiment'] == 3]
    act_neu = real_set[real_set['Sentiment'] == 2]
    act_nega_1 =real_set[real_set['Sentiment'] == 1]
    act_nega = real_set[real_set['Sentiment'] == 0]
    # 得到dev中positive neutral 和 negative的文件序列号，放在一个list。
    act_positive = act_positive['SentenceId'].tolist()
    act_positive_1 = act_positive_1['SentenceId'].tolist()
    act_neu = act_neu['SentenceId'].tolist()        
    act_nega_1 = act_nega_1['SentenceId'].tolist()
    act_nega = act_nega['SentenceId'].tolist()
    if classes == 3:
        sentiments = ["positive","negative","neutral"]
        tp = list()
        fp = list()
        fn = list()
        tn = list()
        tp_1 = list()
        fp_1 = list()
        fn_1 = list()
        number_1 = 0
        number_2 = 0
        number_3 = 0
        number_4 = 0

        
        predicte = posteriors(dev,train,classes,features,confusion_matrix,output_files)
      
        predicte_pos = predicte[predicte['Sentiment1'] == '2']
        predicte_neu = predicte[predicte['Sentiment1'] == '1']
        predicte_nega= predicte[predicte['Sentiment1'] == '0']
        
        predicte_pos_id = predicte_pos['SentenceId'].tolist()
        predicte_neu_id = predicte_neu['SentenceId'].tolist()
        predicte_nega_id = predicte_nega['SentenceId'].tolist()
        
        
        for i in predicte_pos_id:
            if i in act_positive or act_positive_1:
                tp.append(i)
                
        for i in predicte_nega_id:
            if i in act_positive or act_positive_1:
                fn.append(i)
        
        for i in predicte_pos_id:
            if i in act_nega_1 or act_nega :
                fp.append(i)
       
        for i in predicte_nega_id:
            if i in act_nega_1 or act_nega:
                tn.append(i)
        
        
        for i in predicte_neu_id:
            if i in act_neu:
                tp_1.append(i)

        for i in predicte_neu_id:
            if i in act_positive or act_positive_1:
                number_1 = number_1 + 1
        
        for i in predicte_neu_id:
            if i in act_nega_1 or act_nega:
                number_2 = number_2 + 1
                
                
        for i in predicte_pos_id:
            if i in act_neu:
                number_3 = number_3 + 1
        
        for i in predicte_nega_id:
            if i in act_neu:
                number_4 = number_4 + 1

        
        nd_1 = np.matrix([[len(tp),len(fn),(number_1)],
                          [len(fp),len(tn),number_2],
                          [number_3,number_4,len(tp_1)]])
       
        sum_total = len(tp)+len(fn)+(number_1)+len(fp)+len(tn)+number_2+number_3+number_4+len(tp_1)
        accu_positive = (len(tp)+len(tn)+len(tp_1) )/sum_total 
        accu_2 = (len(tp_1)+len(tn) )/sum_total 
        
        f1_positive = 2*len(tp)/(2*len(tp)+len(fp)+len(fn))
        f1_nagetive = 2*len(tn)/(2*len(tn)+len(fp)+len(fn))
        f1_neutral = 2*len(tp_1)/(2*len(tp_1)+ number_1+number_2+number_3+number_4)
        
        final_result = (f1_positive+f1_nagetive+f1_neutral)/classes     
        accuracy = accu_positive
  
        
        if confusion_matrix:
            
            plot_confusion_matrix(cm           = nd_1, 
                                  normalize    = False,
                                  target_names = sentiments,
                                  title        = "Confusion Matrix")
           
            
            return final_result
        else:
           
            print(accuracy)
            return final_result
            
    
    
    else:
        sentiments = ["positive","some_potive","neutral","some_negative","negative"]
        
        predicte = posteriors(dev,train,classes,features,confusion_matrix,output_files)
        
        predicte_pos = predicte[predicte['Sentiment1'] == '4']
        predicte_neu = predicte[predicte['Sentiment1'] == '2']
        predicte_nega= predicte[predicte['Sentiment1'] == '0']
        
        predicte_some_pos = predicte[predicte['Sentiment1'] == '3']
        predicte_some_nega = predicte[predicte['Sentiment1'] == '1']
        
        predicte_pos_id = predicte_pos['SentenceId'].tolist()
        predicte_neu_id = predicte_neu['SentenceId'].tolist()
        predicte_nega_id = predicte_nega['SentenceId'].tolist()
        
        predicte_some_pos_id = predicte_some_pos['SentenceId'].tolist()
        predicte_some_nega_id = predicte_some_nega['SentenceId'].tolist()

        number_1_1 = 0
        number_2_1 = 0
        number_3_1 = 0
        number_4_1 = 0
        number_5_1 = 0
        for i in predicte_pos_id:
            if i in act_positive:
                number_1_1 +=1
            else:  
                if i in act_positive_1:
                    number_2_1 +=1
                else:
                    if i in act_neu:
                        number_3_1 +=1
                    else:
                        
                        if i in act_nega_1:
                            number_4_1 +=1
                        else:
                            number_5_1 +=1
        
                    
        number_1_2 = len([x for x in predicte_some_pos_id if x in act_positive ])
        number_2_2 = len([x for x in predicte_some_pos_id if x in act_positive_1 ])
        number_3_2 = len([x for x in predicte_some_pos_id if x in act_neu ])
        number_4_2 = len([x for x in predicte_some_pos_id if x in act_nega_1 ])
        number_5_2 = len([x for x in predicte_some_pos_id if x in act_nega ])
        
        number_1_3 = len([x for x in predicte_neu_id if x in act_positive ])
        number_2_3 = len([x for x in predicte_neu_id if x in act_positive_1 ])
        number_3_3 = len([x for x in predicte_neu_id if x in act_neu ])
        number_4_3 = len([x for x in predicte_neu_id if x in act_nega_1 ])
        number_5_3 = len([x for x in predicte_neu_id if x in act_nega ])
        
        number_1_4 = len([x for x in predicte_some_nega_id if x in act_positive ])
        number_2_4 = len([x for x in predicte_some_nega_id if x in act_positive_1 ])
        number_3_4 = len([x for x in predicte_some_nega_id if x in act_neu ])
        number_4_4 = len([x for x in predicte_some_nega_id if x in act_nega_1 ])
        number_5_4 = len([x for x in predicte_some_nega_id if x in act_nega ])
        
        number_1_5 = len([x for x in predicte_nega_id if x in act_positive ])
        number_2_5 = len([x for x in predicte_nega_id if x in act_positive_1 ])
        number_3_5= len([x for x in predicte_nega_id if x in act_neu ])
        number_4_5 = len([x for x in predicte_nega_id if x in act_nega_1 ])
        number_5_5 = len([x for x in predicte_nega_id if x in act_nega ])
               
        nd_1 = np.matrix([[number_1_1,number_1_2,number_1_3,number_1_4,number_1_5],
                          [number_2_1,number_2_2,number_2_3,number_2_4,number_2_5],
                          [number_3_1,number_3_2,number_3_3,number_3_4,number_3_5],
                          [number_4_1,number_4_2,number_4_3,number_4_4,number_4_5],
                          [number_5_1,number_5_2,number_5_3,number_5_4,number_5_5]])        
                
                
        for_pos = (2 *number_1_1 )/(2*number_1_1+number_2_1+number_3_1+number_4_1+number_5_1+number_1_2+number_1_3+number_1_4+number_1_5)
        for_some_pos = (2*number_2_2)/(2*number_2_2+number_2_1+number_2_3+number_2_4+number_2_5+number_1_2+number_3_2+number_4_2+number_5_2)
        for_neu = (2*number_3_3)/(2*number_3_3+number_3_1+number_3_2+number_3_4+number_3_5+number_2_3+number_1_3+number_4_3+number_5_3)
        for_some_nega = (2*number_4_4)/(2*number_4_4+number_4_1+number_4_2+number_4_3+number_4_5+number_1_4+number_2_4+number_3_4+number_5_4)
        for_nega = (2*number_5_5)/(2*number_5_5+number_5_1+number_5_2+number_5_3+number_5_4+number_1_5+number_2_5+number_3_5+number_4_5)
        result1 = (for_pos+for_some_pos+for_neu+for_some_nega+for_nega)/classes
        
        
        true_value = number_1_1 +number_2_2+number_3_3+number_4_4+number_5_5
        
       
        
        
        accuracy = true_value/(true_value+number_1_2+number_1_3+number_1_4+number_1_5+number_2_1+number_2_3+number_2_4+number_2_5+
                               number_3_1+number_3_2+number_3_4+number_3_5+
                               number_4_1+number_4_2+number_4_3+number_4_5+
                               number_5_1+number_5_2+number_5_3+number_5_4)
        if confusion_matrix :    
            plot_confusion_matrix(cm           = nd_1, 
                                  normalize    = False,
                                  target_names = sentiments,
                                  title        = "Confusion Matrix")
            
          
  
            return result1
        else:
           
            print(accuracy)
            return result1
            
    
        
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    
   

    if normalize:
        cm = cm.astype('float') / cm.sum()

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.grid(False)
    plt.tight_layout()
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')
    plt.show()

# Gradable method

# This threshold is used to distinguish between negative / neutral / positive tweets
# negative < -threshold <= neutral <= +threshold < positive

    
def main():
    
    inputs=parse_args()
    
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    
    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
     
    
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix
    
    """
    ADD YOUR CODE HERE
    Create functions and classes, using the best practices of Software Engineering
    """

    #You need to change this in order to return your macro-F1 score for the dev set
    f1_score = f1(dev,training,number_classes,features,confusion_matrix,output_files)
 
    
    # use to get the predict sentiment 
    # print(posteriors(test,training,number_classes,features,confusion_matrix,output_files))

    """
    IMPORTANT: your code should return the lines below. 
    However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
    """
    #print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()