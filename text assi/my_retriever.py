
import math
class Retrieve:
    
    # Create new Retrieve object storing index and term weighting
    # scheme. (You can extend this method, as required.)
    def __init__(self, index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        
        
    def compute_number_of_documents(self):
        self.doc_ids = set() 
        for term in self.index:
            self.doc_ids.update(self.index[term])
        
        return len(self.doc_ids)

    def doc_vec_calculation(self):
        # return dict -> [all document id ,sqrt(sum of squares of all word frency ),
       self.doc_freq_sum = dict()
       for term, doc_fre in self.index.items(): 
           for docid, freq in doc_fre.items():
               if self.term_weighting in ['tf', 'tfidf']:
                   if docid in self.doc_freq_sum.keys():
                       self.doc_freq_sum[docid] += freq * freq
                   else:
                       self.doc_freq_sum[docid] = freq *freq
               else:
                   #for Binary method
                   if docid in self.doc_freq_sum.keys():
                       self.doc_freq_sum[docid] += 1
                   else:
                       self.doc_freq_sum[docid] = 1
       # vector size(Denominator in cos function):
       for docid, sq_sum in self.doc_freq_sum.items():
           self.doc_freq_sum[docid] = math.sqrt(sq_sum)
       return self.doc_freq_sum  

    def idf_total(self,query):   
        # return dict-> docu_id,tfidf
        self.docu_fre_total = dict()
        a = self.index 
        for term, doc_fre in a.items(): 
            for docid, freq in doc_fre.items():
  
                if docid in self.docu_fre_total.keys():                              
                    idf_number = math.log(self.num_docs/len(doc_fre.items())) 
                    self.docu_fre_total[docid] += math.pow((freq*idf_number),2)
                else:                   
                    idf_number = math.log(self.num_docs/len(doc_fre.items()) )                       
                    self.docu_fre_total[docid] = math.pow((freq*idf_number),2)

        for docid, sq_sum in self.docu_fre_total.items():
            self.docu_fre_total[docid] = math.sqrt(sq_sum)
        return self.docu_fre_total 

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):

                              
        coop =  self.choose_way(query)  
        return coop           
                
        # return list(range(1,11))
    def choose_way (self,query):
        if self.term_weighting in ['tf' ]:
            self.way1 = self.tf(query)
            return self.way1[-10:]
        if self.term_weighting in [ 'tfidf']:
            self.way2 = self.tf_idf(query)
            return self.way2[-10:]
        if self.term_weighting in [ 'binary']:
            self.way3= self.binary(query)
            return self.way3[-10:]
   
            
    def rela_doc(self,query):
        # simply the total document ，get the docment about query
        self.a= dict()
        ll = self.deal_query(query)
        for key1,value1 in ll.items() :
            if key1 in self.index.keys():
                self.a.update({key1:self.index[key1]})
        return self.a
      
        
    def binary (self,query):
        # run the binary model
        bin_dic = dict()
        rela_do = self.rela_doc(query) 
        result_bin = dict()
        # molucular
        for item in query:
            for item_doc,value_doc in rela_do.items():
                if item_doc == item :
                    for doc_id,doc_vale in value_doc.items():
                        if doc_id in bin_dic.keys():
                            bin_dic[doc_id] = bin_dic[doc_id]+1
                        else:
                            bin_dic[doc_id] = 1 
        b = self.doc_vec_calculation() 
        for bin_id ,bin_val in bin_dic.items():
            a = len(str(b[bin_id]))
            result = math.sqrt(bin_val/a)
            result_bin[bin_id] = result
            
        self.sort_dic2 = sorted(result_bin.items(),key= lambda d:d[1],reverse = True)
            
        return [k[0] for k in self.sort_dic2[:10]]
                        
    def tf_idf (self,query):
        # calculate tf_idf
        self.docu_idf = dict()
        self.result_tfidf = []
        rela_document_fre = self.rela_doc(query) 
        word1 = dict()
        a = self.idf_total(query)
        query1 = self.deal_query(query)
        self.order1 = dict()
        # get the all of word idf, return dict -> word,idf
        
        for doc_id, doc_val2 in self.index.items():
            self.docu_idf[doc_id] = math.log(self.num_docs/len(doc_val2.items())) 


        for re_id,re_val in rela_document_fre.items():          
            for re2_id , re2_val in re_val.items():
                if re2_id in word1.keys():

                    molecular = word1[re2_id] + query1[re_id]*self.docu_idf[re_id] *re_val[re2_id]
                    word1[re2_id] = molecular 
                else:
                    word1[re2_id] = query1[re_id] * self.docu_idf[re_id] * re_val[re2_id]
              
        for wk,wv in word1.items():
            for id_1 , val_1 in a.items():
                if wk == id_1 :
                    self.order1[wk]= wv/val_1
                    
        self.sort_dic1 = sorted(self.order1.items(),key= lambda d:d[1],reverse = True)
            
        return [k[0] for k in self.sort_dic1[:10]]
                              
    def tf (self,query):
        # run the TF model
        rela_document_fre = self.rela_doc(query) 
        self.tf_leng = dict()
        self.word = dict()
        self.result1 =[]
        self.order = dict()
        a = self.deal_query(query)
        for akey,avalue in a.items():            
            for key1, value1 in rela_document_fre.items():
                if akey == key1:
                    for keys2,value2 in value1.items():
                        
                        if keys2 in self.word.keys():
                            new_value = self.word[keys2]+ avalue*value2
                            self.word[keys2] = new_value

                        else:
                            self.word[keys2] = avalue*value2

        b = self.doc_vec_calculation()
        for key_word, value_word in self.word.items():
            for key_tf ,value_tf in b.items():
                if key_word == key_tf: 
                    s= b[key_tf]
                    result_number = value_word/s
                    self.order[key_word] = result_number
        
        self.sort_dic = sorted(self.order.items(),key= lambda d:d[1],reverse = True)
            
        return [k[0] for k in self.sort_dic[:10]]

    def deal_query(self,query):
        # combine the same word in the query return dict-> (word,frequency)
        self.dict = {}
        for key in query:
            self.dict[key] = self.dict.get(key,0)+1
        return self.dict

    

   