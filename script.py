import sys
import processing 
from collections import Counter
import numpy as np
from nltk.tokenize import word_tokenize

class Algorithm:
    def __init__(self, processed_data, text_files):
        self.num_documents = len(processed_data)
        self.processed_data = processed_data
        
        if len(self.processed_data) != len(text_files):
            print("[?] Error in lengths of text files and processed text data")
            return

        self.document_indexing = {}
        for i in range(self.num_documents):
            self.document_indexing[i] = text_files[i]

    def document_frequency(self):
        self.df = {}
        for i in range(self.num_documents):
            tokens = self.processed_data[i]

            for tok in tokens:
                if tok not in self.df:
                    self.df[tok] = set()
                    self.df[tok].add(i)
                else:
                    self.df[tok].add(i)

        for k in self.df:
            self.df[k] = (len(self.df[k]), self.df[k])
        
        self.vocab_size = len(self.df)
        self.vocab = list(self.df)

    def inverse_document_frequency(self):
        self.tf_idf = {}
        
        for i in range(self.num_documents):
            tokens = self.processed_data[i]

            counter = Counter(tokens)
            words_count = len(tokens)

            for tok in list(set(tokens)):
                tf = counter[tok] / words_count

                df = self.df[tok][0] if tok in self.df else 0 
                idf = np.log((self.num_documents)/(df+1))

                self.tf_idf[i, tok] = tf * idf

    def vectorizing_tf_idf_model(self):
        self.doc_vectors = np.zeros((self.num_documents, self.vocab_size))

        for doc_idx, word in self.tf_idf:
            try:
                idx = self.vocab.index(word)
                self.doc_vectors[doc_idx][idx] = self.tf_idf[(doc_idx, word)]
            except:
                pass

    def generate_vectors(self, tokens):
        vectors = np.zeros((len(self.vocab)))

        counter = Counter(tokens)
        words_count = len(tokens)

        for tok in list(set(tokens)):
            tf = counter[tok] / words_count

            df = self.df[tok][0] if tok in self.df else 0 
            idf = np.log((self.num_documents)/(df+1))

            try:
                idx = self.vocab.index(tok)
                vectors[idx] = tf*idf 
            except:
                pass
        return vectors

    @staticmethod
    def cosine_metric(x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    def cosine_similarity(self, query, k):
        """@params:
        query: is the input query
        k: number of relevant documents
        """

        clean = processing.TextClean()
        tokens = clean.cleanse_data(query)

        query_vec = self.generate_vectors(tokens)

        doc_cosines = []
        for doc_vec in self.doc_vectors:
            doc_cosines.append(Algorithm.cosine_metric(query_vec, doc_vec))

        k_docs = np.array(doc_cosines).argsort()[-k:][::-1]
        return k_docs

    def get_doc_names(self, doc_idxs):
        return [self.document_indexing[i] for i in doc_idxs]

if __name__ == '__main__':
    dir_name = str(sys.argv[1])
    query = str(sys.argv[2])
    k = int(sys.argv[3])
    
    p = processing.Processing()

    text_files = p.get_all_text_files(dir_name)
    print("[*] {} text-files detected!".format(len(text_files)))

    text, text_files = p.get_preprocessed_text(text_files)

    algo = Algorithm(text, text_files)

    algo.document_frequency()
    algo.inverse_document_frequency()
    algo.vectorizing_tf_idf_model()

    k_docs = algo.cosine_similarity(query, k)
    k_doc_names = algo.get_doc_names(k_docs)
    
    print("[*] Top {} ranked docs are: ".format(k), k_doc_names)
    # print(algo.tf_idf)

    """
    Additional printing of data, if you want

    # document indexing:
    algo.document_indexing

    # tf-idf (term-frequency inverse_document_frequency)
    algo.tf_idf

    # df (document_frequency)
    algo.df

    # document vectors
    algo.doc_vectors

    # processed and clean data
    algo.processed_data
    """