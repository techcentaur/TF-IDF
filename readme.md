# README / Report


## Content:
- Code Explanation
	- Getting all text-files
	- Pre-Processing of Text
	- Calculation of DF, TF-IDF
	- Ranking using Cosine Similarity
- Code Architecture
- Usage
- Modules Freeze
- Included Test Data

## Code Explanation:

### Getting all text-files:

First, given the path of the directory as CLI argument input, we search through all the possible text files in that directory and return the list of such files. After that, we go one by one through them to process their text.


### Pre-Processing of Text:

When we start to read a file, it can be written in a different encoding. So we have to read it in `utf-8` format that supports all Unicode characters.

We start by pre-processing of text as a string to remove the things that are not needed during analysis.
1. We lower case the data as uppercase and lowercase have significance on our analysis
2. We remove the stop words present in English language (such as the, and, or...) these words often occur and carry very little informativeness about the document.
3. We normalize the apostrophe by converting words such as `n't` into `not` and many more using regex.
4. We remove all punctuation because they are insignificant as per our analysis.
5. We convert all words into their primary stems, i.e., words like play or playing convey the same meaning and have the same level of informativeness.
6. We all convert digits to strings, like 1 to `one` for better analysis.


### Calculation of DF, TF-IDF:

TF is term frequency, and it measures the frequency of a word in a document.
We can write tf(t,d) = count of t in d / # of words present in d

DF is document frequency and it means the occurrence of a word t in documents (if it is present at least one time)

IDF or inverse document frequency, tells the informativeness of a word, it is calculated as `log(N/(df+1))`

tf-idf(t, d) = tf(t, d) + idf(t, d), from this equation we get the actual tf-idf values.

We implement it something like this:
```
for tok in list(set(tokens)):
    tf = counter[tok] / words_count

    df = self.df[tok][0] if tok in self.df else 0 
    idf = np.log((self.num_documents)/(df+1))

    self.tf_idf[i, tok] = tf * idf
```


### Ranking using Cosine Similarity:

Though usual distance metric gives relevant documents, it quite fails when we give long queries, and will not be able to rank them properly. Cosine similarity will mark all the documents as vectors of tf-idf tokens and plots them from the centre. So rather than comparing vectors by varying length, it will calculate similarity by the angle between vectors.


We generate document vectors something like this:
```
    def vectorizing_tf_idf_model(self):
        self.doc_vectors = np.zeros((self.num_documents, self.vocab_size))

        for doc_idx, word in self.tf_idf:
            try:
                idx = self.vocab.index(word)
                self.doc_vectors[doc_idx][idx] = self.tf_idf[(doc_idx, word)]
            except:
                pass
```

We apply cosine metric something like this:
```
    def cosine_metric(x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
```

We also reverse index documents to get the metadata of the document from the top k doc indices.


## Code Architecture

We have design three separate classes are per OOP management architecture for better expandablness of functionality:

```python
class TextClean:
    def lower_case(self, data):
        pass

    def remove_stop_words(self, data):
        pass
    
    def apostrophe_normalisation(self, data):        
        pass
    
    def punctuation_removal(self, data):
		pass        
    
    def stem_processing(self, data):
    	pass
    
    def convert_numbers_to_string(self, data):
        pass
    
    def cleanse_data(self, str_data):
    	pass
```

```python
class Processing:
    def get_all_text_files(self, directory):
        pass

    def get_preprocessed_text(self, text_files):
        pass
```


```python
class Algorithm:
    def __init__(self, processed_data, text_files):
    	pass

    def document_frequency(self):
    	pass

    def inverse_document_frequency(self):
    	pass

    def vectorizing_tf_idf_model(self):
    	pass

    def generate_vectors(self, tokens):
    	pass

    @staticmethod
    def cosine_metric(x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    def cosine_similarity(self, query, k):
        """@params:
        query: is the input query
        k: number of relevant documents
        """
        pass

    def get_doc_names(self, doc_idxs):
    	pass
```

## Usage

Run `python script.py <directory> <long-query> <value-of-k>` giving argument as the directory path where the text files reside.


## Modules Freeze

Updates into `requirements.txt`, install as `pip install -r requirements.txt`

```
nltk==3.5
numpy==1.19.2
num2words==0.5.10
```

## Included Test Data:

4 very short text files from a famous book of Gabriel Garcia Marquez has been uploaded as text data in `testdata` folder.

Example run:
`python script.py testdata/ <query here> <value of k here>`

```console
(venv) :~/$ python script.py testdata/ "love finds good people" 2
[*] 4 text-files detected!
[*] Top 2 ranked docs are:  ['testdata/text3.txt', 'testdata/text2.txt']
```