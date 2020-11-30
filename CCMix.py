import numpy as np
import math
import string

def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.documents_collections = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None #term matrix in corpus
        self.term_doc_collections_matrix = [] #term matrix per collection
        self.background_word_prob = None  # P(w | B)
        self.document_topic_prob = []  # P(z | d)
        self.topic_word_prob = []  # P(w | z, j)
        self.topic_common_word_prob = []  # P(w | z, c)
        self.topic_prob = []  # P(z | d, w)
        self.topic_prob_background = [] # P(z | d, w, B)
        self.topic_prob_common = [] # P(z | d, w, C)
        self.topic_prob_specific = [] # P(z | d, w, j)

        self.number_of_documents = 0
        self.number_of_documents_collections = [] #documents per collection
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        # #############################
        # your code here
        for i, path in enumerate(self.documents_path):
            self.documents_temp = []
            f = open(path, "r", encoding='utf8')
            for x in f:
                doc = str(x)
                table = doc.maketrans(string.punctuation + string.digits,' '*len(string.punctuation) + ' '*len(string.digits))
                doc = doc.translate(table)
                self.documents.append(doc.split())
                self.documents_temp.append(doc.split())
            self.documents_collections.append(self.documents_temp)
            self.number_of_documents_collections.append(len(self.documents_temp))
        
        self.number_of_documents = len(self.documents)
        print(self.number_of_documents_collections)
        # #############################


    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        # #############################
        # your code here
        voc = []
        for i in self.documents:
            voc = np.concatenate((voc,np.unique(i)))
         
        self.vocabulary = np.unique(voc)
        self.vocabulary_size = len(self.vocabulary)
        #print(self.vocabulary)
        # #############################


    def build_term_doc_matrix(self, number_of_collections):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        # ############################
        # your code here
        #Whole collection term doc matrix
        self.term_doc_matrix = np.empty(shape=(self.number_of_documents, self.vocabulary_size))
        for i in range(self.number_of_documents):
            for w in range(self.vocabulary_size):
                self.term_doc_matrix[i][w] = self.documents[i].count(self.vocabulary[w])

        #term doc matrix per collection
        for m, col in enumerate(self.documents_collections):
            term_doc_matrix_temp = np.empty(shape=(len(col), self.vocabulary_size))
            for i in range(len(col)):
                for w in range(self.vocabulary_size):
                    term_doc_matrix_temp[i][w] = self.documents_collections[m][i].count(self.vocabulary[w])
            self.term_doc_collections_matrix.append(term_doc_matrix_temp)
        # ############################
        


    def build_background_model(self):
        """
        Estimate the background model based on the whole collection
        """
        # ############################
        # your code here
        self.background_word_prob = np.sum(self.term_doc_matrix,axis=0)
        self.background_word_prob = self.background_word_prob/np.sum(self.term_doc_matrix)
        
        # ############################
        


    def initialize_randomly(self, number_of_topics, number_of_collections):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        # ############################
        # your code here
        #Initialize
        for i in range(number_of_collections):
            init_temp = np.random.random_sample((self.number_of_documents_collections[i], number_of_topics))
            self.document_topic_prob.append(normalize(init_temp))

            init_temp = np.random.random_sample((number_of_topics, self.vocabulary_size))
            self.topic_word_prob.append(normalize(init_temp))

        self.topic_common_word_prob = np.random.random_sample((number_of_topics, self.vocabulary_size))
        self.topic_common_word_prob = normalize(self.topic_common_word_prob)

        # ############################

        

    def initialize_uniformly(self, number_of_topics, number_of_collections):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_collections, number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = np.divide(self.topic_word_prob, np.reshape(np.sum(self.topic_word_prob,axis=2),(number_of_collections, number_of_topics, 1)))

        self.topic_common_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_common_word_prob = normalize(self.topic_common_word_prob)

    def initialize(self, number_of_topics, number_of_collections, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics, number_of_collections)
        else:
            self.initialize_uniformly(number_of_topics, number_of_collections) #not used

    def expectation_step(self, lambdaB, lambdaC, number_of_collections, number_of_topics):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")
        
        # ############################
        # your code here
        #Background model prob
        print(len(self.documents_collections))
        for i, col in enumerate(self.documents_collections):
            topic_prob_num = np.zeros([self.number_of_documents_collections[i], self.vocabulary_size], dtype=np.float)
            topic_prob_num += lambdaB*self.background_word_prob
            topic_sum = lambdaC*self.topic_common_word_prob + (1-lambdaC)*self.topic_word_prob[i]
            topic_multiply = np.dot(self.document_topic_prob[i], topic_sum)
            topic_normalizer = topic_prob_num + (1-lambdaB)*topic_multiply
            self.topic_prob_background.append((topic_prob_num)/topic_normalizer)
        
            #collection specific model
            topic_prob_num = np.zeros([self.number_of_documents_collections[i], number_of_topics, self.vocabulary_size], dtype=np.float)
            topic_prob_num += (lambdaC*self.topic_common_word_prob + (1-lambdaC)*self.topic_word_prob[i])
            topic_num = np.multiply(np.reshape(self.document_topic_prob[i], (self.number_of_documents_collections[i], number_of_topics, 1)), topic_prob_num)
            topic_normalizer = np.sum(topic_num, axis=1)
            topic_divide = np.divide(topic_num, np.reshape(topic_normalizer, (self.number_of_documents_collections[i], 1, self.vocabulary_size)))
            self.topic_prob_specific.append(topic_divide)

            #common topic model ***may not need to be duplicated across docs***
            topic_prob_num = np.zeros([self.number_of_documents_collections[i], number_of_topics, self.vocabulary_size], dtype=np.float)
            topic_prob_num += lambdaC*self.topic_common_word_prob
            topic_normalizer = topic_prob_num + ((1-lambdaC)*np.reshape(self.topic_word_prob[i], (1, number_of_topics, self.vocabulary_size)))
            self.topic_prob_common.append(topic_prob_num/topic_normalizer)
            
        # ############################

            

    def maximization_step(self, number_of_topics, number_of_collections):
        """ The M-step updates P(w | z)
        """
        print("M step:")
        
        # update P(w | z)
        
        # ############################
        # your code here
        self.topic_common_word_prob = np.zeros([number_of_topics, self.vocabulary_size], dtype=np.float)
        for i, col in enumerate(self.documents_collections):
            #Update "pi's" values
            MStep_num = np.multiply(self.topic_prob_specific[i], np.reshape(self.term_doc_collections_matrix[i], (self.number_of_documents_collections[i], 1, self.vocabulary_size)))
            MStep_num = np.sum(MStep_num, axis=2)
            MStep_normalizer = np.reshape(np.sum(MStep_num, axis=1), (self.number_of_documents_collections[i], 1))
            self.document_topic_prob[i] = np.divide(MStep_num, MStep_normalizer)
            #print(self.document_topic_prob[i])

            #Update common topic word prob
            term_temp = np.reshape(self.term_doc_collections_matrix[i], (self.number_of_documents_collections[i], 1, self.vocabulary_size))
            background_temp = np.reshape((1 - self.topic_prob_background[i]), (self.number_of_documents_collections[i], 1, self.vocabulary_size))
            MStep_num = np.sum(term_temp * background_temp * self.topic_prob_specific[i] * self.topic_prob_common[i], axis=0)
            self.topic_common_word_prob += MStep_num

            #update specific topic word prob
            MStep_num = np.sum(term_temp * background_temp * self.topic_prob_specific[i] * (1 - self.topic_prob_common[i]), axis=0)
            MStep_normalizer = np.sum(MStep_num, axis=1)
            self.topic_word_prob[i] = np.divide(MStep_num, np.reshape(MStep_normalizer, (number_of_topics, 1)))
        
        #normalize topic common word probabilities
        self.topic_common_word_prob = self.topic_common_word_prob/np.reshape(np.sum(self.topic_common_word_prob,axis=1),(number_of_topics, 1))
        # ############################



    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        # ############################
        # your code here
        Pdw = np.dot(self.document_topic_prob,self.topic_word_prob)
        logPd = self.term_doc_matrix*np.log(Pdw)
        log_like = np.sum(logPd)
        self.likelihoods.append(log_like)
        # ############################
        


    def plsa(self, number_of_topics, max_iter, epsilon, number_of_collections, lambdaB, lambdaC):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix(number_of_collections)

        # build background model
        self.build_background_model()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, number_of_collections, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")

            # ############################
            # your code here
            self.expectation_step(lambdaB, lambdaC, number_of_collections, number_of_topics)
            self.maximization_step(number_of_topics, number_of_collections)


        f = open("common.txt", "w")
        for i in range(number_of_topics):
            f.write("Theme " + str(i+1) + " " + str(self.vocabulary[(-self.topic_common_word_prob[i]).argsort()[:10]])+"\n")
        f.close()

        f = open("specific.txt", "w")
        for i in range(number_of_collections):
            for j in range(number_of_topics):
                f.write("Collection " + str(i+1) + " Theme " + str(j+1) + " " + str(self.vocabulary[(-self.topic_word_prob[i][j]).argsort()[:10]])+"\n")
        f.close()
      
            # ############################
            



def main():
    #documents_path = ['data/test.txt', 'data/test2.txt', 'data/test3.txt']
    documents_path = ['data/apple.txt', 'data/dell.txt', 'data/lenovo.txt']
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    np.set_printoptions(edgeitems=9999)
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 5
    number_of_collections = 3
    max_iterations = 100
    epsilon = 0.001
    lambdaB = 0.95
    lambdaC = 0.7
    corpus.plsa(number_of_topics, max_iterations, epsilon, number_of_collections, lambdaB, lambdaC)



if __name__ == '__main__':
    main()
