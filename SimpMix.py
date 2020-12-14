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
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z, j)
        self.topic_common_word_prob = []  # P(w | z, c)
        self.topic_prob = []  # P(z | d, w)
        self.topic_prob_background = None # P(z | d, w, B)
        self.topic_prob_common = None # P(z | d, w, C)
        self.topic_prob_specific = None # P(z | d, w, j)
        self.pseudo_prob = None

        self.number_of_documents = 0
        self.number_of_documents_collections = [] #documents per collection
        self.vocabulary_size = 0

    def build_corpus(self):
        
        # #############################
        # your code here
        for i, path in enumerate(self.documents_path):
            self.documents_temp = []
            f = open(path, "r", encoding='utf8')
            for x in f:
                doc = str(x).lower()
                if "\x9d" in doc:
                    doc = doc.replace("\x9d", " ")
                #remove punctuation and digits
                #translator = str.maketrans('', '', string.punctuation + string.digits)
                #doc = doc.translate(translator)
                #replace punctuation and digits with a space
                table = doc.maketrans(string.punctuation + string.digits,' '*len(string.punctuation) + ' '*len(string.digits))
                doc = doc.translate(table)
                self.documents.append(doc.split())
        
        self.number_of_documents = len(self.documents)
        # #############################


    def build_vocabulary(self):
        
        # #############################
        # your code here
        voc = []
        for i in self.documents:
            voc = np.concatenate((voc,np.unique(i)))
         
        self.vocabulary = np.unique(voc)
        self.vocabulary_size = len(self.vocabulary)
        #print(self.vocabulary)
        # #############################


    def build_term_doc_matrix(self):
        
        # ############################
        # your code here
        #Whole collection term doc matrix
        self.term_doc_matrix = np.empty(shape=(self.number_of_documents, self.vocabulary_size))
        for i in range(self.number_of_documents):
            for w in range(self.vocabulary_size):
                self.term_doc_matrix[i][w] = self.documents[i].count(self.vocabulary[w])
        


    def build_background_model(self):
        """
        Estimate the background model based on the whole collection
        """
        # ############################
        # your code here
        self.background_word_prob = np.sum(self.term_doc_matrix,axis=0)
        self.background_word_prob = self.background_word_prob/np.sum(self.term_doc_matrix)

        f = open("background.txt", "w")
        f.write(str(self.vocabulary[(-self.background_word_prob).argsort()]))
        f.close()
        # ############################
        


    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        # ############################
        # your code here
        #Initialize
        
        init_temp = np.random.random_sample((self.number_of_documents, number_of_topics))
        self.document_topic_prob = (normalize(init_temp))

        init_temp = np.random.random_sample((number_of_topics, self.vocabulary_size))
        self.topic_word_prob = (normalize(init_temp))

        #pseudo counts uniformly
        self.pseudo_prob = np.ones((self.vocabulary_size))/self.vocabulary_size
        # ############################


    def expectation_step(self, lambdaB, number_of_topics):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")
        
        # ############################
        # your code here
        
        self.topic_prob_background = None # P(z | d, w, B)
        self.topic_prob = None # P(z | d, w, j)

        
        #Background model prob (z)
        topic_prob_num = np.zeros([self.number_of_documents, self.vocabulary_size])
        topic_prob_num += lambdaB*self.background_word_prob
        topic_normalizer = topic_prob_num + ((1-lambdaB) * np.dot(self.document_topic_prob, self.topic_word_prob))
        self.topic_prob_background = ((topic_prob_num)/topic_normalizer)
            
        
        #topic model prob 
        topic_prob_num = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size])
        topic_num = np.reshape(self.document_topic_prob, (self.number_of_documents, number_of_topics, 1)) * np.reshape(self.topic_word_prob, (1, number_of_topics, self.vocabulary_size))
        topic_normalizer = np.sum(topic_num, axis=1)
        topic_divide = np.divide(topic_num, np.reshape(topic_normalizer, (self.number_of_documents, 1, self.vocabulary_size)))
        self.topic_prob = (topic_divide)      
         
        # ############################

            

    def maximization_step(self, number_of_topics, mu):
        """ The M-step updates P(w | z)
        """
        print("M step:")
        
        # update P(w | z)
        
        # ############################
        # your code here
        
        #Update "pi's" values
            
        MStep_num = np.multiply(self.topic_prob, np.reshape(self.term_doc_matrix, (self.number_of_documents, 1, self.vocabulary_size)))
        MStep_num = np.sum(MStep_num, axis=2)
        MStep_normalizer = np.reshape(np.sum(MStep_num, axis=1), (self.number_of_documents, 1))
        self.document_topic_prob = np.divide(MStep_num, MStep_normalizer)


        #update topic word prob
        term_reshape = np.reshape(self.term_doc_matrix, (self.number_of_documents, 1, self.vocabulary_size))
        bacground_reshape = np.reshape((1 - self.topic_prob_background), (self.number_of_documents, 1, self.vocabulary_size))
        MStep_num = np.sum(term_reshape * bacground_reshape * self.topic_prob, axis=0) + (mu * np.reshape(self.pseudo_prob, (1, self.vocabulary_size)))
        MStep_normalizer = np.sum(MStep_num, axis=1)
        self.topic_word_prob = np.divide(MStep_num, np.reshape(MStep_normalizer, (number_of_topics, 1)))
        #print(self.topic_word_prob)



    def calculate_likelihood(self, number_of_topics, lambdaB):
        # ############################
        
        topicsum = (1-lambdaB) * np.dot(self.document_topic_prob, self.topic_word_prob)
        background_reshape = np.reshape(lambdaB*self.background_word_prob, (1, self.vocabulary_size))
        logBT = np.log(background_reshape + topicsum)
        wordsum = np.sum(self.term_doc_matrix * logBT, axis=1)
        docsum = np.sum(wordsum)

        self.likelihoods.append(docsum)
        # ############################
        


    def plsa(self, number_of_topics, max_iter, epsilon, lambdaB, mu):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()

        # build background model
        self.build_background_model()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size])

        # P(z | d) P(w | z)
        self.initialize_randomly(number_of_topics)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")

            # ############################
            # your code here
            self.expectation_step(lambdaB, number_of_topics)
            self.maximization_step(number_of_topics, mu)
            self.calculate_likelihood(number_of_topics, lambdaB)
            print(self.likelihoods[-1])
            #print("maxi")
            #print(self.topic_common_word_prob[0])
            if(iteration>1):
                if(self.likelihoods[-1]-self.likelihoods[-2] < epsilon and self.likelihoods[-1] > self.likelihoods[-2]):
                    print ("Converged")
                    break

        #write to file
        f = open("SimpMix output.txt", "w")
        for i in range(number_of_topics):
            f.write("Theme " + str(i+1) + "\n" + str(self.vocabulary[(-self.topic_word_prob[i]).argsort()[:8]])+ "\n" + str(self.topic_word_prob[i][(-self.topic_word_prob[i]).argsort()[:8]]) + "\n\n")
        f.close()

        # ############################
            



def main():
    np.set_printoptions(edgeitems=9999, linewidth=9999)
    #documents_path = ['data/test.txt', 'data/test2.txt', 'data/test3.txt']
    #documents_path = ['data/apple.txt', 'data/dell.txt', 'data/lenovo.txt']
    documents_path = ['data/Afghanistan war.txt', 'Data/Iraq war.txt']
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 5
    max_iterations = 1000
    epsilon = 0.01
    lambdaB = 0.98
    mu = 1
    corpus.plsa(number_of_topics, max_iterations, epsilon, lambdaB, mu)



if __name__ == '__main__':
    main()
