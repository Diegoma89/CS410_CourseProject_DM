import numpy as np
import string

narray1 = np.array([4, 6, 8, 3, 0 ,9 ,12, 56, 73, 4, 7, 11, 20, 190, 17, 0, 1, 2, 7, 9, 10, 190])

#print((-narray1).argsort()[:5])

narray2 = np.array([1, 0, 1, 0])
narray3 = np.array([1, 0, 0, 1])

string_a = "Hello world, this has punctuation and numbers 12345!"
print(string_a)
translator = str.maketrans('', '', string.punctuation + string.digits)

print(string_a.translate(translator))