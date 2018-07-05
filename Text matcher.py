# from structshape import structshape
from collections import Counter
import os
import json

print_steps = True
def step(message):
    if print_steps:
        print("\n" + "*"*70 + '\n{:*^70}\n'.format(message))

# Set working directory to the directory where your corpus folder is:
# Computer: C:/Users/yeachan153/iCloudDrive/iCloud~com~omz-software~Pythonista3/Basic Probability/Week 4
# iPad: /private/var/mobile/Library/Mobile Documents/iCloud~com~omz-software~Pythonista3/Documents/Basic Probability/Week 4
working_directory = input("Enter the directory path that your corpus folder is located in below: \n")
os.chdir(working_directory)


# To speed up things, use a subset of documents during development:
corpus = json.load(open('corpus/corpus-subset.json', 'r'))
#corpus = json.load(open('corpus/corpus.json', 'r'))

def split_text(text):
    return text.lower().split(' ')

def get_file_freqs(filename):
    freqs = Counter()
    with open(filename, 'r', encoding='utf8') as file:
        # line below is a string obj like 'hello my name is' 
        for line in file: 
            words = split_text(line) 
            freqs.update(words)
    return freqs

# 1)
corpus_freqs = Counter()
for docid, info in corpus.items():
    corpus_freqs.update(get_file_freqs(info['filename']))
  
print('Number of unique words in corpus:', len(corpus_freqs))

# 2)
voc_size = 100
common_words_and_values = corpus_freqs.most_common(voc_size)
vocabulary = [each_tuple[0] for each_tuple in common_words_and_values]

# 3)
def freqs_to_vector(freqs, vocabulary):
    vocabulary2 = vocabulary.copy()
    keys_list = [each_key for each_key in freqs if each_key in vocabulary2]
    for index, word in enumerate(vocabulary2):
        for each_key in keys_list:
            if word == each_key:
                vocabulary2[index] = freqs[each_key]
    for idx, each in enumerate(vocabulary2):
        if type(each) == str:
            vocabulary2[idx] = 0
    return vocabulary2

'''
For sanity checks
washington_txt_1789 = get_file_freqs('corpus/1789-Washington.txt')
washington_txt_1793 = get_file_freqs('corpus/1793-Washington.txt')
whitman_leaves = get_file_freqs('corpus/whitman-leaves.txt')

washington_txt_1789_vector = freqs_to_vector(washington_txt_1789, vocabulary)
washington_txt_1793_vector = freqs_to_vector(washington_txt_1793, vocabulary)
whitman_leaves_vector = freqs_to_vector(whitman_leaves, vocabulary)
'''

# 4)
docid = [docid for docid, idx in corpus.items()]
file_names = [corpus[key]['filename'] for key in corpus]

for i in range(len(docid)):
    corpus[docid[i]]['freq_vect'] = freqs_to_vector(get_file_freqs(file_names[i]), vocabulary)

# 5)
def norm(vector):
    import math
    return math.sqrt(sum([number**2 for number in vector]))

#6)
def similarity(A,B):
    if len(A) != len(B):
        print("Length A and B don't match")
        return
    else:
        return sum([A[i]*B[i] for i in range(len(A))]) / (norm(A) * norm(B))

# 7)
washington1 = corpus['1789-Washington']['freq_vect']
washington2 = corpus['1793-Washington']['freq_vect']
whitman = corpus['whitman-leaves']['freq_vect']

print("\nCosine similarities")
print("Washington 1 vs Washington 2:", similarity(washington1, washington2))
print("Washington 1 vs Whitman:", similarity(washington1, whitman))
print("Washington 2 vs Whitman:", similarity(washington2, whitman))

# 8)
def text_to_vector(text, vocabulary):
    return freqs_to_vector(Counter(split_text(text)), vocabulary)

def rank_documents(query_vector, corpus, num=100):
    similarities = {}
    for doc_id, info in corpus.items():
        freq_vect = corpus[doc_id]['freq_vect']
        similarities[doc_id] = similarity(query_vector, freq_vect)

    ranked_ids = sorted(similarities, key=lambda i: similarities[i], reverse=True)
    ranked_sims = [similarities[id] for id in ranked_ids]
    return ranked_ids[:num], ranked_sims[:num]

# 9
adams_txt = "When it was first perceived, in early times, that no middle \
course for America remained between unlimited submission to a foreign \
legislature and a total independence of its claims, men of reflection \
were less apprehensive of danger from the formidable power of fleets \
and armies they must determine to resist than from those contests and \
dissensions which would certainly arise concerning the forms of government \
to be instituted over the whole and over the parts of this extensive country."

print(rank_documents(text_to_vector(adams_txt, vocabulary), corpus))

a,b = rank_documents(text_to_vector(adams_txt, vocabulary), corpus)
for i in range(len(a)):
    print(a[i],b[i])
# Do play around with our querying system. To use the full collection,
# rather than the 3 corpus we used so far, uncomment the line
# `corpus = json.load(open('documents.json', 'r'))` at the top of this file.
# You note that our system isn't very reliable, and can be improved in
# many ways. The first thing you would want to do is tackle stop-words.
