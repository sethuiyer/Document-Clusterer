'''
    A simple topic model using singular value decomposition
    applied to a corpus of CNN stories.
'''
import json
import numpy as np
from collections import Counter
from scipy.cluster.vq import kmeans2
from numpy.linalg import svd
from shutil import copyfile
def low_rank_approx(matrix, k=10):
    """
    Computes an k-rank approximation of a matrix

    """

    U,sigma,V= np.linalg.svd(matrix, full_matrices=False)

    Ar = np.zeros((len(U), len(V)))
    for i in xrange(k):
        Ar += sigma[i] * np.outer(U.T[i], V[i])
    return U[:,:k],Ar, V[:k,:]

def normalize(matrix):
    '''
        Normalize a document term matrix according to a local
        and global normalization factor. For this we chose a
        simple logarithmic local normalization with a global
        normalization based on entropy.
    '''
    numWords, numDocs = matrix.shape
    localFactors = np.log(np.ones(matrix.shape) + matrix.copy())

    probabilities = matrix.copy()
    rowSums = np.sum(matrix, axis=1)

    # divide each column by the row sums
    assert all(x > 0 for x in rowSums)
    probabilities = (probabilities.T / rowSums).T

    entropies = (probabilities * np.ma.log(probabilities).filled(0) /
                 np.log(numDocs))
    globalFactors = np.ones(numWords) + np.sum(entropies, axis=1)

    # multiply each column by the global factors for the rows
    normalizedMatrix = (localFactors.T * globalFactors).T
    return normalizedMatrix


def makeDocumentTermMatrix(data):
    '''
        Return the document-term matrix for the given list of stories.
        stories is a list of dictionaries {string: string|[string]}
        of the form

            {
                'filename': string
                'words': [string]
                'text': string
            }

        The list of words include repetition, and the output document-
        term matrix contains as entry [i,j] the count of word i in story j
    '''
    words = allWords(data)
    wordToIndex = dict((word, i) for i, word in enumerate(words))
    indexToWord = dict(enumerate(words))
    indexToDocument = dict(enumerate(data))

    matrix = np.zeros((len(words), len(data)))
    for docID, document in enumerate(data):
        docWords = Counter(document['words'])
        for word, count in docWords.items():
            matrix[wordToIndex[word], docID] = count

    return matrix, (indexToWord, indexToDocument)


def cluster(vectors):
    """
    Does a simple k means clustering
    Requires: scipy
    """
    return kmeans2(vectors, k=len(vectors[0]))


def allWords(data):
    words = set()
    for entry in data:
        words |= set(entry['words'])
    return list(sorted(words))


def load():
    with open('all_stories.json', 'r') as infile:
        data = json.loads(infile.read())
    return data



data = load()
matrix, (indexToWord, indexToDocument) = makeDocumentTermMatrix(data)
matrix = normalize(matrix)
print 'Classifying the cnn data into 10 parts'
U,sigma,V = low_rank_approx(matrix,k=10)

projectedDocuments = np.dot(matrix.T, U)
documentCenters, documentClustering = cluster(projectedDocuments)
documentClusters = [
    [indexToDocument[i]['filename']
     for (i, x) in enumerate(documentClustering) if x == j]
    for j in range(len(set(documentClustering)))
]
clusterNumber=1
for clusters in documentClusters:
    for clusteredDocumentName in clusters:
        srcfile='data/cnn-stories/'+clusteredDocumentName
        dstfile='clusteredDocuments/'+str(clusterNumber)+'/'+clusteredDocumentName
        copyfile(srcfile,dstfile)
    clusterNumber+=1
