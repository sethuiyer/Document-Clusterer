import json
import os

from nltk.corpus import stopwords,wordnet
from nltk import pos_tag,word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

def words():
    allWords = None
    with open('data/one-grams.txt', 'r') as infile:
        allWords = [line.strip() for line in infile]

    return set(allWords)


# Extract a list of tokens from a cleaned string.
def tokenize(s):
    stopWords = set(stopwords.words('english'))
    wordsToKeep = words() - stopWords

    return [x.lower() for x in word_tokenize(s)
            if x in wordsToKeep and len(x) >= 3]


def wordnetPos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

documentDict=dict()
for filename in os.listdir('data/cnn-stories'):
    if filename[-3:] == 'txt':
        with open(os.path.join('data/cnn-stories',filename),'r') as infile:
            documentDict[filename]=infile.read()
print "Cleaning...."
documents=[]
for filename,docutext in documentDict.items():
    tokens=tokenize(docutext)
    tagged_tokens=pos_tag(tokens)
    lemma=WordNetLemmatizer()
    stemmedTokens = [lemma.lemmatize(word, wordnetPos(tag)).lower()
                     for word, tag in tagged_tokens]
    documents.append({
        'filename': filename,
        'text': docutext,
        'words': stemmedTokens,
    })
with open('all_stories.json', 'w') as outfile:
    outfile.write(json.dumps(documents))
print 'Cleaning is done!'
