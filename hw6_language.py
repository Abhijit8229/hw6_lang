"""
15-110 Hw6 - Language Modeling Project
Name:
AndrewID:
"""

import hw6_language_tests as test

project = "Language" # don't edit this

### WEEK 1 ###

'''
loadBook(filename)
#1 [Check6-1]
Parameters: str
Returns: 2D list of strs
'''
def loadBook(filename):
    corpus = []
    try:
        with open(filename, 'r') as f:
            tp = []
            for i in  f:
                sen = i.strip().split()
                if sen:
                    corpus.append(sen)
        return(corpus)
    except IOError:
        print("Error: could not find  " + filename)


'''
getCorpusLength(corpus)
#2 [Check6-1]
Parameters: 2D list of strs
Returns: int
'''
def getCorpusLength(corpus):
    c = 0
    for i in range(len(corpus)):
        c+=len(corpus[i])
    return c


'''
buildVocabulary(corpus)
#3 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def buildVocabulary(corpus):
    import collections
    d = {}
    over = [i for ch in corpus for i in ch]
    d = collections.Counter(over)
    d = dict(d)
    unique = d.keys()
    # print(list(unique))
    return list(unique)


'''
countUnigrams(corpus)
#4 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countUnigrams(corpus):
    import collections
    d = {}
    over = [i for ch in corpus for i in ch]
    d = collections.Counter(over)
    d = dict(d)

    return d


'''
getStartWords(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def getStartWords(corpus):
    import collections
    d = {}
    over = [lt[0] for lt in corpus ]
    d = collections.Counter(over)
    d = dict(d)
    unique = d.keys()
    return list(unique)


'''
countStartWords(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countStartWords(corpus):
    import collections
    d = {}
    over = [lt[0] for lt in corpus ]
    d = collections.Counter(over)
    d = dict(d)

    return d


'''
countBigrams(corpus)
#6 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def countBigrams(corpus):
    new_dict = {}
    for sentence in corpus:
        for word in range(len(sentence)-1):
            if (sentence[word] not in new_dict):
                new_dict[sentence[word]] = {}
            
            if sentence[word+1] not in new_dict[sentence[word]]:
                    new_dict[sentence[word]][sentence[word+1]] = 0
            new_dict[sentence[word]][sentence[word+1]]+=1
   
    return new_dict


### WEEK 2 ###

'''
buildUniformProbs(unigrams)
#1 [Check6-2]
Parameters: list of strs
Returns: list of floats
'''
def buildUniformProbs(unigrams):
    lt= []
    for i in  unigrams:
        lt.append(1/len(unigrams))
    return lt


'''
buildUnigramProbs(unigrams, unigramCounts, totalCount)
#2 [Check6-2]
Parameters: list of strs ; dict mapping strs to ints ; int
Returns: list of floats
'''
def buildUnigramProbs(unigrams, unigramCounts, totalCount):
    lt  = []
    for key,value in unigramCounts.items():
        lt.append(value/totalCount)
    return lt


'''
buildBigramProbs(unigramCounts, bigramCounts)
#3 [Check6-2]
Parameters: dict mapping strs to ints ; dict mapping strs to (dicts mapping strs to ints)
Returns: dict mapping strs to (dicts mapping strs to (lists of values))
'''
def buildBigramProbs(unigramCounts, bigramCounts):
    new_dict = {}
    for prev_word in bigramCounts:
        words = []
        prob = []
        for i,j in bigramCounts[prev_word].items():
            words.append(i)
            prob.append(j/unigramCounts[prev_word])
        temp_dict = {
            "words":words,
            "probs":prob
        }
        new_dict[prev_word] = temp_dict
        # print(new_dict)
    return new_dict


'''
getTopWords(count, words, probs, ignoreList)
#4 [Check6-2]
Parameters: int ; list of strs ; list of floats ; list of strs
Returns: dict mapping strs to floats
'''
def getTopWords(count, words, probs, ignoreList):
    new_dict={}
    d = {}
    t = 0
    for i in range(len(words)):
        if words[i] not in d and words[i] not in ignoreList :
            d[words[i]] = probs[i]
            t+=1
    sd = sorted(d.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)
    sd = dict(sd)
    ld = list(sd.items())[:count]
    return dict(ld)


'''
generateTextFromUnigrams(count, words, probs)
#5 [Check6-2]
Parameters: int ; list of strs ; list of floats
Returns: str
'''
from random import choices
def generateTextFromUnigrams(count, words, probs):
    generated_words = []
    for _ in range(count):
        
        word = choices(words, weights=probs)[0]
        
        generated_words.append(word)
    return ' '.join(generated_words)

    return


'''
generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs)
#6 [Check6-2]
Parameters: int ; list of strs ; list of floats ; dict mapping strs to (dicts mapping strs to (lists of values))
Returns: str
'''
def generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs):
    generated_words = []
    
    for _ in range(count):
        if len(generated_words) == 0 or generated_words[-1] == '.':
            next_word = choices(startWords, weights=startWordProbs)[0]
        else:
       
            last_word = generated_words[-1]
     
            if last_word in bigramProbs:
                next_word = choices(bigramProbs[last_word]['words'], 
                                            weights=bigramProbs[last_word]['probs'])[0]
            else:
                
                next_word = choices(startWords, weights=startWordProbs)[0]
        
       
        generated_words.append(next_word)
    
   
    return ' '.join(generated_words)


### WEEK 3 ###

ignore = [ ",", ".", "?", "'", '"', "-", "!", ":", ";", "by", "around", "over",
           "a", "on", "be", "in", "the", "is", "on", "and", "to", "of", "it",
           "as", "an", "but", "at", "if", "so", "was", "were", "for", "this",
           "that", "onto", "from", "not", "into" ]

'''
graphTop50Words(corpus)
#3 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTop50Words(corpus):
    unigrams = buildVocabulary(corpus)
    prob = buildUnigramProbs(unigrams,countUnigrams(corpus),getCorpusLength(corpus))
    top_50 = getTopWords(50,unigrams,prob,ignoreList=[])
    barPlot(top_50,title="Top 50 Words")




'''
graphTopStartWords(corpus)
#4 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTopStartWords(corpus):
    startWords = getStartWords(corpus)
    prob = buildUnigramProbs(startWords,countStartWords(corpus),len(startWords))
    top_50 = getTopWords(50,startWords,prob,ignoreList=[])
    barPlot(top_50,title="Top 50 Start Words")
    


'''
graphTopNextWords(corpus, word)
#5 [Hw6]
Parameters: 2D list of strs ; str
Returns: None
'''
def graphTopNextWords(corpus, word):
    bigram_counts = countBigrams(corpus)
    unigram_counts = countUnigrams(corpus)
    bigram = buildBigramProbs(unigram_counts,bigram_counts)
    top_next_words = getTopWords(10,bigram[word]["words"],bigram[word]["probs"], ignore)
    barPlot(top_next_words, title="Top 10 Next Words in Corpus")
    return


'''
setupChartData(corpus1, corpus2, topWordCount)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int
Returns: dict mapping strs to (lists of values)
'''
def setupChartData(corpus1, corpus2, topWordCount):
    corpus1_probabilities = []
    corpus2_probabilities = []
    words = buildVocabulary(corpus1)
    unigrams = countUnigrams(corpus1)
    length = getCorpusLength(corpus1)
    unigram_prob = buildUnigramProbs(list(unigrams.keys()),unigrams,length)
    dictionary = getTopWords(topWordCount,words,unigram_prob,ignore)
    words_2 = buildVocabulary(corpus2)
    unigrams_2 = countUnigrams(corpus2)
    length_2 = getCorpusLength(corpus2)
    unigram_prob_2 = buildUnigramProbs(list(unigrams_2.keys()),unigrams_2,length_2)
    dictionary_2 = getTopWords(topWordCount,words_2,unigram_prob_2,ignore)
    top_words = []
    for word in dictionary:
        if word not in top_words:
            top_words.append(word)
    for word in dictionary_2:
        if word not in top_words:
            top_words.append(word)

    for word in top_words:
        if word in dictionary:
            corpus1_probabilities.append(dictionary[word])
        else:
            corpus1_probabilities.append(0)
        if word in dictionary_2:
            corpus2_probabilities.append(dictionary_2[word])
        else:
            corpus2_probabilities.append(0)
    return {"topWords": top_words,"corpus1Probs" : corpus1_probabilities,"corpus2Probs":corpus2_probabilities}
  


'''
graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; str ; 2D list of strs ; str ; int ; str
Returns: None
'''
def graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title):
    data = setupChartData(corpus1, corpus2, numWords)
    print(data)
    top_words = data["topWords"]
    probabilities1 = data["corpus1Probs"]
    probabilities2 = data["corpus2Probs"]
    sideBySideBarPlots(top_words, probabilities1, probabilities2, name1, name2,title)
    return


'''
graphTopWordsInScatterplot(corpus1, corpus2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int ; str
Returns: None
'''
def graphTopWordsInScatterplot(corpus1, corpus2, numWords, title):
    data = setupChartData(corpus1, corpus2, numWords)

    probabilities1 = data["corpus1Probs"]
    probabilities2 = data["corpus2Probs"]
    top_words = data["topWords"]

 
    scatterPlot(probabilities1, probabilities2, labels=top_words, title=title)
    return


### WEEK 3 PROVIDED CODE ###

"""
Expects a dictionary of words as keys with probabilities as values, and a title
Plots the words on the x axis, probabilities as the y axis and puts a title on top.
"""
def barPlot(dict, title):
    import matplotlib.pyplot as plt

    names = []
    values = []
    for k in dict:
        names.append(k)
        values.append(dict[k])

    plt.bar(names, values)

    plt.xticks(rotation='vertical')
    plt.title(title)

    plt.show()

"""
Expects 3 lists - one of x values, and two of values such that the index of a name
corresponds to a value at the same index in both lists. Category1 and Category2
are the labels for the different colors in the graph. For example, you may use
it to graph two categories of probabilities side by side to look at the differences.
"""
def sideBySideBarPlots(xValues, values1, values2, category1, category2, title):
    import matplotlib.pyplot as plt

    w = 0.35  # the width of the bars

    plt.bar(xValues, values1, width=-w, align='edge', label=category1)
    plt.bar(xValues, values2, width= w, align='edge', label=category2)

    plt.xticks(rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Note that this limits the graph to go from 0x0 to 0.02 x 0.02.
"""
def scatterPlot(xs, ys, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xs, ys)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xs[i], ys[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.xlim(0, 0.02)
    plt.ylim(0, 0.02)

    # a bit of advanced code to draw a y=x line
    ax.plot([0, 1], [0, 1], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    test.week1Tests()
    print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    test.runWeek1()

    ## Uncomment these for Week 2 ##

    print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()


    ## Uncomment these for Week 3 ##

    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()
