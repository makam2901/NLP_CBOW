# NLP - Continuous Bag of Words

## Objective 
The objective of this project is to be able to predict the `middle word` of a phrase when its `neighbouring (context) words` are given.

## Background 
### Word2vec
Word2vec is a group of related models that are used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words. Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space. Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located close to one another in the space.
### CBOW 
CBOW or Continous bag of words is to use embeddings in order to train a neural network where the context is represented by multiple words for a given target words.
For example, we could use “cat” and “tree” as context words for “climbed” as the target word.
This calls for a modification to the neural network architecture.
## Working Principle
### Libraries required
``` 
import numpy as np
import keras.backend as K
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, Lambda
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import pandas as pd
```
### Loading data 
More the amount of training data, more the accuracy in predicting the target word is achieved.
For the problem in hand, I have chosen the wikipedia introduction of `amnesia` as training data.
The training data can be accessed through loading the `text.txt` file. 
```
file_name = 'text.txt'
corpus = open(file_name).readlines()
```
**Text:** 
```Amnesia is a deficit in memory caused by brain damage or disease, but it can also be caused temporarily by the use of various sedatives and hypnotic drugs.  
The memory can be either wholly or partially lost due to the extent of damage that was caused.  
There are two main types of amnesia: retrograde amnesia and anterograde amnesia.   
Retrograde amnesia is the inability to retrieve information that was acquired before a particular date, usually the date of an  
accident or operation.  
In some cases the memory loss can extend back decades, while in others the person may lose only a few months of memory.  
Anterograde amnesia is the inability to transfer new information from the short-term store into the long-term store.  
People with anterograde amnesia cannot remember things for long periods of time.  
These two types are not mutually exclusive; both can occur simultaneously.  
Case studies also show that amnesia is typically associated with damage to the medial temporal lobe.  
In addition, specific areas of the hippocampus (the CA1 region) are involved with memory.  
Research has also shown that when areas of the diencephalon are damaged, amnesia can occur.  
Recent studies have shown a correlation between deficiency of RbAp48 protein and memory loss.  
Scientists were able to find that mice with damaged memory have a lower level of RbAp48 protein compared to normal, healthy  
mice.  
In people suffering with amnesia, the ability to recall immediate information is still retained, and they may still be able   
to form new memories.  
However, a severe reduction in the ability to learn new material and retrieve old information can be observed.  
Patients can learn new procedural knowledge.  
In addition, priming (both perceptual and conceptual) can assist amnesiacs in the learning of fresh non-declarative knowledge.  
Amnesic patients also retain substantial intellectual, linguistic, and social skill despite profound impairments in the  
ability to recall specific information encountered in prior learning episodes.  
```
**Note:** Each sentence of the training data must in a new line when created via Notepad, Word ,etc.

### Data Preprocessing 
* The sentences containing 2 or more words are only considered.
* The sentences were cleaned by omitting all the punctuation marks, extra spaces, etc using `Tokenizer`
* The sentences are converted into arrays containing the indices of the respective words.
* Total number of words `(n_samples)` and number of unique words `(V)` is found out.
* Create a dictionary of the words
```
corpus = [sentence for sentence in corpus if sentence.count(" ") >= 2]

tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'+"'")
tokenizer.fit_on_texts(corpus)

corpus = tokenizer.texts_to_sequences(corpus) 

n_samples = sum(len(s) for s in corpus)       
V = len(tokenizer.word_index) + 1       

words = list((tokenizer.word_index.items()))
```
```
words = [('the', 1), ('of', 2), ('amnesia', 3), ('in', 4), ('to', 5), ('can', 6), ('memory', 7), ('and', 8),...]
```
### Window size
It is the number of words on either one of the sides considered to predict the middle word. `Window_size * 2` gives us the total number of neighbours of the target word.  
Here, 
```
window_size = 2
neighbours = window_size*2
```
### Generating training data
```
X_cbow, y_cbow = generate_data_cbow(corpus, window_size, V)
```
**Note:** All the codes to the functions are available in `.ipynb` or `.py` files.
Every word in the corpus has 4 neighbours and the corner words have 2 or 3 neighbours.  

In this step, 
* a matrix, `X_cbow` is created where each row contains the indices of the 4 neighbours of each of the target words.
```
X_cbow =  [[  0   0  10   9]
 [  0   3   9  56]
 [  3  10  56   4]
 ...
 [167   4  55 169]
 [  4 168 169   0]
 [168  55   0   0]]
```
**Note:** If a word has no neighbour it is represented as 0.
* another matrix, `y_cbow` is created, where each row represents the `one hot encoding` of each of the target words.
```
y_cbow[0] =  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0.]
 ```
### CBOW framework
```
dim = 10
cbow = Sequential()

    
cbow.add(Embedding(input_dim=V, 
                   output_dim=dim,
                   input_length = window_size*2, # Note that we now have 2L words for each input entry
                   embeddings_initializer='glorot_uniform'))

cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(dim, )))

cbow.add(Dense(V, activation='softmax', kernel_initializer='glorot_uniform'))

cbow.compile(optimizer=keras.optimizers.Adam(),
             loss='categorical_crossentropy',
             metrics=['accuracy'])
    
cbow.summary()
```
* First, set the desired number of dimensions for the model to learn.
* A sequential model with 3 layers: the Embedding layer, the Lambda layer and the Dense layer is created.
* The optimizer used is `Adam`, which is an optimization technique for gradient descent.
* The activation function used is `softmax`, which  converts vectors of numbers to vectors of probabilities. 
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 4, 10)             1700      
_________________________________________________________________
lambda_1 (Lambda)            (None, 10)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 170)               1870      
=================================================================
Total params: 3,570
Trainable params: 3,570
Non-trainable params: 0
_________________________________________________________________
```

### Training 
* The CBOW model created is fitted to the training data for it learn the weights and be able to predict any outcomes.
* Epochs represent the iterations and batch size is the number of training examples in one pass.
 ```
 cbow.fit(X_cbow, y_cbow, batch_size=64, epochs=500, verbose=1)
 
 Epoch 1/500
5/5 [==============================] - 3s 4ms/step - loss: 5.1366 - accuracy: 0.0000e+00
Epoch 2/500
5/5 [==============================] - 0s 4ms/step - loss: 5.1326 - accuracy: 0.0031
Epoch 3/500
5/5 [==============================] - 0s 4ms/step - loss: 5.1292 - accuracy: 0.0063
....
Epoch 498/500
5/5 [==============================] - 0s 6ms/step - loss: 1.1477 - accuracy: 0.8176
Epoch 499/500
5/5 [==============================] - 0s 5ms/step - loss: 1.1440 - accuracy: 0.8176
Epoch 500/500
5/5 [==============================] - 0s 6ms/step - loss: 1.1403 - accuracy: 0.8176
```
**Note:** The accuracy of the model increases with number of `epochs`.
 
### PCA
Principal Component Analysis (PCA) is a dimension reduction tool that can be used to reduce a large set of variables to a small set that still contains most of the information in the original set.
* Embedding layer weights are accessed by using `cbow.get_weights()[0]` and stored them in `embedding`
```
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
final_result = pca.fit_transform(embedding)
fig, ax = plt.subplots(1,1,figsize = (20,20))
plt.scatter(final_result[:,0], final_result[:,1])
for i, word in enumerate(words):
    ax.annotate(word, xy = (final_result[i, 0], final_result[i, 1]))
```
![](https://github.com/makam2901/NLP_CBOW/blob/master/pca.png)

**Note:** This is to only give a rough idea how PCA works. The training data and accuracy of the model needs to be increased in order to get related words close by in the plot.

## Input Prompt
```
Enter the neighbouring words  : inability to new information  
input_words =  ['inability to new information']
```
**Note:** The words entered in the prompt should be space seperated 
## Output Function
* First, the function converts the input words into their respective `indices` (tokens).
```
input_words =  ['inability to new information']
input_tokens = tokenizer.texts_to_sequences(input_words)

input_tokens: [[28, 5, 17, 12]]
```
* Using `cbow.predict()` we can get the softmax output of the input words.
* Mapping the `max` of the softmax elements of the `cbow.predict()` with the words dictionary, we get our desired output.
```
Required target word: 
transfer
```

 
 
 
 
 
 
 
