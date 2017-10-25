
# coding: utf-8

# In[119]:


# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 17:15:16 2017

@author: Shalaka
"""

# Text Classification using Word Embeddings
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

tokenizer = RegexpTokenizer(r'\w+')


newsgroups_train = fetch_20newsgroups(subset="train",categories=['alt.atheism','talk.religion.misc'])
newsgroups_test = fetch_20newsgroups(subset="test",categories=['alt.atheism','talk.religion.misc'])
pprint(list(newsgroups_train))
vectorizer = TfidfVectorizer()
vectors = 0
vectors = vectorizer.fit_transform(newsgroups_train.data)
for i in vectors:
    print('\n',i)
#pprint("Shape of vectors",vectors.shape[0])
pprint(vectors.shape)
embedding_size = 20
#import tensorflow as tf
#a = tf.constant(2)

import tensorflow as tf
from tensorflow.contrib import lookup
from tensorflow.python.platform import gfile

PADWORD = 'ZYXW'
MAX_DOCUMENT_LENGTH = 5  




lines = []
reg_expression = "Subject:"
cnt = 0
for doc in newsgroups_train.data:
#     if (cnt >= 20):
#         break
    endLoop = 0
    for line in doc.split('\n'):
        if reg_expression in line:
            if ('Re: ') in line:
                lines.append(line.strip(reg_expression+'Re: '))
                break
    #cnt += 1

max_length = 0
for doc in lines:
    if(len(doc.split()) >= max_length):
        max_length = len(doc.split())
                
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_length)
vocab_processor.fit(lines)
print('Max document length',max_length)
            
with gfile.Open('vocabNewsGroup.tsv', 'wb') as f:
    f.write("{}\n".format(PADWORD))
    for word, index in vocab_processor.vocabulary_._mapping.items():
      f.write("{}\n".format(word))

N_WORDS = len(vocab_processor.vocabulary_)



# In[120]:


table = lookup.index_table_from_file(
  vocabulary_file='vocabNewsGroup.tsv', num_oov_buckets=1, vocab_size=None, default_value=-1)
print(N_WORDS)


# In[123]:


# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with open('vocabNewsGroup.tsv') as f:
    data = tf.compat.as_str(f.read()).split()
  return data
vocabulary = read_data('vocabNewsGroup.tsv')
print('Data size', len(vocabulary))
print('Data',vocabulary)


# In[124]:


numbers = table.lookup(tf.constant('islamic muslim'.split()))
with tf.Session() as sess:
  tf.tables_initializer().run()
  print(numbers.eval())


# In[125]:


titles = tf.constant(lines)
words = tf.string_split(titles)


# In[126]:


print(words)


# In[127]:


print(titles)


# In[128]:


densewords = tf.sparse_tensor_to_dense(words, default_value=PADWORD)
numbers = table.lookup(densewords)


# In[129]:


with tf.Session() as sess:
  tf.tables_initializer().run()
  print(numbers.eval()) 


# In[130]:


max = 0
for doc in lines:
    
    if len(doc.split()) >= max:
        print(doc.split())
        max = len(doc)
print(max)


# In[131]:


padding = tf.constant([[0,0],[0,max_length]])
padded = tf.pad(numbers, padding)
sliced = tf.slice(padded, [0,0], [-1, max])


# In[132]:


EMBEDDING_SIZE = 10
embeddings = tf.Variable(
    tf.random_uniform([N_WORDS, EMBEDDING_SIZE], -1.0, 1.0))


# In[133]:


print(embeddings)


# In[134]:


import math
nce_weights = tf.Variable(
  tf.truncated_normal([N_WORDS, EMBEDDING_SIZE],
                      stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([N_WORDS]))


# In[136]:


batch_size = 856
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])


# In[137]:


def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary


# In[138]:


# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = N_WORDS

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)


# In[146]:


import numpy as np
import collections
import random
data_index = 0
# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    if data_index == len(data):
      #buffer[:] = data[:span]
      for word in data[:span]:
        buffer.append(word)
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels


# In[147]:


batch, labels = generate_batch(batch_size=856, num_skips=2, skip_window=1)
for i in range(856):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])


# In[148]:


valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

num_sampled = 16    # Number of negative examples to sample.


# In[149]:


graph = tf.Graph()


# In[150]:


with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)
  

  # Add variable initializer.
  init = tf.global_variables_initializer()


# In[154]:


# Step 5: Begin training.
num_steps = 100000


# In[155]:


from six.moves import urllib
from six.moves import xrange  

skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.


with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 10
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()


# In[157]:


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')


# In[ ]:




