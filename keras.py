import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding 
from tensorflow.python.keras.optimizers import Adam 
from tensorflow.python.keras.preprocessing.text import Tokenizer 
from tensorflow.python.keras.preprocessing.sequence import pad_sequences 
import imdb 
imdb.maybe_download_and_extract()
input_text_train, target_train = imdb.load_data(train=True) 
input_text_test, target_test = imdb.load_data(train=False)
print("Size of the trainig set: ", len(input_text_train)) 
print("Size of the testing set:  ", len(input_text_test))
text_data = input_text_train + input_text_test 
print (input_text_train[1])

num_top_words = 10000 
tokenizer_obj = Tokenizer(num_words=num_top_words) 
tokenizer_obj.fit_on_texts(text_data) 

# print (tokenizer_obj.word_index)
# print (tokenizer_obj.word_index['the'])
# print (input_text_train[1])
input_train_tokens = tokenizer_obj.texts_to_sequences(input_text_train) 
print (input_train_tokens[1])

input_test_tokens = tokenizer_obj.texts_to_sequences(input_text_test) 

total_num_tokens = [len(tokens) for tokens in input_train_tokens + input_test_tokens] 
total_num_tokens = np.array(total_num_tokens)

print (total_num_tokens)

#Get the average number of tokens 
print (np.mean(total_num_tokens))

print (np.max(total_num_tokens))

max_num_tokens = np.mean(total_num_tokens) + 2 * np.std(total_num_tokens) 
max_num_tokens = int(max_num_tokens) 
print (max_num_tokens)

print (np.sum(total_num_tokens < max_num_tokens) / len(total_num_tokens))
seq_pad = 'pre'
input_train_pad = pad_sequences(input_train_tokens, maxlen=max_num_tokens, padding=seq_pad, truncating=seq_pad)
input_test_pad = pad_sequences(input_test_tokens, maxlen=max_num_tokens, padding=seq_pad, truncating=seq_pad) 

print (input_train_pad.shape)

print (input_train_pad[1])

embedding_layer_size = 8

rnn_type_model = Sequential()
rnn_type_model.add(Embedding(input_dim=num_top_words,
                            output_dim=embedding_layer_size,
                            input_length=max_num_tokens,
                            name='embedding_layer'))
rnn_type_model.add(GRU(units=16, return_sequences=True))
rnn_type_model.add(GRU(units=4))
rnn_type_model.add(Dense(1, activation='sigmoid')) 
model_optimizer = Adam(lr=1e-3)
rnn_type_model.compile(loss='binary_crossentropy',
                        optimizer=model_optimizer,
                        metrics=['accuracy']) 

rnn_type_model.summary()

rnn_type_model.fit(input_train_pad, target_train,
                    validation_split=0.05, epochs=3, batch_size=64)

model_result = rnn_type_model.evaluate(input_test_pad, target_test)
