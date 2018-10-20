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
print("Accuracy: {0:.2%}".format(model_result[1]))

target_predicted = rnn_type_model.predict(x=input_test_pad[0:1000])
target_predicted = target_predicted.T[0] 
class_predicted = np.array([1.0 if prob>0.5 else 0.0 for prob in target_predicted]) 
class_actual = np.array(target_test[0:1000]) 
incorrect_samples = np.where(class_predicted != class_actual) 
incorrect_samples = incorrect_samples[0] 
print ('len',len(incorrect_samples))

index = incorrect_samples[0]
print ('index', index)

incorrectly_predicted_text = input_text_test[index] 
print ('incorrectly_predicted_text',incorrectly_predicted_text)

print (target_predicted[index])

print (class_actual[index])
test_sample_1 = "This movie is fantastic! I really like it because it is so good!" 
test_sample_2 = "Good movie!" 
test_sample_3 = "Maybe I like this movie." 
test_sample_4 = "Meh ..." 
test_sample_5 = "If I were a drunk teenager then this movie might be good." 
test_sample_6 = "Bad movie!" 
test_sample_7 = "Not a good movie!" 
test_sample_8 = "This movie really sucks! Can I get my money back please?" 
test_samples = [test_sample_1, test_sample_2, test_sample_3, test_sample_4, test_sample_5, test_sample_6, test_sample_7, test_sample_8] 
test_samples_tokens = tokenizer_obj.texts_to_sequences(test_samples) 
test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=max_num_tokens,
                           padding=seq_pad, truncating=seq_pad) 
print (test_samples_tokens_pad.shape)
rnn_type_model.predict(test_samples_tokens_pad)
