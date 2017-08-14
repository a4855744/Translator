import tensorflow as tf
from collections import namedtuple

HyperParameter = namedtuple('HyperParameter', 'source_num_hidden, target_num_hidden, learning_rate, epoch')


def load_file(file_path):
    file = open(file_path, encoding='utf-8')
    sentences = [sentence.replace('\n', '') for sentence in file]
    return sentences


def tokenizing(sentences):
    token = ['S', 'E', 'P']
    for sentence in sentences:
        token.extend(sentence.split())
    return list(set(token))


def create_dictionary(token):
    dictionary = {key: value for value, key in enumerate(token)}
    return dictionary


def make_data(sentences, dictionary):
    data = []
    for sentence in sentences:
        data.append([dictionary[word] for word in sentence.split()])
    return data


def max_len(data):
    max_length = 0
    for one_data in data:
        max_length = max(max_length, len(one_data))
    return max_length


def padding(data, dictionary, max_length):
    for index in range(len(data)):
        data[index] = data[index] + [dictionary['P']] * (max_length - len(data[index]))
    return data


train_source_sentences = load_file('train_source.txt')
train_target_sentences = load_file('train_target.txt')

train_source_tok = tokenizing(train_source_sentences)
train_target_tok = tokenizing(train_target_sentences)

train_source_dic = create_dictionary(train_source_tok)
train_target_dic = create_dictionary(train_target_tok)

input_data = make_data(train_source_sentences, train_source_dic)
output_data = make_data(train_target_sentences, train_target_dic)
target_data = make_data(train_target_sentences, train_target_dic)

input_max_len = max_len(input_data)
output_max_len = max_len(output_data)

input_data = padding(input_data, train_source_dic, input_max_len)
output_data = padding(output_data, train_target_dic, output_max_len)
target_data = padding(target_data, train_target_dic, output_max_len)

for index in range(len(output_data)):
    output_data[index] = [train_target_dic['S']] + output_data[index]
    target_data[index] = target_data[index] + [train_target_dic['E']]

hps = HyperParameter(source_num_hidden=len(train_source_tok),
                     target_num_hidden=len(train_target_tok),
                     learning_rate=0.01,
                     epoch=3000)

sess = tf.Session()

enc_input = tf.placeholder(tf.float32, [None, None, hps.source_num_hidden])
dec_input = tf.placeholder(tf.float32, [None, None, hps.target_num_hidden])
target = tf.placeholder(tf.int32, [None, None])

input_one_hot = sess.run(tf.one_hot(input_data, hps.source_num_hidden, dtype=tf.int32))
output_one_hot = sess.run(tf.one_hot(output_data, hps.target_num_hidden, dtype=tf.int32))

with tf.variable_scope('encoder'):
    enc_cell = tf.nn.rnn_cell.GRUCell(hps.source_num_hidden)
    enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_input, dtype=tf.float32)
with tf.variable_scope('decoder'):
    dec_cell = tf.nn.rnn_cell.GRUCell(hps.target_num_hidden)
    dec_output, dec_state = tf.nn.dynamic_rnn(dec_cell, dec_input, initial_state=enc_state, dtype=tf.float32)

model = tf.layers.dense(dec_output, hps.target_num_hidden)
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=target))
optimizer = tf.train.AdamOptimizer(hps.learning_rate).minimize(cost)

sess.run(tf.global_variables_initializer())

for step in range(hps.epoch):
    sess.run(optimizer, feed_dict={enc_input: input_one_hot, dec_input: output_one_hot, target: target_data})
    if (step+1) % 100 == 0:
        print(sess.run(cost, feed_dict={enc_input: input_one_hot, dec_input: output_one_hot, target: target_data}))

test_source_sentences = load_file('test_source.txt')
test_input_data = make_data(test_source_sentences, train_source_dic)
test_input_data = padding(test_input_data, train_source_dic, input_max_len)
test_output_data = [train_target_dic['S']] + [train_target_dic['P']] * output_max_len

test_input_one_hot = sess.run(tf.one_hot(test_input_data, hps.source_num_hidden, dtype=tf.int32))
test_output_one_hot = []
for index in range(len(test_input_data)):
    test_output_one_hot.append(sess.run(tf.one_hot(test_output_data, hps.target_num_hidden, dtype=tf.int32)))

prediction = tf.argmax(model, 2)
results = sess.run(prediction, feed_dict={enc_input: test_input_one_hot, dec_input: test_output_one_hot})

decoded = []
for result in results:
    decoded.append([train_target_tok[index] for index in result])
print(decoded)
