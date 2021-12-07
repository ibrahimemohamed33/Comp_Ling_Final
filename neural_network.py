import numpy as np
import tensorflow as tf
import pandas as pd


from string import ascii_lowercase

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Sequential, Input
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.python.keras.backend import dtype
from tensorflow.python.keras.losses import MeanAbsoluteError


output_sequence_length = 7

vocabulary = list(ascii_lowercase)
IPA_vocab = list('bdʤʒfghjklmnŋprsʃtʧθðvwxzaæɛəɪiɒʌʊɑːɔue')

text_vectorizer = TextVectorization(output_mode='int',
                                    output_sequence_length=output_sequence_length)


def neural_network(loss_func: str) -> Sequential:
    '''
    Creates the Sequential neural network
    '''
    if loss_func == 'binary':
        loss = BinaryCrossentropy()
    elif loss_func == 'mean_absolute_error':
        loss = MeanAbsoluteError()
    else:
        loss = MeanSquaredError()

    model = Sequential([
        Input(shape=(output_sequence_length, ), dtype=tf.int64),
        Dense(output_sequence_length),
        Dense(output_sequence_length)
    ])

    metrics = [
        BinaryCrossentropy(from_logits=True, name='binary_crossentropy'),
        'accuracy'
    ]

    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    return model


def format_arrays(text_vector: TextVectorization, data_X: pd.Series,
                  data_y: pd.Series) -> tf.Tensor:
    '''
    Formats arrays within the text vectorizer text_vector so they can be
    properly inputted into the neural network
    '''
    X = text_vector.__call__(np.asarray(data_X.to_numpy()).astype(str))
    y = text_vector.__call__(np.asarray(data_y.to_numpy()).astype(str))
    return X, y


@ tf.autograph.experimental.do_not_convert
def run_neural_network(data_train, data_test, output_train, output_test,
                       vocab=vocabulary, loss='binaryx', epochs=10):
    '''
    Runs the model and evaluates it using a binary cross entropy loss function
    or a mean absolute error function. The model is a neural network that
    leverages the adaptive elarning optimization algorithm ADAM, and the
    text vectorizer to convert a noun into data.
    '''

    text_vectorizer.adapt(vocab)
    X, y = format_arrays(text_vectorizer, data_train, output_train)
    Xt, yt = format_arrays(text_vectorizer, data_test, output_test)
    print(X[:10], y[:10])

    NN: Sequential = neural_network(loss)
    print(NN.summary())
    tensorboard_callback = TensorBoard(log_dir="logts")
    NN.fit(x=X, y=y, epochs=epochs, callbacks=[tensorboard_callback])
    return NN.evaluate(Xt,  yt, verbose=2, callbacks=[tensorboard_callback])
