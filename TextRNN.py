import functools
import os
import numpy as np
import random
import tensorflow as tf
tf.enable_eager_execution()

#

class CharRNN:

    def __init__(self, text, start_char='▶️', stop_char='🛑',
                 sequence_length=250, batch_size=64, embedding_dim=64, shuffle_buffer_size=1000,
                 rnn_units=128, checkpoint_dir='./training_checkpoints'):
        
        self.text = text
        self.text_size = len(self.text)
        print('Text length: %s' %(self.text_size))
        
        self.start_char = start_char
        self.stop_char  = stop_char
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.shuffle_buffer_size = shuffle_buffer_size
        self.rnn_units = rnn_units
        self.checkpoint_dir = checkpoint_dir
        
        #get all characters
        self.vocabulary = sorted(set(self.text))
        self.vocabulary_size = len(self.vocabulary)
        print('Text composed of %s different characters' %(self.vocabulary_size))
        
        #build lookup tables
        self.char_2_index = {char: index for index, char in enumerate(self.vocabulary)}
        self.index_2_char = np.array([char for char in self.vocabulary])
        
        #build a very long vector of all the data
        self.text_indices = np.array([self.char_2_index[char] for char in self.text])

        #transform data for our learning task
        character_dataset = tf.data.Dataset.from_tensor_slices(self.text_indices)
        sequence_dataset  = character_dataset.batch(self.sequence_length,
                                                    drop_remainder=True)
        final_dataset = sequence_dataset.map(self._split_dataset)
        
        final_dataset = final_dataset.shuffle(self.shuffle_buffer_size).batch(self.batch_size, drop_remainder=True)
        self.final_dataset = final_dataset
        
        if tf.test.is_gpu_available():
            print('GPU available :)')
            self.rnn = tf.keras.layers.CUDNNGRU
        else:
            print('No GPU available :(')
            self.rnn = functools.partial(tf.keras.layers.GRU, recurrent_activation='sigmoid')
    


    def _split_dataset(self, dataset):
        input_dataset  = dataset[:-1]
        output_dataset = dataset[1:]
        return(input_dataset, output_dataset)

    
    def _build_model(self, vocabulary_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.Sequential([tf.keras.layers.Embedding(vocabulary_size, embedding_dim,
                                                               batch_input_shape=[batch_size, None]),
                                     self.rnn(rnn_units,
                                              return_sequences=True,
                                              stateful=True
                                              ),
                                     tf.keras.layers.Dense(vocabulary_size),
                                    ])
        return(model)
    
    
    def _loss(self, labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    

    def build_compile(self):
        self.model = self._build_model(vocabulary_size = self.vocabulary_size,
                                       embedding_dim   = self.embedding_dim,
                                       rnn_units       = self.rnn_units,
                                       batch_size      = self.batch_size)
        self.model.summary()

        self.model.compile(optimizer = tf.train.AdamOptimizer(),
                           loss = self._loss)

        self.checkpoint_fp = os.path.join(self.checkpoint_dir, 'checkpoint_{epoch}')
        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_fp,
                                                                      save_weights_only=True)

        
    def fit(self, epochs=5):
        sequences_per_epoch = self.text_size//self.sequence_length
        steps_per_epoch = sequences_per_epoch//self.batch_size
        print('%s steps per epoch' %(steps_per_epoch))
        
        history = self.model.fit(self.final_dataset.repeat(),
                                 epochs = epochs,
                                 steps_per_epoch = steps_per_epoch,                                 callbacks=[self.checkpoint_callback])


    def get_last_trained_model(self):
        last_trained_model_fp = tf.train.latest_checkpoint(self.checkpoint_dir)
        print('Last trained model located at: %s' %(last_trained_model_fp))
        
        self.trained_model = self._build_model(vocabulary_size = self.vocabulary_size,
                                               embedding_dim   = self.embedding_dim,
                                               rnn_units       = self.rnn_units,
                                               batch_size      = 1)

        self.trained_model.load_weights(last_trained_model_fp)
        self.trained_model.build(tf.TensorShape([1, None]))
        self.trained_model.summary()


    def generate_text(self, start_string, seed=None, length_limit=10000):
        if seed is not None:
            tf.random.set_random_seed(seed)
        
        predicted_text = ''

        # start_string to tensor
        input_indices = [self.char_2_index[s] for s in start_string]
        input_tensors = tf.expand_dims(input_indices, 0)


        self.trained_model.reset_states()
        for i in range(length_limit):
            #generates results of (batch=1, 2, vocabulary_size)
            output = self.trained_model(input_tensors)
            #drop the batch dimension
            output = tf.squeeze(output, 0)
            
            output_ids = tf.random.categorical(output, num_samples=1, seed=seed)
            output_ids = output_ids[-1,0].numpy()
            
            #Pass inputs back
            predicted_char = self.index_2_char[output_ids]
            predicted_text += predicted_char
            if predicted_char == self.stop_char:
                break
            else:
                #on to the next one, pass inputs and states back to RNN
                input_tensors = tf.expand_dims([output_ids], 0)

        #reset random seed
        r = random.randint(1000,9999)
        tf.random.set_random_seed(r)
        
        return (start_string + predicted_text)
