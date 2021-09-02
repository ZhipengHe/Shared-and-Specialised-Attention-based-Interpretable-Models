import os
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend, Model, callbacks
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt

def shared_model(vec, weights, indexes, pre_index, args):

    EXPERIMENT = args['experiment']
    prefix_len = args['prefix_length']

    if prefix_len == 'fixed':
        MAX_LEN = args['n_size']
    else:
        MAX_LEN = vec['prefixes']['x_ac_inp'].shape[1]

    incl_time = True 
    incl_res = True

    dropout = 0.15
    lstm_size_alpha = 50
    lstm_size_beta = 50
    l2reg=0.0001
    dropout_input = 0.15
    dropout_context=0.15

    ac_weights = weights['ac_weights']
    rl_weights = weights['rl_weights']
    next_activity = weights['next_activity']
    index_ac = indexes['index_ac']
    index_rl = indexes['index_rl']
    index_ne = indexes['index_ne']
    ac_index = pre_index['ac_index']
    rl_index = pre_index['rl_index']
    ne_index = pre_index['ne_index']


    ac_input = layers.Input(shape=(vec['prefixes']['x_ac_inp'][:,:MAX_LEN].shape[1], ), name='ac_input') 
    rl_input = layers.Input(shape=(vec['prefixes']['x_rl_inp'][:,:MAX_LEN].shape[1], ), name='rl_input')
    t_input = layers.Input(shape=(vec['prefixes']['xt_inp'][:,:MAX_LEN].shape[1], 1), name='t_input')
    
    

    
    ac_embs = layers.Embedding(ac_weights.shape[0],
                                    ac_weights.shape[1],
                                    weights=[ac_weights], #the one hot encoded activity weight matrix used as the initial weight matrix
                                    input_length=vec['prefixes']['x_ac_inp'].shape[1],
                                    trainable=True, name='ac_embedding')(ac_input)
    rl_embs = layers.Embedding(rl_weights.shape[0],
                                    rl_weights.shape[1],
                                    weights=[rl_weights],
                                    input_length=vec['prefixes']['x_rl_inp'].shape[1],
                                    trainable=True, name='rl_embedding')(rl_input)


    dim_ac =ac_weights.shape[1]
    dim_rl = rl_weights.shape[1]

    #Time input
    time_embs=t_input
    dim_t = 1

    #Concatenated Input Vector

    full_embs = layers.concatenate([ac_embs,rl_embs,time_embs],name = 'full_embs')
    full_embs = layers.Dropout(dropout)(full_embs)
    dim = dim_ac + dim_rl + dim_t

    #Set up the LSTM networks

    #LSTM 
    alpha = layers.Bidirectional(layers.LSTM(lstm_size_alpha, return_sequences=True),
                                        name='alpha')
    beta = layers.Bidirectional(layers.LSTM(lstm_size_beta, return_sequences=True),
                                    name='beta')


    #Dense layer for attention
    alpha_dense = layers.Dense(1, kernel_regularizer=l2(l2reg))
    beta_dense = layers.Dense(dim,
                                activation='tanh', kernel_regularizer=l2(l2reg))

    #Compute alpha, timestep attention

    alpha_out = alpha(full_embs)
    alpha_out = layers.TimeDistributed(alpha_dense, name='alpha_dense')(alpha_out)
    alpha_out = layers.Softmax(name='timestep_attention', axis=1)(alpha_out)

    #Compute beta, feature attention
    beta_out = beta(full_embs)
    beta_out = layers.TimeDistributed(beta_dense, name='feature_attention')(beta_out)

    #Compute context vector based on attentions and embeddings
    c_t = layers.Multiply(name = 'feature_importance')([alpha_out, beta_out,full_embs])
    c_t = layers.Lambda(lambda x: backend.sum(x, axis=1))(c_t)

    contexts = layers.Dropout(dropout)(c_t)

    act_output = layers.Dense(next_activity,
                        activation='softmax',
                        kernel_initializer='glorot_uniform',
                        name='act_output')(contexts)
    
    shared = Model(inputs=[ac_input, rl_input, t_input], outputs=act_output)

    plot_model(
        shared,
        to_file="shared_model.png",
        show_shapes=False,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )

    return shared

def shared_model_fit(vec_train, shared, indexes, pre_index, MY_WORKSPACE_DIR, batch_size, epochs, args):

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10)

    output_file_path = os.path.join(os.path.join(MY_WORKSPACE_DIR,
                                        'models'),'model_shared_' +args['milestone']+
                                        '_{epoch:02d}-{val_loss:.2f}.h5')
    
    #print('This is the output file path ', output_file_path)
        # Saving
    #model_checkpoint = callbacks.ModelCheckpoint(output_file_path,
    #                                    monitor='val_loss',
    #                                    verbose=1,
    #                                    save_best_only=True,
    #                                    save_weights_only=False,
    #                                    mode='auto')

    lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_loss',
                                    factor=0.5,
                                    patience=10,
                                    verbose=0,
                                    mode='auto',
                                    min_delta=0.0001,
                                    cooldown=0,
                                    min_lr=0)

    model_inputs,model_outputs = generate_inputs_shared(vec_train,args,indexes)
    #model_val_inputs,model_val_outputs = generate_inputs(vec_train,args,indexes)

    shared_history = shared.fit(model_inputs,
              {'act_output':model_outputs},
              validation_split=0.15,
              #validation_data=(model_val_inputs, model_val_outputs),
              verbose=1,
              callbacks=[early_stopping, lr_reducer], #callbacks=[early_stopping, model_checkpoint,lr_reducer],
              batch_size=batch_size,
              epochs=epochs)

    return shared_history

def generate_inputs_shared(vec,args,indexes):

    index_ac = indexes['index_ac']
    index_rl = indexes['index_rl']
    index_ne = indexes['index_ne']

    experiment = args['experiment']

    prefix_len = args['prefix_length']

    if prefix_len == 'fixed':
        MAX_LEN = args['n_size']
    else:
        MAX_LEN = vec['prefixes']['x_ac_inp'].shape[1]
    

    x = [vec['prefixes']['x_ac_inp'][:,:MAX_LEN]]
    x.append(vec['prefixes']['x_rl_inp'][:,:MAX_LEN])
    x.append(vec['prefixes']['xt_inp'][:,:MAX_LEN])
    
    y = vec['next_activity']

    return x,y

def plot_shared(history):
    #Training and validation curves

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('shared model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()




