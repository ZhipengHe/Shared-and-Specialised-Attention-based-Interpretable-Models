import os
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend, Model, callbacks
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt

def specialised_model(vec, weights, indexes, pre_index, args):

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


    if EXPERIMENT == 'OHE':

        ac_input_ohe = to_categorical(vec['prefixes']['x_ac_inp'][:,:MAX_LEN], num_classes=len(ac_index))
        ac_input = layers.Input(shape=(ac_input_ohe.shape[1],ac_input_ohe.shape[2], ), name='ac_input')

        rl_input_ohe = to_categorical(vec['prefixes']['x_rl_inp'][:,:MAX_LEN], num_classes=len(rl_index))
        rl_input = layers.Input(shape=(rl_input_ohe.shape[1],rl_input_ohe.shape[2], ), name='rl_input')

        t_input = layers.Input(shape=(vec['prefixes']['xt_inp'][:,:MAX_LEN].shape[1], 1), name='t_input')

    else: 
        ac_input = layers.Input(shape=(vec['prefixes']['x_ac_inp'][:,:MAX_LEN].shape[1], ), name='ac_input') 
        rl_input = layers.Input(shape=(vec['prefixes']['x_rl_inp'][:,:MAX_LEN].shape[1], ), name='rl_input')
        t_input = layers.Input(shape=(vec['prefixes']['xt_inp'][:,:MAX_LEN].shape[1], 1), name='t_input')
    
    if EXPERIMENT == 'OHE':
        ac_embs = ac_input
        rl_embs = rl_input
  
    else:
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

    time_embs=t_input
    dim_t = 1

    #Set up the LSTM networks
    #LSTM 
    alpha = layers.Bidirectional(layers.LSTM(lstm_size_alpha, return_sequences=True),
                                        name='alpha')
    beta_ac = layers.Bidirectional(layers.LSTM(lstm_size_beta, return_sequences=True),
                                    name='beta_ac')
    beta_rl = layers.Bidirectional(layers.LSTM(lstm_size_beta, return_sequences=True),
                                    name='beta_rl')
    beta_t = layers.Bidirectional(layers.LSTM(lstm_size_beta, return_sequences=True),
                                    name='beta_t')

    #Dense layer for attention
    alpha_dense = layers.Dense(1, kernel_regularizer=l2(l2reg))
    beta_dense_ac = layers.Dense(dim_ac,
                                activation='tanh', kernel_regularizer=l2(l2reg))
    beta_dense_rl = layers.Dense(dim_rl,
                                activation='tanh', kernel_regularizer=l2(l2reg))
    beta_dense_t = layers.Dense(dim_t,
                                activation='tanh', kernel_regularizer=l2(l2reg))

    #Compute beta, feature attention
    beta_out_ac = beta_ac(ac_embs)
    beta_out_ac = layers.TimeDistributed(beta_dense_ac, name='feature_attention_ac')(beta_out_ac)
    c_t_ac = layers.Multiply(name = 'ac_importance')([beta_out_ac,ac_embs])

    beta_out_rl = beta_rl(rl_embs)
    beta_out_rl = layers.TimeDistributed(beta_dense_rl, name='feature_attention_rl')(beta_out_rl)
    c_t_rl = layers.Multiply(name = 'rl_importance')([beta_out_rl,rl_embs])
    
    beta_out_t = beta_t(time_embs)
    beta_out_t = layers.TimeDistributed(beta_dense_t, name='feature_attention_t')(beta_out_t)
    c_t_t = layers.Multiply(name = 't_importance')([beta_out_t,time_embs])
    
    

    c_t = layers.concatenate([c_t_ac,c_t_rl,c_t_t],name = 'concat')

    #Compute alpha, timestep attention

    alpha_out = alpha(c_t)
    alpha_out = layers.TimeDistributed(alpha_dense, name='alpha_dense')(alpha_out)
    alpha_out = layers.Softmax(name='timestep_attention', axis=1)(alpha_out)



    
    #Compute context vector based on attentions and embeddings
    c_t = layers.Multiply()([alpha_out, c_t])
    c_t = layers.Lambda(lambda x: backend.sum(x, axis=1))(c_t)

    
    #contexts = L.concatenate([c_t,age_input,cl_input_d], name='contexts')
    contexts = layers.Dropout(dropout_context)(c_t)


    
    act_output = layers.Dense(next_activity,
                        activation='softmax',
                        kernel_initializer='glorot_uniform',
                        name='act_output')(contexts)

    model = Model(inputs=[ac_input, rl_input, t_input], outputs=act_output)

    plot_model(
        model,
        to_file="specialised_model.png",
        show_shapes=False,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )

    return model

def specialised_model_fit(vec_train, specialised, indexes, pre_index, MY_WORKSPACE_DIR, batch_size, epochs, args):
    
    EXPERIMENT = args['experiment']

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10)

    output_file_path = os.path.join(os.path.join(MY_WORKSPACE_DIR,
                                        'models'),'model_specialised_' +args['milestone']+
                                        '_{epoch:02d}-{val_loss:.2f}.h5')
    print('This is the output file path ', output_file_path)
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

    model_inputs,model_outputs = generate_inputs(vec_train,args,indexes,EXPERIMENT)
    #model_val_inputs,model_val_outputs = generate_inputs(vec_train,args,indexes)

    specialised_history = specialised.fit(model_inputs,
              {'act_output':model_outputs},
              validation_split=0.15,
              #validation_data=(model_val_inputs, model_val_outputs),
              verbose=1,
              callbacks=[early_stopping, lr_reducer],#callbacks=[early_stopping, model_checkpoint,lr_reducer],
              batch_size=batch_size,
              epochs=epochs)

    return specialised_history


def generate_inputs(vec,args,indexes,experiment):

    index_ac = indexes['index_ac']
    index_rl = indexes['index_rl']
    index_ne = indexes['index_ne']

    experiment = args['experiment']

    prefix_len = args['prefix_length']

    if prefix_len == 'fixed':
        MAX_LEN = args['n_size']
    else:
        MAX_LEN = vec['prefixes']['x_ac_inp'].shape[1] 

    if experiment == 'OHE':
        x = [to_categorical(vec['prefixes']['x_ac_inp'][:,:MAX_LEN],
                                                num_classes=len(index_ac))]
        x.append(to_categorical(vec['prefixes']['x_rl_inp'][:,:MAX_LEN],
                                                num_classes=len(index_rl)))
        x.append(vec['prefixes']['xt_inp'][:,:MAX_LEN])

    else:

        x = [vec['prefixes']['x_ac_inp'][:,:MAX_LEN]]
        x.append(vec['prefixes']['x_rl_inp'][:,:MAX_LEN])
        x.append(vec['prefixes']['xt_inp'][:,:MAX_LEN])
    
    y = vec['next_activity']

    return x,y

def plot_specialised(history):
    #Training and validation curves

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('specialised model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def results_df(y_test,y_pred,index_ne):
    df_results = pd.DataFrame(columns=['sample_index','prediction','ground_truth','prediction_prob','pred_class'])
    sample_index = []
    prediction =[]
    ground_truth =[]
    prediction_prob = []
    pred_class = []

    for i,_ in enumerate(y_test):
        sample_index.append(i)
        prediction.append (index_ne[y_pred[i].argmax()])
        ground_truth.append(index_ne[y_test[i].argmax()])
        prediction_prob.append(round(y_pred[i][y_pred[i].argmax()],4))
        pred_class.append(y_test[i].argmax() == y_pred[i].argmax())

    df_results ['sample_index'] = sample_index
    df_results ['prediction'] = prediction
    df_results ['ground_truth'] = ground_truth
    df_results ['prediction_prob'] = prediction_prob
    df_results ['pred_class'] = pred_class

    return df_results