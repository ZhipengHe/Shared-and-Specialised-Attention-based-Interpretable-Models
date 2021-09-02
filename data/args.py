import os
import pickle

def _params_bpic12(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, EXPERIMENT, N_SIZE):

    parameters = dict()

    parameters['folder'] =  os.path.join(MILESTONE_DIR, "output_files")
    #       Specific model training parameters
    parameters['lstm_act'] = None # optimization function see keras doc
    parameters['dense_act'] = None # optimization function see keras doc
    parameters['optim'] = 'Adam' #'Adagrad' # optimization function see keras doc
    parameters['norm_method'] = 'lognorm' # max, lognorm
    # Model types --> specialized, concatenated, shared_cat, joint, shared
    parameters['model_type'] = 'shared_cat'
    parameters['l_size'] = 50 # LSTM layer sizes
    parameters['n_size'] = N_SIZE
    #    Generation parameters

    parameters['file_name'] = os.path.join(MY_WORKSPACE_DIR,'BPIC_2012_Prefixes.csv') 
    parameters['file_name_all'] = os.path.join(MY_WORKSPACE_DIR,'BPIC_2012_Prefixes_all.csv') 
    parameters['processed_file_name'] = os.path.join(MILESTONE_DIR, 'BPIC_2012_Processed.csv')
    parameters['processed_training_vec'] = os.path.join(MILESTONE_DIR,'vec_training.p')
    parameters['processed_test_vec'] = os.path.join(MILESTONE_DIR, 'vec_test.p')
    parameters['processed_val_vec'] = os.path.join(MILESTONE_DIR, 'vec_val.p')
    parameters['weights'] = os.path.join(MILESTONE_DIR ,'weights.p')
    parameters['indexes'] = os.path.join(MILESTONE_DIR, 'indexes.p')
    parameters['pre_index'] = os.path.join(MILESTONE_DIR, 'pre_index.p') 
    parameters['args'] = os.path.join(MILESTONE_DIR,'args.p')
    parameters['milestone']=MILESTONE
    parameters['experiment'] = EXPERIMENT
    parameters['prefix_length']='fixed' #'variable'

    parameters['log_name'] = 'bpic12'

    return parameters

def get_parameters(dataset, MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, EXPERIMENT,N_SIZE):
    
    if dataset == 'bpic12':
        return _params_bpic12(MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, EXPERIMENT,N_SIZE)
    else:
        raise  ValueError("Please specific dataset 'bpic12'")

def saver(args, vec_train, vec_test, ac_weights, rl_weights, ne_index, index_ac, index_rl, index_ne):

    # saving the processed tensor
    with open(args['processed_training_vec'], 'wb') as fp:
        pickle.dump(vec_train, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open(args['processed_test_vec'], 'wb') as fp:
        pickle.dump(vec_test, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # converting the weights into a dictionary and saving
    weights = {'ac_weights':ac_weights, 'rl_weights':rl_weights, 'next_activity':len(ne_index)}
    with open(args['weights'], 'wb') as fp:
        pickle.dump(weights, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # converting the weights into a dictionary and saving
    indexes = {'index_ac':index_ac, 'index_rl':index_rl,'index_ne':index_ne}
    with open(args['indexes'], 'wb') as fp:
        pickle.dump(indexes, fp, protocol=pickle.HIGHEST_PROTOCOL)

    #saving the arguements (args)
    with open(args['args'], 'wb') as fp:
        pickle.dump(args, fp, protocol=pickle.HIGHEST_PROTOCOL)

def loader(MILESTONE_DIR):
    with open(os.path.join(MILESTONE_DIR,'args.p'), 'rb') as fp:
        args = pickle.load(fp)
    
    with open(args['processed_training_vec'], 'rb') as fp:
        vec_train = pickle.load(fp)
    with open(args['processed_test_vec'], 'rb') as fp:
        vec_test = pickle.load(fp)
        
    with open(args['weights'], 'rb') as fp:
        weights = pickle.load(fp)

    with open(args['indexes'], 'rb') as fp:
        indexes = pickle.load(fp)
    
    return args, vec_train, vec_test, weights, indexes