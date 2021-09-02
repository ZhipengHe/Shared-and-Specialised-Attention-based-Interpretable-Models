import os
import pandas as pd
import numpy as np
import math
import itertools
import random

from tensorflow.keras.utils import to_categorical

def reduce_loops(df):
    """Reduce the loops of a trace joining contiguous activities, 
    and exectuted by the same resource  

    Args:
        df : [description]

    Returns:
        dataframe: [description]
    """
    df_group = df.groupby('prefix_id')
    reduced = list()
    for name, group in df_group:
        temp_trace = list()
        group = group.sort_values('task_index', ascending=True).reset_index(drop=True)
        temp_trace.append(dict(prefix_id=name, 
                          caseid=group.iloc[0].caseid,
                          task=group.iloc[0].task, 
                          event_type=group.iloc[0].event_type, 
                          role=group.iloc[0].role, 
                          timelapsed=group.iloc[0].timelapsed,
                          milestone=group.iloc[0].milestone,
                          next_activity=group.iloc[0].next_activity, 
                          task_index=group.iloc[0].task_index))
        for i in range(1, len(group)):
            if group.iloc[i].task == temp_trace[-1]['task'] and group.iloc[i].role == temp_trace[-1]['role'] :
                temp_trace[-1]['task_index'] = group.iloc[i].task_index
                temp_trace[-1]['timelapsed'] = group.iloc[i].timelapsed

            else:
                temp_trace.append(dict(prefix_id=name, 
                          caseid=group.iloc[i].caseid,
                          task=group.iloc[i].task, 
                          event_type=group.iloc[i].event_type, 
                          role=group.iloc[i].role, 
                          timelapsed=group.iloc[i].timelapsed,
                          milestone=group.iloc[i].milestone,
                          next_activity=group.iloc[i].next_activity, 
                          task_index=group.iloc[i].task_index))
        reduced.extend(temp_trace)
    return pd.DataFrame.from_records(reduced)

def create_index(log_df, column):
    """Creates an idx for a categorical attribute.
    Args:
        log_df: dataframe.
        column: column name.
    Returns:
        index of a categorical attribute pairs.
    """
    temp_list = temp_list = log_df[log_df[column] != 'none'][[column]].values.tolist() #remove all 'none' values from the index
    subsec_set = {(x[0]) for x in temp_list}
    subsec_set = sorted(list(subsec_set))
    alias = dict()
    if column !='next_activity':
      for i, _ in enumerate(subsec_set):          
          alias[subsec_set[i]] = i + 1
      alias['none'] = 0
    else:
      for i, _ in enumerate(subsec_set):
          alias[subsec_set[i]] = i  
    #reorder by the index value
    alias = {k: v for k, v in sorted(alias.items(), key=lambda item: item[1])}
    return alias

def normalize_events(log_df,args,features):
    """[summary]

    Args:
        log_df (DataFrame): The dataframe with eventlog data
        args (Dictionary): The set of parameters
        features (list): the list of feature name

    Returns:
        Dataframe: Returns a Dataframe with normalized numerical features
    """

    for feature in features:
        if args['norm_method'] == 'max':
            mean_feature = np.mean(log_df.feature)
            std_feature = np.std(log_df.feature)
            norm = lambda x: (x[feature]-mean_feature)/std_feature
            log_df['%s_norm'%(feature)] = log_df.apply(norm, axis=1)
        elif args['norm_method'] == 'lognorm':
            logit = lambda x: math.log1p(x[feature])
            log_df['%s_log'%(feature)] = log_df.apply(logit, axis=1)
            mean_feature = np.mean(log_df['%s_log'%(feature)])
            std_feature=np.std(log_df['%s_log'%(feature)])
            norm = lambda x: (x['%s_log'%(feature)]-mean_feature)/std_feature
            log_df['%s_norm'%(feature)] = log_df.apply(norm, axis=1)
    return log_df

def reformat_events(log_df, ac_index, rl_index,ne_index):
    """Creates series of activities, roles and relative times per trace.
    Args:
        log_df: dataframe.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
    Returns:
        list: lists of activities, roles and relative times.
    """
    log_df = log_df.to_dict('records')

    temp_data = list()
    log_df = sorted(log_df, key=lambda x: (x['prefix_id'], x['task_index']))
    for key, group in itertools.groupby(log_df, key=lambda x: x['prefix_id']):
        trace = list(group)
        #dynamic features
        ac_order = [x['ac_index'] for x in trace]
        rl_order = [x['rl_index'] for x in trace]
        tbtw = [x['timelapsed_norm'] for x in trace]

    #Reversing the dynamic feature order : Based on "An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism"
        ac_order.reverse()
        rl_order.reverse()
        tbtw.reverse()

        #outcome
        next_activity = max(x['ne_index'] for x in trace)

        temp_dict = dict(caseid=key,
                         ac_order=ac_order,
                         rl_order=rl_order,
                         tbtw=tbtw,
                         next_activity = next_activity)
        temp_data.append(temp_dict)

    return temp_data


def vectorization(log, ac_index, rl_index, ne_index,trc_len,cases):
    """Example function with types documented in the docstring.
    Args:
        #log: event log data in a dictionary.
        #ac_index (dict): index of activities.
        #rl_index (dict): index of roles (departments).
        #di_index (dict) : index of diagnosis codes.

    Returns:
        vec: Dictionary that contains all the LSTM inputs. """

    vec = {'prefixes':dict(), 'prop':dict(),'next_activity':[]} 
    len_ac = trc_len  

    for i ,_ in enumerate(log):

        padding = np.zeros(len_ac-len(log[i]['ac_order']))

        if i == 0:
                vec['prefixes']['x_ac_inp'] = np.array(np.append(log[i]['ac_order'],padding))
                vec['prefixes']['x_rl_inp'] = np.array(np.append(log[i]['rl_order'],padding))
                vec['prefixes']['xt_inp'] = np.array(np.append(log[i]['tbtw'],padding))
                vec['next_activity'] = np.array(log[i]['next_activity'])
                vec['prop']['len'] = np.array(len(log[i]['ac_order']))


        vec['prefixes']['x_ac_inp'] = np.concatenate((vec['prefixes']['x_ac_inp'],
                                                                np.array(np.append(log[i]['ac_order'],padding))), axis=0)
        vec['prefixes']['x_rl_inp'] = np.concatenate((vec['prefixes']['x_rl_inp'],
                                                                np.array(np.append(log[i]['rl_order'],padding))), axis=0)
        vec['prefixes']['xt_inp'] = np.concatenate((vec['prefixes']['xt_inp'],
                                                            np.array(np.append(log[i]['tbtw'],padding))), axis=0)
        vec['next_activity'] = np.append(vec['next_activity'],log[i]['next_activity'])
        vec['prop']['len'] = np.append(vec['prop']['len'],len(log[i]['ac_order']))

    #The concatenation returns a flattened vector. Hence, reshaping the vectors at the end
    vec['prefixes']['x_ac_inp'] = np.reshape(vec['prefixes']['x_ac_inp'],(cases,len_ac))
    vec['prefixes']['x_rl_inp'] = np.reshape(vec['prefixes']['x_rl_inp'],(cases,len_ac))
    vec['prefixes']['xt_inp'] = np.reshape(vec['prefixes']['xt_inp'],(cases,len_ac))

    #one-hot-encoding the y class
    vec['next_activity'] = to_categorical(vec['next_activity'],
                                                num_classes=len(ne_index))

    return vec


# Support function for Vectirization

def lengths(log):
    """This function returns the maximum trace length (trc_len), 
    and the number of cases for train and test sets (cases). 
    The maximum out of trc_len for train and test sets will be 
    used to define the trace length of the dataset that is fed to lstm.

    Args:
        log ([type]): [description]

    Returns:
        trc_len: maximum trace length 
        cases: number of cases for train and test sets
    """
    trc_len = 1
    cases = 1

    for i,_ in enumerate(log):

        if trc_len <len(log[i]['ac_order']):

            trc_len = len(log[i]['ac_order'])
            cases += 1
        else:
            cases += 1

    return trc_len, cases


def split_train_test(df, percentage):
  

  cases = df.prefix_id.unique()
  num_test_cases = int(np.round(len(cases)*percentage))
    
  prefixes = pd.DataFrame(df['prefix_id'].unique(),columns= ['prefix_id'])
  prefixes = prefixes.sample(frac=1, axis=1).reset_index(drop=True)

  test_cases = prefixes[:num_test_cases]
  train_cases = prefixes[num_test_cases:]

  df_train = df.merge(train_cases,on = 'prefix_id')
  df_train = df_train.sort_values(['prefix_id', 'task_index'], ascending = (True, True))

  df_test = df.merge(test_cases,on = 'prefix_id')
  df_test = df_test.sort_values(['prefix_id', 'task_index'], ascending = (True, True))
    
  return df_train, df_test 

