import pandas as pd
import numpy as np
from pyflowchart import *
import plotly.express as px
from statistics import mode
from scipy import stats as s

#----------------------------------------------------------------------------------------------------------------------------------
#-- Functions to visualize the modified shared model attention layer
#----------------------------------------------------------------------------------------------------------------------------------

def shared_explain_global(shared_output_with_attention,x_test,y_test,index_ac,index_rl,n,prediction = None): 

  #Generate Feature Names
  feature_names_all = []
  for key in range(len(index_ac)):
    feature_names_all.append(str(index_ac[key]))
  for key in range(len(index_rl)):
    feature_names_all.append(str(index_rl[key]))
  feature_names_all.append('t_lapsed') 

  feature_names_all = np.asarray(feature_names_all)  

  features_short = []

  for i,_ in enumerate(feature_names_all):
    if feature_names_all[i][:5] == 'role_':
      features_short.append(feature_names_all[i])
                                         
    else:
      features_short.append(feature_names_all[i][:5])
    
  feature_dict = {feature_names_all[i]: features_short[i] for i in range(len(features_short))}

  #Obtain attention vectors and input vectors
  timestep_att = shared_output_with_attention[1]
  contexts = shared_output_with_attention[2]
  activity = x_test[0]
  role = x_test[1]

  #Clustering the attention by the prediction

  if prediction != None:
    y_pred_shared = shared_output_with_attention[0]
    pred_ind = []
    for i,_ in enumerate(y_pred_shared):
      if y_pred_shared[i].argmax() == prediction:
        pred_ind.append(i)

    timestep_att=timestep_att[pred_ind]
    contexts=contexts[pred_ind]
    activity=activity[pred_ind]
    role=role[pred_ind]

    
  #timestep attention
  timestep_att = timestep_att.mean(axis=0).flatten()
  #top 3 timesteps to which attention is paid by the model 

  ind = timestep_att.argsort()[-1*n:][::-1]

  #a vector that is used to isolate only top timesteps in the context(the vector which has feature importances) vector
  all_steps = np.zeros(len(timestep_att))
  all_steps[ind] = 1

  #actual number of steps in the trace
  steps = x_test[0].shape[1]
  

  #output of the multiply layer of the timestep_attention, feature_attention and full_emb, and get only the values of most important timesteps
  contexts = contexts.mean(axis=0)
  contexts = contexts.T*all_steps*timestep_att
  contexts = contexts.T


  #generating a dictionary which takes the weights from context vector and puts them against the relevant (actual) timestep and feature
  influential_features = dict()
  timestep = []
  top_features = []
  feature_category = []
  feature_contribution = []
  

  for i,_ in enumerate(contexts):
    timestep.append((np.ones(3)*(steps-i)).tolist()) #when we fed the data into the LSTM, we reversed the sequence. Hence, here we correct it
    ind_ac = int(s.mode(activity.T[i])[0])
    ind_rl = int(s.mode(role.T[i])[0])
    top_features.append(index_ac[ind_ac])
    feature_contribution.append(abs(contexts[i][ind_ac]))
    feature_category.append('activity')
    
    top_features.append(index_rl[ind_rl])
    feature_contribution.append(abs(contexts[i][len(index_ac)+ind_rl]))
    feature_category.append('role')
    
    top_features.append('t_lapsed')
    feature_contribution.append(abs(contexts[i][len(index_ac)+len(index_rl)]))
    feature_category.append('time_lapsed')
    


  influential_features['timestep'] = np.asarray(timestep).flatten()
  influential_features['feature']= np.asarray(top_features).flatten()
  influential_features['feature_category'] = np.asarray(feature_category).flatten()
  influential_features['contribution']= np.asarray(feature_contribution).flatten()

 #if the feature is present in the trace (by evaluating if the feature is actually a part of the actual timestep, given the timestep in inside the actual trace)

  

  #visualizing the influential features in a scatter plot

  df_vis = pd.DataFrame(columns = ['timestep','feature', 'contribution','feature_category'])
  df_vis['timestep']=influential_features['timestep']
  df_vis['feature']=influential_features['feature']
  df_vis['contribution']=influential_features['contribution']
  df_vis['feature_category']=influential_features['feature_category']
  df_vis['contribution'] = df_vis['contribution'].replace({0.0:np.nan, -0.0:np.nan})
  df_vis['feature_short'] = df_vis['feature'].map(feature_dict)
 

  fig = px.bar(df_vis, x="timestep", y="contribution", 
                 title="Feature Contribution",
                 labels={"contribution":"Feature Contribution in key timesteps"}, color = 'feature_category',color_discrete_map={
        'activity': 'darkcyan',
        'role': 'darkslateblue', 'time_lapsed':'lightslategray'},text="feature",  # customize axis label
                barmode = 'group',width=1000, height=600)
  fig.update_traces(textposition='auto',textfont_size=14,textangle=90)
  fig.update_layout(plot_bgcolor='rgb(182,182,184)',bargap = 0.01,uniformtext=dict(minsize=14, mode='show'))
 

  fig.show()
  
  
def shared_explain_local(shared_output_with_attention,x_test,y_test,index_ac, index_rl, index_ne, n,m):

  #prediction output
  y_pred =  shared_output_with_attention[0]
  predicted = index_ne[np.argmax(y_pred[m])]
  ground_truth  = index_ne[np.argmax(y_test[m])]
  probability = y_pred[m].max()
  print('prediction: '+predicted)
  print('ground truth: '+ground_truth)
  print('prediction probability:'+str(round(probability,4)))

  

  #Generate Feature Names
  feature_names_all = []
  for key in range(len(index_ac)):
    feature_names_all.append(str(index_ac[key]))
  for key in range(len(index_rl)):
    feature_names_all.append(str(index_rl[key]))
  feature_names_all.append('t_lapsed') 

  feature_names_all = np.asarray(feature_names_all)  

  features_short = []

  for i,_ in enumerate(feature_names_all):
    if feature_names_all[i][:5] == 'role_':
      features_short.append(feature_names_all[i])
                                         
    else:
      features_short.append(feature_names_all[i][:5])
    
  feature_dict = {feature_names_all[i]: features_short[i] for i in range(len(features_short))}

    
  #timestep attention
  timestep_att = shared_output_with_attention[1][m].flatten()
  #top 3 timesteps to which attention is paid by the model 

  ind = timestep_att.argsort()[-1*n:][::-1]

  #a vector that is used to isolate only top timesteps in the context(the vector which has feature importances) vector
  all_steps = np.zeros(len(timestep_att))
  all_steps[ind] = 1

  #actual number of steps in the trace
  steps = x_test[0].shape[1]
  

  #output of the multiply layer of the timestep_attention, feature_attention and full_emb, and get only the values of most important timesteps
  contexts = shared_output_with_attention[2][m]
  contexts = contexts.T*all_steps*timestep_att
  contexts = contexts.T


  #generating a dictionary which takes the weights from context vector and puts them against the relevant (actual) timestep and feature
  influential_features = dict()
  timestep = []
  top_features = []
  feature_category = []
  feature_contribution = []
  

  for i,_ in enumerate(contexts):
    timestep.append((np.ones(3)*(steps-i)).tolist()) #when we fed the data into the LSTM, we reversed the sequence. Hence, here we correct it
    ind_ac = int(x_test[0][m][i])
    ind_rl = int(x_test[1][m][i])
    
    top_features.append(index_ac[ind_ac])
    feature_contribution.append(abs(contexts[i][ind_ac]))
    feature_category.append('activity')
    
    top_features.append(index_rl[ind_rl])
    feature_contribution.append(abs(contexts[i][len(index_ac)+ind_rl]))
    feature_category.append('role')
    
    top_features.append('t_lapsed')
    feature_contribution.append(abs(contexts[i][len(index_ac)+len(index_rl)]))
    feature_category.append('time_lapsed')


  influential_features['timestep'] = np.asarray(timestep).flatten()
  influential_features['feature']= np.asarray(top_features).flatten()
  influential_features['feature_category'] = np.asarray(feature_category).flatten()
  influential_features['contribution']= np.asarray(feature_contribution).flatten()

 #if the feature is present in the trace (by evaluating if the feature is actually a part of the actual timestep, given the timestep in inside the actual trace)

  

  #visualizing the influential features in a scatter plot

  df_vis = pd.DataFrame(columns = ['timestep','feature', 'contribution','feature_category'])
  df_vis['timestep']=influential_features['timestep']
  df_vis['feature']=influential_features['feature']
  df_vis['contribution']=influential_features['contribution']
  df_vis['feature_category']=influential_features['feature_category']
  df_vis['contribution'] = df_vis['contribution'].replace({0.0:np.nan, -0.0:np.nan})
  df_vis['feature_short'] = df_vis['feature'].map(feature_dict)
 

  fig = px.bar(df_vis, x="timestep", y="contribution", 
                 title="Feature Contribution",
                 labels={"contribution":"Feature Contribution in key timesteps"}, color = 'feature_category',color_discrete_map={
        'activity': 'darkcyan',
        'role': 'darkslateblue', 'time_lapsed':'lightslategray'},text="feature",  # customize axis label
                barmode = 'group',width=1000, height=600)
  fig.update_traces(textposition='auto',textfont_size=14,textangle=90)
  fig.update_layout(plot_bgcolor='rgb(182,182,184)',bargap = 0.05,uniformtext=dict(minsize=14, mode='show'))

  fig.show()

  #Obtaining the actual trace as a flow chart

  trace = dict()

  act = x_test[0][m][:steps][::-1]
  role = x_test[1][m][:steps][::-1]
  tr = x_test[2][m][:steps][::-1]
  ne = y_test[m]

  st = StartNode(index_ac[act[0]]+'_role:'+index_rl[role[0]])

  for i in range(steps -1):
      globals()[f'op_{i+1}'] = OperationNode(index_ac[act[i+1]]+'_role:'+index_rl[role[i+1]])
      if i == 0:
        st.connect(globals()[f'op_{i+1}'])
      else:
        globals()[f'op_{i}'].connect(globals()[f'op_{i+1}'])
        del globals()[f'op_{i}']

  
  #e = EndNode(next_activity)
  #globals()[f'op_{steps-1}'].connect(e)
  del globals()[f'op_{steps-1}']

  fc = Flowchart(st)
  print('process flowchart')
  print(fc.flowchart())
  print('\n')
    
#----------------------------------------------------------------------------------------------------------------------------------
#-- Functions to visualize the modified model attention layer
#----------------------------------------------------------------------------------------------------------------------------------

def explain_global(output_with_attention,x_test,y_test,index_ac, index_rl,n,prediction=None):

  

  #Generate Feature Names
  feature_names_all = []
  for key in range(len(index_ac)):
    feature_names_all.append(str(index_ac[key]))
  for key in range(len(index_rl)):
    feature_names_all.append(str(index_rl[key]))
  feature_names_all.append('t_lapsed') 

  feature_names_all = np.asarray(feature_names_all)  

  features_short = []

  for i,_ in enumerate(feature_names_all):
    if feature_names_all[i][:5] == 'role_':
      features_short.append(feature_names_all[i])
                                         
    else:
      features_short.append(feature_names_all[i][:5])
    
  feature_dict = {feature_names_all[i]: features_short[i] for i in range(len(features_short))}

  #Obtain attention vectors and input vectors
  timestep_att = output_with_attention[1]
  contexts_ac = output_with_attention[2]
  contexts_rl = output_with_attention[3]
  contexts_t = output_with_attention[4]
  activity = x_test[0]
  role = x_test[1]

  #Clustering the attention by the prediction

  if prediction != None:
    y_pred = output_with_attention[0]
    pred_ind = []
    for i,_ in enumerate(y_pred):
      if y_pred[i].argmax() == prediction:
        pred_ind.append(i)

    timestep_att =timestep_att[pred_ind]
    contexts_ac= contexts_ac[pred_ind]
    contexts_rl= contexts_rl[pred_ind]
    contexts_t= contexts_t[pred_ind]
    activity= activity[pred_ind]
    role= role[pred_ind]
    

    
  #timestep attention
  timestep_att = timestep_att.mean(axis=0).flatten()
  #top 3 timesteps to which attention is paid by the model 

  ind = timestep_att.argsort()[-1*n:][::-1]

  #a vector that is used to isolate only top timesteps in the context(the vector which has feature importances) vector
  all_steps = np.zeros(len(timestep_att))
  all_steps[ind] = 1

  #actual number of steps in the trace
  steps = x_test[0].shape[1]
  

  #output of the multiply layer of the timestep_attention, feature_attention and full_emb, and get only the values of most important timesteps
  contexts_ac = contexts_ac.mean(axis=0)
  contexts_ac = contexts_ac.T*all_steps*timestep_att
  contexts_ac = contexts_ac.T

  contexts_rl = contexts_rl.mean(axis=0)
  contexts_rl = contexts_rl.T*all_steps*timestep_att
  contexts_rl = contexts_rl.T
  
  contexts_t = contexts_t.mean(axis=0)
  contexts_t = contexts_t.T*all_steps*timestep_att
  contexts_t = contexts_t.T


  #generating a dictionary which takes the weights from context vector and puts them against the relevant (actual) timestep and feature
  influential_features = dict()
  timestep = []
  top_features = []
  feature_category = []
  feature_contribution = []
  

  for i,_ in enumerate(contexts_ac):
    timestep.append((np.ones(3)*(steps-i)).tolist()) #when we fed the data into the LSTM, we reversed the sequence. Hence, here we correct it
    ind_ac = contexts_ac[i].argmax()
    ind_rl = contexts_rl[i].argmax()
    
    top_features.append(index_ac[ind_ac])
    feature_contribution.append(abs(contexts_ac[i][ind_ac]))
    feature_category.append('activity')
    
    top_features.append(index_rl[ind_rl])
    feature_contribution.append(abs(contexts_rl[i][ind_rl]))
    feature_category.append('role')
    
    top_features.append('t_lapsed')
    feature_contribution.append(abs(contexts_t[i]))
    feature_category.append('time_lapsed')


  influential_features['timestep'] = np.asarray(timestep).flatten()
  influential_features['feature']= np.asarray(top_features).flatten()
  influential_features['feature_category'] = np.asarray(feature_category).flatten()
  influential_features['contribution']= np.asarray(feature_contribution).flatten()

 #if the feature is present in the trace (by evaluating if the feature is actually a part of the actual timestep, given the timestep in inside the actual trace)

  

  #visualizing the influential features in a scatter plot

  df_vis = pd.DataFrame(columns = ['timestep','feature', 'contribution','feature_category'])
  df_vis['timestep']=influential_features['timestep']
  df_vis['feature']=influential_features['feature']
  df_vis['contribution']=influential_features['contribution']
  df_vis['feature_category']=influential_features['feature_category']
  df_vis['contribution'] = df_vis['contribution'].replace({0.0:np.nan, -0.0:np.nan})
  df_vis['feature_short'] = df_vis['feature'].map(feature_dict)
 

  fig = px.bar(df_vis, x="timestep", y="contribution", 
                 title="Feature Contribution",
                 labels={"contribution":"Feature Contribution in key timesteps"}, color = 'feature_category',color_discrete_map={
        'activity': 'darkcyan',
        'role': 'darkslateblue', 'time_lapsed':'lightslategray'},text="feature",  # customize axis label
                barmode = 'group',width=1000, height=600)
  fig.update_traces(textposition='auto',textfont_size=14,textangle=90)
  fig.update_layout(plot_bgcolor='rgb(237,237,235)',bargap = 0.05,uniformtext=dict(minsize=14, mode='show'))


  fig.show()

def explain_local(output_with_attention,x_test,y_test,index_ac,index_rl,index_ne,n,m):

    

    #prediction output
    y_pred =  output_with_attention[0]
    predicted = index_ne[np.argmax(y_pred[m])]
    ground_truth  = index_ne[np.argmax(y_test[m])]
    probability = y_pred[m].max()
    
    print('prediction: '+predicted)
    print('ground truth: '+ground_truth)
    print('prediction probability:'+str(round(probability,4)))


    #Generate Feature Names
    feature_names_all = []
    for key in range(len(index_ac)):
        feature_names_all.append(str(index_ac[key]))
    for key in range(len(index_rl)):
        feature_names_all.append(str(index_rl[key]))
    feature_names_all.append('t_lapsed') 

    feature_names_all = np.asarray(feature_names_all)  

    features_short = []

    for i,_ in enumerate(feature_names_all):
        if feature_names_all[i][:5] == 'role_':
            features_short.append(feature_names_all[i])
                                         
        else:
            features_short.append(feature_names_all[i][:5])
    
    feature_dict = {feature_names_all[i]: features_short[i] for i in range(len(features_short))}

        
    #timestep attention
    timestep_att = output_with_attention[1][m].flatten()
    #top 3 timesteps to which attention is paid by the model 

    ind = timestep_att.argsort()[-1*n:][::-1]

    #a vector that is used to isolate only top timesteps in the context(the vector which has feature importances) vector
    all_steps = np.zeros(len(timestep_att))
    all_steps[ind] = 1

    #actual number of steps in the trace
    steps = x_test[0].shape[1]
    

    #output of the multiply layer of the timestep_attention, feature_attention and full_emb, and get only the values of most important timesteps
    contexts_ac = output_with_attention[2][m]
    contexts_ac = contexts_ac.T*all_steps*timestep_att
    contexts_ac = contexts_ac.T

    contexts_rl = output_with_attention[3][m]
    contexts_rl = contexts_rl.T*all_steps*timestep_att
    contexts_rl = contexts_rl.T
    
    contexts_t = output_with_attention[4][m]
    contexts_t = contexts_t.T*all_steps*timestep_att
    contexts_t = contexts_t.T


    #generating a dictionary which takes the weights from context vector and puts them against the relevant (actual) timestep and feature
    influential_features = dict()
    timestep = []
    top_features = []
    feature_category = []
    feature_contribution = []
    

    for i,_ in enumerate(contexts_ac):
        timestep.append((np.ones(3)*(steps-i)).tolist()) #when we fed the data into the LSTM, we reversed the sequence. Hence, here we correct it
        ind_ac = contexts_ac[i].argmax()
        ind_rl = contexts_rl[i].argmax()
        
        top_features.append(index_ac[ind_ac])
        feature_contribution.append(abs(contexts_ac[i][ind_ac]))
        feature_category.append('activity')
        
        top_features.append(index_rl[ind_rl])
        feature_contribution.append(abs(contexts_rl[i][ind_rl]))
        feature_category.append('role')
        
        top_features.append('t_lapsed')
        feature_contribution.append(abs(contexts_t[i]))
        feature_category.append('time_lapsed')


    influential_features['timestep'] = np.asarray(timestep).flatten()
    influential_features['feature']= np.asarray(top_features).flatten()
    influential_features['feature_category'] = np.asarray(feature_category).flatten()
    influential_features['contribution']= np.asarray(feature_contribution).flatten()

    #if the feature is present in the trace (by evaluating if the feature is actually a part of the actual timestep, given the timestep in inside the actual trace)

    

    #visualizing the influential features in a scatter plot

    df_vis = pd.DataFrame(columns = ['timestep','feature', 'contribution','feature_category'])
    df_vis['timestep']=influential_features['timestep']
    df_vis['feature']=influential_features['feature']
    df_vis['contribution']=influential_features['contribution']
    df_vis['feature_category']=influential_features['feature_category']
    df_vis['contribution'] = df_vis['contribution'].replace({0.0:np.nan, -0.0:np.nan})
    df_vis['feature_short'] = df_vis['feature'].map(feature_dict)
    

    fig = px.bar(df_vis, x="timestep", y="contribution", 
                    title="Feature Contribution",
                    labels={"contribution":"Feature Contribution in key timesteps"}, color = 'feature_category',color_discrete_map={
        'activity': 'darkcyan',
        'role': 'darkslateblue', 'time_lapsed':'lightslategray'},text="feature",  # customize axis label
                    barmode = 'group',width=1000, height=600)
    fig.update_traces(textposition='auto',textfont_size=14,textangle=90)
    fig.update_layout(plot_bgcolor='rgb(237,237,235)',bargap = 0.05,uniformtext=dict(minsize=14, mode='show'))

    fig.show()

    #Obtaining the actual trace as a flow chart

    trace = dict()

    act = x_test[0][m][:steps][::-1]
    careunit = x_test[1][m][:steps][::-1]
    ne = y_test[m]



    st = StartNode(index_ac[act[0]])

    for i in range(steps -1):
        globals()[f'op_{i+1}'] = OperationNode(index_ac[act[i+1]]+'_'+index_rl[careunit[i+1]])
        if i == 0:
            st.connect(globals()[f'op_{i+1}'])
        else:
            globals()[f'op_{i}'].connect(globals()[f'op_{i+1}'])
            del globals()[f'op_{i}']


    next_activity = index_ne[np.argmax(y_test[m])]
    
    del globals()[f'op_{steps-1}']

    fc = Flowchart(st)
    print('process flowchart')
    print(fc.flowchart())
    print('\n')

def results_df(y_test,y_pred,index_ne):
    df_results = pd.DataFrame(columns=['sample_index','prediction','ground_truth','prediction_prob','pred_class'])
    sample_index = []
    prediction =[]
    ground_truth =[]
    prediction_prob = []
    pred_class = []

    for i,_ in enumerate(y_test):
        sample_index.append(i)
        prediction.append(index_ne[y_pred[i].argmax()])
        ground_truth.append(index_ne[y_test[i].argmax()])
        prediction_prob.append(round(y_pred[i][y_pred[i].argmax()],4))
        pred_class.append(y_test[i].argmax() == y_pred[i].argmax())

    df_results ['sample_index'] = sample_index
    df_results ['prediction'] = prediction
    df_results ['ground_truth'] = ground_truth
    df_results ['prediction_prob'] = prediction_prob
    df_results ['pred_class'] = pred_class

    return df_results