## Building Interpretable Models for Business Process Prediction using Shared and Specialised Attention Mechanisms

> Please install `Git LFS` if you want to clone this repo.

### Environment

- Jupyter Lab
- TensorFlow>=2.5
- pandas
- scipy
- numpy
- seaborn
- pyflowchart
- plotly

### BPIC 2012

An application is submitted through a webpage. Then, some automatic checks are performed, after which the application is complemented with additional information. This information is obtained trough contacting the customer by phone. If an applicant is eligible, an offer is sent to the client by mail. After this offer is received back, it is assessed. When it is incomplete, missing information is added by again contacting the customer. Then a final assessment is done, after which the application is approved and activated.

#### Event type explanation

| **Event Type**            | **Meaning**                                                                                                                   |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| States starting with ‘A_’ | States of the application                                                                                                     |
| States starting with ‘O_’ | States of the offer belonging to the application                                                                              |
| States starting with ‘W_’ | States of the work item belonging to the application                                                                          |
| COMPLETE                  | The task (of type ‘A_’ or ‘O_’) is completed                                                                                  |
| SCHEDULE                  | The work item (of type ‘W_’) is created in the queue (automatic step following manual actions)                                |
| START                     | The work item (of type ‘W_’) is obtained by the resource                                                                      |
| COMPLETE                  | The work item (of type ‘W_’) is released by the resource and put back in the queue or transferred to another queue (SCHEDULE) | 

#### Event Translations

Below, we present some translations for the most important events in this event log.

| **Dutch state name**           | **English translation**                                 |
| ------------------------------ | ------------------------------------------------------- |
| W_Afhandelen leads             | W_Fixing incoming lead                                  |
| W_Completeren aanvraag         | W_Filling in information for the application            |
| W_Valideren aanvraag           | W_Assessing the application                             |
| W_Nabellen offertes            | W_Calling after sent offers                             |
| W_Nabellen incomplete dossiers | W_Calling to add missing information to the application |
| W_Wijzigen contractgegevens    | W_Change contract details                               |
| W_Beoordelen fraude            | W_Assess fraud                                          | 


### Experiment Notebooks

#### Model Performance

- BPIC 2012 Complete -- `bpic_2012_all.ipynb`
- BPIC 2012 A -- `bpic_2012_A.ipynb`
- BPIC 2012 O -- `bpic_2012_O.ipynb`
- BPIC 2012 W Complete -- `bpic_2012_W.ipynb`

#### Decision Point Analysis
- Decision Point "A_PREACCEPTED" -- `bpic_2012_A_PREACCEPTED.ipynb`
- Decision Point "W_Nabellen offertes" -- `bpic_2012_W_Nabellen offertes.ipynb`



