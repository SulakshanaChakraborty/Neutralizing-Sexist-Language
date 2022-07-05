# Automatic Neutralisation of Sexist Language 


This repository contains all the code, data and pretrained models for our project on automatically neutralising sexist language.  
<img src="sentences.jpeg" width="800" height="200"/>

The repo is structured as follows:
- The current directory contains two python notebooks that can be run using Google Colab. The "NLP_Model" notebook contains the code to pre-process the data, fine tune the model, run inference and evaluate our models. The "Demo" notebook is a shorter version of the other notebook which can be used to demonstrate our code. <br>
- The neutralizing-bias folder contains the code for the pre-trained model created by Pryzant et. al (2019). The paper for this code can be found here: https://arxiv.org/pdf/1911.09709.pdf. <br>

This repository was used to fine tune and run inference on the models. We made some changes to the following files to work with our dataset: <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - harvest/gen_data_from_crawl.py <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - harvest/add_tags.py <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - src/shared/data.py<br>
    
* The data folder contains the raw and processed data for the models. This is broken up into train, test, augmented_data and human_eval folders. The train and test folders were used to fine tune the pre-trained model baseline, the augmented data was used to train a second fine tuned model and the human_eval data was used for manually evaluating the results of the baseline, fine tuned, and augmented data models. 

* The baseline_model folder contains the results of running inference on the baseline model created by Pryzant et. al (2019). 

* The fine_tuned_model folder contains the results of fine tuning the baseline model on the Call Me Sexist dataset. This dataset was created by Samory et. al (2021) with the associated paper here: https://arxiv.org/pdf/2004.12764v2.pdf

* The augmented_data_model folder contains of the second iteration of fine tuning the fine tuned model on another sexist dataset (in train1, train2 and train3). This second dataset is the Workplace Sexism Dataset introduced by Grosz et. al (2020) and the associated paper can be found here: https://arxiv.org/abs/2007.04181. The results of running inference on a test set from another sexist dataset is in the test folder. 

* The demo folder is used for the interactive part of the demo (see Demo.ipynb for more details)

