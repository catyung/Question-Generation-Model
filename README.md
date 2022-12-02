# Question-Generation-Model

# Aim :
#### Helping teacher / instrutor to genearte question (MCQ / Short Question) from context, the current MVP support 2 languages : 
- English 
- Chinese (Simplifed Chinese - > Traditional Chinese) 

By using MT5 with XQUAD (https://huggingface.co/datasets/xquad/viewer/xquad.zh/validation) to fine-tune the MT5 model for question answering task

* !!! MT5 do not have any downstream task !!!


# Tech Stack & Packages :
- Python 3.9 
- T5 & MT5
- Transformers 
- PyTorch 
- KeyBERT


# Models :
The current fine-tune model is only for Chinese model, for English one is fine-tune by iarfmoose 
- English model : iarfmoose/t5-base-question-generator (T5)
- Chinese model : imxly/t5-pegasus-small (MT5)

Credit to imxly to provide the model in T5 Pegasus, coz I have tried with Mt5-base model, the performance on Chinese is not good.

## Concept :
User will have to provide 

1. <context> for the question, like an acticle (current limitation is 512 token - due to using BERT)

2. <answer> of the question (optional) , if user didnt provide answer, the KeyBERT model will extract the keyphrases from the context automatically, top_n = 5 

3. Language : "en" or "chi" (if you are using Traditional Chinese, pleaes translate to Simplified Chinese) 

# Roadmap :
- Thinking of generating the MCQ options (wrong answer) , thinking of different approaches by considering if the keywords can be found on context ? since the student can easily spot the answer by looking at the context, if we dont have suitable distration for them.
