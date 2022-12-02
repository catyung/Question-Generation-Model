from utils import T5PegasusTokenizer
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import jieba
import pandas as pd 
from transformers import (
    AdamW,
    MT5ForConditionalGeneration,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
import torch

# Using GPU 
device = torch.device("cuda")

def answer_generation(context,answer="",lang="") :

    # Manual Setting the Answer 

    if answer == "" :
        if lang == "chi":
        # Generate Answer from KeyBERT 
            context = " ".join(jieba.cut(context))

        sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        kw_model = KeyBERT(model=sentence_model)
        keywords = kw_model.extract_keywords(context,keyphrase_ngram_range=(1,2),use_mmr=True,diversity=0.9,top_n=3)

        answer_list = []

        for keyword in keywords :
            answer_list.append(keyword[0])
    else :
        answer_list = []
        answer_list = [answer]
    return context,answer_list 

def question_generation(context,answer_list:list,lang="") :
    
    if lang =="en":
        eng_model_path = "iarfmoose/t5-base-question-generator"
        model = T5ForConditionalGeneration.from_pretrained(eng_model_path).to(device)
        tokenizer = T5Tokenizer.from_pretrained(eng_model_path)
        
    else :
        chi_model_path = "final-mt5"
        model = MT5ForConditionalGeneration.from_pretrained(chi_model_path).to(device)
        tokenizer = T5PegasusTokenizer.from_pretrained(chi_model_path)
    
    # Generate Question based on answer     
    
    output_list = {"question":[],"answer":[]}

    for i in range(len(answer_list)) :
        format_input = "qa-question: " + "<answer>" + answer_list[i] + '<context>' + context

        input_ids = tokenizer.encode(format_input, return_tensors='pt').to(device)
        output = model.generate(input_ids,
                            decoder_start_token_id=tokenizer.cls_token_id, #101
                            eos_token_id=tokenizer.sep_token_id, #102
                            max_length=64)
        
        result = tokenizer.decode(output[0])
        result = result.replace("<pad> ","")
        result = result.replace("</s>","")
        
        if lang=="chi":
            result = ''.join(result).replace(' ', '')
            result = result.replace("[CLS]","")
            result = result.replace("</s>[SEP]","")
        
        output_list['question'].append(result)
        output_list['answer'].append(answer_list[i])
        
    return output_list
  
# Testing the output 
eng_context = "Wistron Corporation is an electronics manufacturer based in Taiwan. It was the manufacturing arm of Acer Inc. before being spun off in 2000. As an original design manufacturer, the company designs and manufactures products for other companies to sell under their brand name. Wistron products include notebook and desktop computers, servers, storage, LCD TVs, handheld devices, and devices and equipment for medical applications."
chi_context = """
纬创资通股份有限公司，简称纬创，是一家ODM企业，于2001年由宏碁拆分出来，营运总部位于台湾。纬创资通是全球资讯产品主要供应商之一，全球员工逾80,000名。主要产品包括可携式电脑系统、桌上型电脑系统、伺服器及网路储存设备、资讯家电、通讯产品、云端及绿资源技术。"""

context,ans = answer_generation(chi_context,answer="宏碁",lang="chi")

result_dict = question_generation(context,ans,lang="chi")

qa_df = pd.DataFrame(result_dict)

qa_df
