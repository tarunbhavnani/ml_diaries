# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:30:02 2024

@author: tarun
"""

import transformers
import torch
model='EleutherAI/gpt-neo-125m'

tokenizer= transformers.AutoTokenizer.from_pretrained(model)

pipeline= transformers.pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    torch_dtype= torch.bfloat16,
    trust_remote_code=True,
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id= tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    )


from langchain import HuggingFacePipeline, PromptTemplate, LLMChain

llm= HuggingFacePipeline(pipeline=pipeline)

prompt= PromptTemplate(input_variables= ['content'], template="Tell me a story about {content}.")

chain= LLMChain(llm=llm, prompt=prompt)

print(chain.run("aliens"))
