# KI Doc

# What are LLMs
- https://www.analyticsvidhya.com/blog/2023/03/an-introduction-to-large-language-models-llms/ unwillingly subscribe to newsletter
- https://medium.com/data-science-at-microsoft/how-large-language-models-work-91c362f5b78f
# My demo notebook not shared
- https://colab.research.google.com/drive/1FoRHdJOAWq4JYhkCRTs3Yi0ghXA1kjc8#scrollTo=CkLQo3pmo-Jk

# Ressource
- https://huggingface.co/vilsonrodrigues/falcon-7b-instruct-sharded

# Docu Falcon 7b
- https://huggingface.co/blog/falcon
- https://vilsonrodrigues.medium.com/run-your-private-llm-falcon-7b-instruct-with-less-than-6gb-of-gpu-using-4-bit-quantization-ff1d4ffbabcc

# translate Dataset from en to de
- https://www.kaggle.com/code/dsxavier/nlp-med-dialogue-analysis-engineering/input Sourcefile in freather format in english

# Translation
- `pip install deep-translator`

# Test Ausgabe
{'User': 'Hi. I have gone through your query with diligence and would like you to know that I am here to help you. For further information consult a neurologist online --> https://www.icliniq.com/ask-a-doctor-online/neurologist  ',
 'Prompt': 'Hi doctor,I am just wondering what is abutting and abutment of the nerve root means in a back issue. Please explain. What treatment is required for\xa0annular bulging and tear?'}

# Chainlit Char Dku
- https://docs.chainlit.io/get-started/overview
 # Linklist
- username: petethegreatest
- KI Jason:
- https://www.youtube.com/watch?v=Q9zv369Ggfk&t=177s
- https://youtu.be/c_nCjlSB1Zk?si=QlPmGKNRoqKEeubW

- https://docs.google.com/spreadsheets/d/1u2bbcSRV99t0Bg9AHFtakpnI3NrC_cVXlR6tZ7yOKlM/edit#gid=456317866
- https://app.relevanceai.com/notebook/f1db6c/f86edbc1-fcb6-41f9-b9b6-be14a6f06412/ef6acb93-c3c2-4e83-86aa-5bb93c9f78ef/use/app
- https://colab.research.google.com/drive/1FoRHdJOAWq4JYhkCRTs3Yi0ghXA1kjc8#scrollTo=1GUD7mBRp2qH
- https://huggingface.co/vilsonrodrigues/falcon-7b-instruct-sharded/blob/main/README.md
- https://pypi.org/project/bitsandbytes/
- https://huggingface.co/datasets/shibing624/medical/blob/main/finetune/test_en_1.json
- https://www.labellerr.com/blog/hands-on-with-fine-tuning-llm/
- https://www.kaggle.com/datasets/dsxavier/diagnoise-me/code
- https://www.kaggle.com/code/damianpanek/diagnose-me-fine-tuning-gptneo-125m/notebook
- https://www.kaggle.com/code/dsxavier/nlp-med-dialogue-analysis-engineering/comments
- https://www.kaggle.com/datasets?search=health
- https://geekflare.com/de/midjourney-prompts/

- https://huggingface.co/jianghc/medical_chatbot

streamlit.io relevanceio.com

# finetune on M1 Mac
- https://huggingface.co/blog/abhishek/phi3-finetune-macbook

- https://stackoverflow.com/questions/76589840/cant-run-transformer-fine-tuning-with-m1-mac-cpu

My suggestion would be to remove any moving code & do it through the transformers library. You can specify no_cuda in TrainingArguments as False so that the training objects aren't moved to GPU.