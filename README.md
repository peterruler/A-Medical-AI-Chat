# A Medical AI Chat

# Prerequisites
- install ollama https://ollama.com - install for your system
- choose the models for your domain: https://ollama.com/library
- mistral is also in german
- `ollama run mistral:latest` first call downloads model and runs it, ask something like:
- `Schreibe mir ein Python Flask Server, der ein Hello World zurück gibt`
- same command to run a prompt
- Bookmark the olama terminal in dock
- Check in a terminal: `ollama -v`
- `ollama list` lists the installed models

# Test commandline chat
- `ollama run biomistral` 7b - german, then ask sth. in german, see example prompts further down
- `Was ist der haupt Wirkstoff in Aspirin?`

# Exit chat
- stop commmandline chat by typing: `/bye`

# use as a chainlit chat
- to use ollama as a python chainlit chat you can do the following:

# Based on a youtube clip
- https://www.youtube.com/watch?v=bANziaFj_sA

# Chainlit Documentation
- https://docs.chainlit.io/get-started/overview

# Installation of python conda env (optional)
- install miniconda first https://docs.conda.io/en/latest/miniconda.html
- then
- `conda create --name torch python=3.9`
- `conda activate torch`
- `pip install -r ./langchain-gemma-ollama-chainlit/requirements.txt`
- `python -m ipykernel install --user --name torch --display-name "Python 3.9 (torch)"`
- `pip install transformers==4.20.0`
- `pip install langchain`
- `pip install chainlit`
- `pip install openai`
- `pip install googletrans==4.0.0-rc1`
- `conda install torch  -c pytorch-nightly`

# GPU Acceleration M1 MacOS - use mps (optional)
- if using conda: `conda install pytorch torchvision torchaudio -c pytorch-nightly`
- or with pip: `pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu`

# Create a pytorch conda environment (optional)
- Create new env from Jeff Heaton Youtube / Github pages
- MacOS M1 https://www.youtube.com/watch?v=VEDy-c5Sk8Y&t=380s
- other: https://www.youtube.com/results?search_query=jeff+heaton+pytorch+installation

- then activate env by typing in console:
- `conda activate torch`

# Load Model - ggml-model-Q8_0.gguf - 7B Model  - M1 Mac needed - in german (optional)
- In case a model is not listed on https://ollama.com/library:
- First install the huggingface-cli:
- `pip3 install huggingface-hub`
- Load the model file to local cache (danger, have at least 6 GB of free space):
- `huggingface-cli download BioMistral/BioMistral-7B-GGUF ggml-model-Q8_0.gguf --local-dir . --local-dir-use-symlinks False`
- `ollama create biomistral1 -f Modelfile`
- `ollama run biomistral1`
- `cd langchain-gemma-ollama-chainlit` change directory 

# Run the chainlit-chat (optional)
- replace in langchain-gemma-ollama-chainlit-de.py: `model = Ollama(model="gemma:2b")` - instead of gemma:2b with your model name e.g. mistral:latest.
- and run the chat in a browser:
- `chainlit run langchain-gemma-ollama-chainlit-de.py`

# Install a model thats not listed in the ollama directory (/library) (optional)
- Create the ollama file from model:
- `ollama create biomistral -f Modelfile`

# Additional browse for other med gguf M1 Mac models
- for MacOS M1 SOC GPU no Cuda (PC & Linux)
- https://huggingface.co/search/full-text?q=health+gguf&type=model

# Delete Models in Huggingface cache if not needed or HD is full
- show llms: `cd ~/.cache/huggingface/hub`
- `ls -lia`
- `python -m pip install -U "huggingface_hub[cli]"`
- start removing llms: `huggingface-cli delete-cache`
- select via `Space` - don't select no action then enter und choose `Y`(es)

# Install dependencies  (optional)
- `conda activate torch`
- `cd langchain-gemma-ollama-chainlit`
- `pip install -r requirements.txt`
- or more complicated, seperately:
```
pip install langchain
pip install chainlit
pip install openai
pip install googletrans==4.0.0-rc1
```

# run chainlit chat (optional)
- `cd langchain-gemma-ollama-chainlit` change directory 
- and run the chat in a browser:
- `chainlit run langchain-gemma-ollama-chainlit.py`

# Example prompts
- What are the symptoms of the common cold?
- What causes the seasonal flu?
- What medication would be prescribed for a headache?

# Example prompts (german)
- Was sind die Symptome einer Erkältung?
- Was verursacht die saisonale Grippe?
- Welche Medikamente würden gegen Kopfschmerzen verschrieben?
- ollama run mistral:latest
- Wie importiert man ein Package in Java?
- Wie definiert man einen Type mit einem String und einem boolean in Typescript?

# finetune gguf models (optional)
- https://github.com/ggerganov/llama.cpp/tree/master/examples/finetune
- https://rentry.org/cpu-lora

# Screenshot of demo
![Proof](/german-proof.png?raw=true "It works")