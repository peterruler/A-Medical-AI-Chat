# A Medical AI Chat

# Prerequisites
- Have a fast, modern PC (best is with nvidia cuda support and >= 12 GB graphics RAM), have patience, or use a M1 Mac or an AI web hosting with available GPU e.g. Azure:
- install ollama https://ollama.com - install for your system
- choose the models for your domain: https://ollama.com/library
- mistral is not in german anymore:
- `ollama run mistral` first call downloads model and runs it, ask something like:
- `Schreibe mir ein Python Flask Server, der ein Hello World zurück gibt`
- Mistral german is now custom only:-( - take source from: https://huggingface.co/TheBloke/em_german_leo_mistral-GGUF instead

![Proof](/flask.png?raw=true "flask")

# Useful tips and commands with ollama:
- Bookmark the ollama terminal in MacOS Dock / Windows task bar
- Check in a terminal: `ollama -v`
- `ollama list` lists the installed models!
- `/?` for help
- `ollama rm <Modelname>` delete an installed model!
- exit commmandline chat by typing: `/bye`

# Official Mistral Website with AI gpt comparison
- https://mistral.ai/news/mistral-large

# Test commandline chat
- download and install via Modelfile first, then: `ollama run biomistral1` 7b - german, then ask sth. in german, see example prompts and needed Modelfile further down
- `Was ist der Haupt-Wirkstoff in Aspirin?`
![Proof](/aspirin.png?raw=true "biomistral1")

# Use as a chainlit chat
- to use ollama as a python chainlit chat you can do the following, result can be seen on following screenshot:

![Proof](/flask2.png?raw=true "flask")

# Based on this youtube clip:
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
- `pip install googletrans==4.0.0-rc1` (optional)
- `conda install torch  -c pytorch-nightly`

# GPU Acceleration M1 MacOS - use mps (optional)
- if using conda: `conda install pytorch torchvision torchaudio -c pytorch-nightly`
- or with pip: `pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu`

# Detailed explanation, create a pytorch conda environment (optional)
- Create new env from Jeff Heaton Youtube / Github pages
- MacOS M1 https://www.youtube.com/watch?v=VEDy-c5Sk8Y&t=380s
- other: https://www.youtube.com/results?search_query=jeff+heaton+pytorch+installation

- then activate env by typing in console:
- `conda activate torch`

# Load custom Model - ggml-model-Q8_0.gguf - 7B Model  - M1 Mac needed - in german (optional)
- In case a model is not listed on https://ollama.com/library:
- First install the huggingface-cli:
- `pip3 install huggingface-hub`
- Load the model file to local cache (danger, have at least 6 GB of free space):
- `huggingface-cli download BioMistral/BioMistral-7B-GGUF ggml-model-Q8_0.gguf --local-dir . --local-dir-use-symlinks False`
- `ollama create biomistral1 -f Modelfile`
- `ollama run biomistral1`

# Run the chainlit-chat (optional)
- `cd langchain-gemma-ollama-chainlit` change directory 
- replace in langchain-gemma-ollama-chainlit-de.py: `model = Ollama(model="gemma:2b")` - instead of gemma:2b with your model name e.g. `mistral` or custom installation `biomistral1`.
- and run the mistral code chat in a browser:
- `chainlit run langchain-mistral-ollama-chainlit.py`
- and run the biomistral (install custom) chat in a browser:
- `chainlit run langchain-gemma-ollama-chainlit-de.py`
![Proof](/aspirin2.png?raw=true "biomistral1")

# Install a model thats not listed in the ollama directory (/library) (optional)
- Create the ollama file from custom model:
- (you can initialize, in case of manifest missing error do a `ollama run mistral` first, then:)
- edit Modefile with correct template
- `ollama create biomistral1 -f Modelfile`

# Additional browse for other medical gguf M1 Mac models
- for MacOS M1 SOC GPU no Cuda (PC & Linux)
- https://huggingface.co/search/full-text?q=health+gguf&type=model

# Delete Models in Huggingface cache if not needed or HD is full
- show llms: `cd ~/.cache/huggingface/hub`
- `ls -lia`
- `python -m pip install -U "huggingface_hub[cli]"`
- start removing llms: `huggingface-cli delete-cache`
- select via `Space` - don't select no action then enter und choose `Y`(es)

# Run chainlit chat (optional)
- `cd langchain-gemma-ollama-chainlit` change directory 
- and run the chat in a browser:
- `chainlit run langchain-gemma-ollama-chainlit-de.py` -> medical
- `chainlit run langchain-mistral-ollama-chainlit-de.py` -> code

# Example prompts (english)
- What are the symptoms of the common cold?
- What causes the seasonal flu?
- What medication would be prescribed for a headache?

# Example prompts (german) - with Biomistral trained and finetuned on medical domain:
- Was sind die Symptome einer Erkältung?
- Was verursacht die saisonale Grippe?
- Welche Medikamente würden gegen Kopfschmerzen verschrieben?

# Mistral Model (Programminglanguages better use ollama run codestral, very slow): ollama run mistral
- Wie importiert man ein Package in Java?
- Wie definiert man einen Type mit einem String und einem boolean in Typescript?
- Schreibe mir einen Python Flask Server, der ein Hello World zurück gibt

# Finetune gguf models (very optional, for experts only)
- https://github.com/ggerganov/llama.cpp/tree/master/examples/finetune
- https://rentry.org/cpu-lora

# Screenshot of demo of another prompt:

![Proof](/mistral-code.png?raw=true "Mistral in english only")