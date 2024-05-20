# A Medical AI Chat

# Based on a youtube clip
- https://www.youtube.com/watch?v=bANziaFj_sA

# installation of python conda env
- install miniconda first https://docs.conda.io/en/latest/miniconda.html
- `conda create --name torch python=3.9`
- `conda activate torch`
- `pip install transformers==4.20.0`
- `conda install torch  -c pytorch-nightly`
- `python -m ipykernel install --user --name torch --display-name "Python 3.9 (torch)"`

# create a pytorch conda environment (optional)
- Create new env from Jeff Heaton Youtube / Github pages
- MacOS M1 https://www.youtube.com/watch?v=VEDy-c5Sk8Y&t=380s
- other: https://www.youtube.com/results?search_query=jeff+heaton+pytorch+installation

- then activate env by typing in console:
- `conda activate torch`

# Load Model - BioMistral-7B.Q5_K_M.gguf
- First install the huggingface-cli:
- `pip3 install huggingface-hub`
- Load the model file to local cache (danger have 20GB of free space):
- `huggingface-cli download MaziyarPanahi/BioMistral-7B-GGUF BioMistral-7B.Q5_K_M.gguf --local-dir . --local-dir-use-symlinks False`

# Download Ollama Commandline Tool
- From URL: https://ollama.ai/ - and install for your system
- Check in a terminal: `ollama -v`
- Create the ollama file from model:
- `ollama create biomistral -f Modelfile`

# GPU Acceleration M1 MacOS - use mps
- if using conda: `conda install pytorch torchvision torchaudio -c pytorch-nightly`
- or with pip: `pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu`

# Additional browse for other med gguf M1 Mac models
- for MacOS M1 SOC GPU no Cuda (PC & Linux)
- https://huggingface.co/search/full-text?q=health+gguf&type=model

# Delete Models in Huggingface cache if not needed or HD is full
- show llms: `cd ~/.cache/huggingface/hub`
- `ls -lia`
- `python -m pip install -U "huggingface_hub[cli]"`
- start removing llms: `huggingface-cli delete-cache`
- select via `Space` - don't select no action then enter und choose `Y`(es)

# Test commandline chat
- `ollama run biomistral`

# Exit chat
- stop by typing: `/bye`

# Use env alternatively to conda
- https://sourabhbajaj.com/mac-setup/Python/virtualenv.html
- `pip install virtualenv`
- `virtualenv venv`
- `source venv/bin/activate`

# exit env
- enter `deactivate` in the console

# install dependencies
- `conda activate torch`
- `cd langchain-gemma-ollama-chainlit`
- `pip install -r requirements.txt`
- or more complicated, seperately:
```
pip install langchain
pip install chainlit
pip install openai
```

# run chainlit chat
- `cd langchain-gemma-ollama-chainlit` change directory 
- and run the chat in a browser:
- `chainlit run langchain-gemma-ollama-chainlit.py`

# Example prompts
- What are the symptoms of the common cold?
- What causes the seasonal flu?
- What medication would be prescribed for a headache?

# finetune  gguf models (optional)
- https://github.com/ggerganov/llama.cpp/tree/master/examples/finetune
- https://rentry.org/cpu-lora

# run another model -> failed in my case
- `huggingface-cli download garcianacho/MedLlama-2-7B-GGUF MedLlama-2-7B.q5_K_M.gguf --local-dir . --local-dir-use-symlinks False`

- `ollama create medllama2 -f ModelfileMedLLama`

- `ollama run medlama2`

# Screenshot of demo
![Proof](/proof.png?raw=true "It works")