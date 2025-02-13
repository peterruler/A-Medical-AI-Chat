# A Medical AI Chat

## What this repo is about:
- Install and use a medical GPT chat with custom model files and model downloads from Hugging Face.
- Run a Mistral programming code generation GPT chat from Ollama's library provided by default.

## Prerequisites
- Have a fast, modern PC (best with NVIDIA CUDA support and >= 12 GB graphics RAM), have patience, or use an M1 Mac or an AI web hosting with available GPU (e.g., Azure).
- Install Ollama: [https://ollama.com](https://ollama.com) - install for your system.
- Choose the models for your domain: [https://ollama.com/library](https://ollama.com/library).
- Note: Mistral is no longer available in German by default.
  - Run `ollama run mistral` to download the model and run it. Example prompt: `Schreibe mir einen Python Flask Server, der ein Hello World zurück gibt`.
  - For German, use the custom model from: [https://huggingface.co/TheBloke/em_german_leo_mistral-GGUF](https://huggingface.co/TheBloke/em_german_leo_mistral-GGUF).

## Based on this YouTube clip:
- [https://www.youtube.com/watch?v=bANziaFj_sA](https://www.youtube.com/watch?v=bANziaFj_sA)

## Chainlit Documentation
- [https://docs.chainlit.io/get-started/overview](https://docs.chainlit.io/get-started/overview)

## Official Mistral Website with AI GPT comparison
- [https://mistral.ai/news/mistral-large](https://mistral.ai/news/mistral-large)

## Example Ollama run Mistral with prompt:
![Proof](/flask.png?raw=true "flask")

## Useful tips and commands with Ollama:
- Bookmark the Ollama terminal in MacOS Dock / Windows taskbar.
- Check in a terminal: `ollama -v`.
- `ollama list` lists the installed models.
- `/?` for help.
- `ollama rm <Modelname>` deletes an installed model.
- Exit command line chat by typing: `/bye`.

## Additional browse for other medical GGUF M1 Mac models:
- For MacOS M1 SOC GPU (no CUDA for PC & Linux): [https://huggingface.co/search/full-text?q=health+gguf&type=model](https://huggingface.co/search/full-text?q=health+gguf&type=model).

## Test command line chat:
- Download and install via model file first, then run: `ollama run biomistral1` (7B - German). Example prompt: `Was ist der Haupt-Wirkstoff in Aspirin?`.
![Proof](/aspirin.png?raw=true "biomistral1")

## Use as a Chainlit chat:
- To use Ollama as a Python Chainlit chat, follow these steps. Result can be seen in the following screenshot:
![Proof](/flask2.png?raw=true "flask")

## Installation of Python Conda environment (optional):
- Install Miniconda first: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).
- Then run:
  ```sh
  conda create --name torch python=3.9
  conda activate torch
  pip install -r ./langchain-gemma-ollama-chainlit/requirements.txt
  python -m ipykernel install --user --name torch --display-name "Python 3.9 (torch)"
  pip install transformers==4.20.0
  pip install googletrans==4.0.0-rc1 (optional)
  conda install torch -c pytorch-nightly
  ```

## GPU Acceleration M1 MacOS - use mps (optional)
- if using conda: `conda install pytorch torchvision torchaudio -c pytorch-nightly`
- or with pip: `pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu`

## Detailed explanation, create a pytorch conda environment (optional)
- Create new env from Jeff Heaton Youtube / Github pages
- MacOS M1 https://www.youtube.com/watch?v=VEDy-c5Sk8Y&t=380s
- other: https://www.youtube.com/results?search_query=jeff+heaton+pytorch+installation

- activate env by typing in console:
- `conda activate torch`

## Load custom Model - ggml-model-Q8_0.gguf - 7B Model - in german (optional)
- In case a model is not listed on https://ollama.com/library:
- First install the huggingface-cli:
- `python -m pip install -U "huggingface_hub[cli]"`
- `pip3 install huggingface-hub`
- Load the model file to local cache (danger, have at least 6 GB of free space):
- `huggingface-cli download BioMistral/BioMistral-7B-GGUF ggml-model-Q8_0.gguf --local-dir . --local-dir-use-symlinks False`
- `ollama create biomistral1 -f Modelfile`
- `ollama run biomistral1`

## Install a model thats not listed in the ollama directory (/library)
- `cd langchain-gemma-ollama-chainlit` change directory 
- replace in langchain-gemma-ollama-chainlit-de.py: `model = Ollama(model="gemma:2b")` - instead of gemma:2b with your model name e.g. `mistral` or custom installation `biomistral1`.
- edit Modefile with correct template
- `ollama create biomistral1 -f Modelfile`

## On manifest missing error (optional)
- Create the ollama file from custom model:
- (you can initialize, in case of manifest missing error do a `ollama run mistral` first)


## Run the chat in a browser:
- `cd langchain-gemma-ollama-chainlit` change directory 
- `chainlit run langchain-gemma-ollama-chainlit-de.py` -> medical
- `chainlit run langchain-mistral-ollama-chainlit.py` -> code


![Proof](/aspirin2.png?raw=true "biomistral1")

## Delete Models in Huggingface cache if not needed or HD is full
- show llms: `cd ~/.cache/huggingface/hub`
- `ls -lia`
- `python -m pip install -U "huggingface_hub[cli]"`
- start removing llms: `huggingface-cli delete-cache`
- select via `Space` - don't select no action then enter und choose `Y`(es)

## Example prompts (english)
- What are the symptoms of the common cold?
- What causes the seasonal flu?
- What medication would be prescribed for a headache?

## Example prompts (german) - with Biomistral trained and finetuned on medical domain:
- Was sind die Symptome einer Erkältung?
- Was verursacht die saisonale Grippe?
- Welche Medikamente würden gegen Kopfschmerzen verschrieben?

## Mistral Model (Programminglanguages better use ollama run codestral, very slow): ollama run mistral
- Wie importiert man ein Package in Java?
- Wie definiert man einen Type mit einem String und einem boolean in Typescript?
- Schreibe mir einen Python Flask Server, der ein Hello World zurück gibt

## Finetune gguf models (very optional, for experts only)
- https://github.com/ggerganov/llama.cpp/tree/master/examples/finetune
- https://rentry.org/cpu-lora

## Screenshot of demo of another prompt:

![Proof](/mistral-code.png?raw=true "Mistral in english only")