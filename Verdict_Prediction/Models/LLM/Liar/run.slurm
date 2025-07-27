#!/bin/bash
#SBATCH --partition=contrib-gpuq                    # need to set 'gpuq' or 'contrib-gpuq'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relvant QOS
#SBATCH --job-name=FCLiarQ                     # Job name
#SBATCH --output=/home/mbiswas2/Fact-Checking/Data/ErrorFile/FCLiarQ.out   # Output file
#SBATCH --error=/home/mbiswas2/Fact-Checking/Data/ErrorFile/FCPLiarQ.err    # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12                 # number of cores needed
#SBATCH --gres=gpu:H100.80gb:4                # up to 8; only request what you need
#SBATCH --mem=400GB
#SBATCH --export=ALL
#SBATCH --time=0-24:00:00                   # set to 2hr; please choose carefully

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

module load gnu10

module load python



source  /projects/ouzuner/mbiswas2/Llama/llama_env/bin/activate
#pip install datetime
# pip install torch accelerate transformers



#source llamaenvoz/bin/activate 
#pip install -q -U google-genai
#pip install groq
#pip install json
#pip install -U sentence-transformers
#pip install torch==2.2.2
#pip install torch accelerate transformers json datetime numpy
#pip list
#pip install sentencepiece
#pip install scikit-learn
#pip install scikit-learn matplotlib
#pip install gdown
# pip install pandas
#pip install datasets
#pip install peft
#pip install trl
#pip install textstat
#pip install bitsandbytes
#pip install huggingface_hub
#python llamasft.py
python MultipleFileQwen2.5.py