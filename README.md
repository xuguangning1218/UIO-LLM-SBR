# IntGNN

Official source code for paper I Got You: The User Intention Oriented Session-Based Recommendation

### Environment Setting
```
conda create -n sbr python=3.10 -y && conda activate sbr
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install tqdm seaborn matplotlib scipy numba scikit-learn datasets transformers peft tensorboard bitsandbytes
pip install vllm==0.8.5
```  
For HearInt
```
conda create -n herint python=3.10 -y && conda activate herint
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
conda install -c conda-forge faiss-gpu -y
pip install tqdm "numpy<2" entmax
pip install dgl -f https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html
pip install "scipy<1.13"
```

###  Source Files Description

```
-- datasets # dataset folder
  -- diginetica # diginetica dataset 
  -- retailRocket_DSAN # retail dataset
  -- Tmall # Tmall dataset
  -- Nowplaying # Nowplaying dataset
-- figure # figure provider
  -- model.jpg # architecture of UIO-SBR model 
-- model # main code of the project
  -- DUIGNN.py # the DUI-GNN module implementation
  -- IUIGNN.py # the IUI-GNN module implementation
  -- RES.py # the RES module implementation
  -- UIOSBR.py # the model of UIO-SBR
main.py # the running script of the UIO-SBR
```

### Run

When the environment and datasets are cloned, you can train the UIO-SBR by running the following code:

```
cd ./UI-SBR
python main.py
```
