# IntGNN

Official source code for paper I Got You: The User Intention Oriented Session-Based Recommendation

### Environment Setting
```
pytorch==1.12.0
numpy==1.20.3
tqdm==4.61.2
torchvision==0.13.0
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
