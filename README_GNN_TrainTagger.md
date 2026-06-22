[Original Train Tagger Repo](https://github.com/CMS-L1T-Jet-Tagging/TrainTagger)

## What contains what:
### Analysis Folder- 
	- Contains emulation_data_1 -> Simplified HLS4ML simulation data from hls4ml emulator on CMSSW for gnn_tagger_v2
	- baseline.h5- Tensorflow baseline model(Deepset model) 
	- jetTuple_extended_5.root - HLS4ML simulation data from hls4ml emulator on CMSSW for gnn_tagger_v2
	- saved_model.h5 - Tensorflow GNN model(Only Graph Conv applied model) 
### firmware Folder-
	- L1TGNNJetModel_V1 - Contains C++ code and extras for gnn_tagger_v2 model
	- L1TSC4NGJetModel_test - Contains C++ code and extras for baseline model.
### output2 Folder-
	- baseline - Contains Tensorflow model for Deepset 
	- gnn_tagger_v1 - Contains Tensorflow model for GNN v1 model
	- gnn_tagger_v2 - Contains Tensorflow model for GNN v2 model
### part2.ipynb - 
	- To analyse the outputs
### requirements.txt - 
	- Contains requirements that are needed to be created in python3.11 env
### tagger/model/GNNModel.py - 
	- Contains GNN model
### tagger/model/configs/gnn_model.yaml - 
	- Contains GNN model configs
### tagger/data/tools.py 
	- def _build_X_and_ADJ(split, input_vars, knn): makes graphs from jet data with checking min(n_cand_in_jet,5) neighbors. Uses pairwise dR to find neighbors. Also row-normalize -> D^-1 A (each row sums to 1)
	- def load_data(outdir, percentage, test_ratio=0.1, fields=None, build_graph=False, knn=5): Returns train and test data for All types of model, but operates differently for GNN model
### tagger/model/GNNModel.py 
	- Makes GNN model based on QkerasModel, and quantises it. Also have code to convert it into C++ code via firmware_convert() function.