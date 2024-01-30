This project is the RNA-seq Generator Diffusion of the paper 'RsGD:Generating Cancer-Cell RNA-seq by Diffusion Model and Accelerate Sampling'.

How to train:
Put your own RNA-seq data or use the exist data in the 'datasets' folder.
Reset args.dataset_path parameter in settings.py to load a dataset.
The dataset should be a 2 dim (sample num * gene num) numpy ndarray saved as '.npy' file.
You can edit a script to preprocess the raw data.
Then write the other parameters in settings.py to set the number of training epochs, how often to save checkpoints, etc.
Run train.py to train your model.

How to generate samples:
Place your trained model in the models folder.
Check generate.py, set model_path to be the directory of the model you want to use.
Information about other Settings is provided in the code comments in generate.py.
Run generate.py and check the results. (default saved in 'results' folder)

Other files: (Ignore the follows if you just want to train your model or generate RNA-seq samples.)
loader.py: Load the dataset (.npy) to DataLoader of pytorch.
modeling.py: Neural network structure was writen here.
diffusion.py: Code implementation of the Diffusion Models.