# scRDiT: A tool for virtual scRNA-seq data generating based on Diffusion Transformers, and do accelerated sampling by DDIM.

## Data prepare

We put the datasets from the paper as an example in the **'datasets'** directory.
The dataset read by scRDiT must be a matrix stored as **'.npy'** files with shape like ***(sample_size, gene_size)***.
You can edit a script to preprocess the raw data.
Put your own dataset in **'datasets'** directory and copy the file path to `args.dataset_path` in **settings.py**.

`args.dataset_path = 'datasets/file_name.npy'`

## Training
All settings of training process are in  **'settings.py'**.
First, set the experiment's name like:

`args.run_name = 'run-name'`

and the file path of the training checkpoints of this experiment would be `ckpts/run-name/run-name_epochX.pt`.

We have 2 models, `Unet1d()` and `DiT()`, for the noise predictor of DDPMs. Set up the model like:

`args.model = Unet1d()` or `args.model = DiT(input_size: int, patch_size: int, hidden_size: int, depth: int, num_heads: int)`

Other setting parameters' example are as follows:

```python
args.epochs = 1600  # epochs of training
args.batch_size = 16  # depends on your GPU memory size
args.gene_size = 2000  # size of gene set
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.lr = 3e-4  # learning rate
args.save_frequency = 20  # how many epochs to save a checkpoint
args.ckpt = False  # load checkpoint or not
args.ckpt_epoch = 0  # which checkpoint to load
```

Finally, run **'train.py'** to train your own model.

## Generate
Place your trained model in the **'models'** folder. 
Settings of sampling process are alterable in **'generate.py'** .
Here is an example:

```python
    # generating settings
    model_path = 'models/fibroblast_diffusion_ckpt.pt'
    sample_amount = 1024  # number of generated samples
    method = 'ddim'  # use 'ddpm' or 'ddim'
    save = 'results/fibroblast_samples'  # result savepath
    acc_rate = 10  # accelerate rate
    model_structure = Unet1d()
```

Run **'generate.py'** after confirmed all the settings.
The results are stored in the **'results'** directory by default.
