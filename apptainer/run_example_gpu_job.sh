# salloc --time=0:10:0 --mem-per-cpu=4G --ntasks=4 --account=def-mbowling --nodes=1 --gpus-per-node=v100:1
# nvidia-smi 
# should read out a v100

# Load apptainer and copy over the container
module load apptainer/1.2.4

# apptainer image
IMAGE_NAME=${1:-minigrid_pytorch_2.4.0-python3.12.5-devell.sif}

cp ~/projects/def-mbowling/aorenste/apptainer_images/$IMAGE_NAME $SLURM_TMPDIR/
cd $SLURM_TMPDIR/

# Set up SOCKS5 proxy
echo "Setting up SOCKS5 proxy..."
ssh -q -N -T -f -D 8888 `echo $SSH_CONNECTION | cut -d " " -f 3`
export ALL_PROXY=socks5h://localhost:8888

# clone an example repo
git config --global http.proxy $ALL_PROXY
git clone --quiet https://github.com/AdrianOrenstein/docker-pytorch-template.git $SLURM_TMPDIR/project

# sanity check we have access to the pip packages
apptainer run \
    --env ALL_PROXY=$ALL_PROXY \
    --env APPEND_PATH=$SLURM_TMPDIR/project \
    $SLURM_TMPDIR/$IMAGE_NAME python -c "import torch, minigrid; print(f'{torch.__version__=}\t{minigrid.__version__=}')"

# check that we can use torch on the GPU
apptainer run \
    --nv \
    --env ALL_PROXY=$ALL_PROXY \
    --env APPEND_PATH=$SLURM_TMPDIR/project \
    $SLURM_TMPDIR/$IMAGE_NAME python -c "import torch; a=torch.ones(50).cuda(); out=(a+a)*2; print(out.shape, out)"

# check that we can run python modules on the cloned repo
apptainer run \
    --env ALL_PROXY=$ALL_PROXY \
    --env APPEND_PATH=$SLURM_TMPDIR/project \
    $SLURM_TMPDIR/$IMAGE_NAME python -m black $SLURM_TMPDIR/project
