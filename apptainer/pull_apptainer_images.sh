module load apptainer/1.2.4
cd ~/projects/def-mbowling/aorenste/
mkdir -p apptainer_imgs
cd apptainer_imgs
apptainer pull docker://adrianorenstein/pytorch:2.4.0-python3.12.5-devell && \
    apptainer pull docker://adrianorenstein/minigrid_pytorch:2.4.0-python3.12.5-devell & \
    apptainer pull docker://adrianorenstein/atari_pytorch:2.4.0-python3.12.5-devell &
wait
echo "done"
