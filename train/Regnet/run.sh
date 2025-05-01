HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 )
docker run  -d --rm  -p 8888:8888 \
    -v ~/mltrain-chi/workspace_ray:/home/jovyan/work/ \
    -e RAY_ADDRESS=http://${HOST_IP}:8265/ \
    --name jupyter \
    jupyter-ray



HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 )
docker run  -d --rm  -p 8888:8888 \
    --gpus all \
    --shm-size 16G \
    -v ~/mltrain-chi/workspace_mlflow:/home/jovyan/work/ \
    -v food11:/mnt/ \
    -e MLFLOW_TRACKING_URI=http://${HOST_IP}:8000/ \
    -e FOOD11_DATA_DIR=/mnt/Food-11 \
    --name jupyter \
    jupyter-mlflow


docker run -d --rm -p 8888:8888 \
    --gpus all \
    --shm-size 16G \
    -v ~/train:/home/jovyan/work/ \
    -e MLFLOW_TRACKING_URI=http://${HOST_IP}:8000/ \
    -e RAY_ADDRESS=http://${HOST_IP}:8265/ \
    --name jupyter-all \
    jupyter-all


docker run -d --rm -p 8888:8888 \
    --gpus all \
    --shm-size 16G \
    -v ~/train:/home/jovyan/work/ \
    -v Project3Data:/mnt/ \
    -e MLFLOW_TRACKING_URI=http://${HOST_IP}:8000/ \
    -e RAY_ADDRESS=http://${HOST_IP}:8265/ \
    --name jupyter-all \
    jupyter-all

docker run --rm -it -v Project3Data:/mnt alpine ls -l /mnt/AiVsHuman/


http://192.5.86.177:8000

docker run -d --rm -p 8888:8888 \
    --gpus all \
    --shm-size 16G \
    -v /home/cc/work:/home/jovyan/work/ \
    --name jupyter \
    jupyter

http://129.114.109.50:8888/lab?token=19604a7ef3be26b86ee94785050945d3253053ae22e13ecf



python -m torch.distributed.run --nproc_per_node=2 train.py 