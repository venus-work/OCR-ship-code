python demo.py video --config /mnt/data1/ngapt30/nanodet/config/nanodet-plus-m-1.5x_416.yml --model /mnt/data1/ngapt30/nanodet/model/nanodet-plus-m-1.5x_416_checkpoint.ckpt --path /mnt/data1/ngapt30/nanodet/IMG_1376.MOV --save_result 

python demo2.py image --config /mnt/data1/ngapt30/nanodet/config/nanodet-plus-m-1.5x_416.yml --model /mnt/data1/ngapt30/nanodet/model/nanodet-plus-m-1.5x_416_checkpoint.ckpt --path /mnt/data1/ngapt30/nanodet/test.png


export LD_LIBRARY_PATH=/mnt/data2/miniconda3/envs/ngapt_nanodet/lib/python3.8/site-packages/nvidia/cudnn/lib
python demo.py video --config /mnt/data1/ngapt30/nanodet/config/nanodet-plus-m-1.5x_416.yml --model /mnt/data1/ngapt30/nanodet/model/nanodet-plus-m-1.5x_416_checkpoint.ckpt --path /mnt/data1/ngapt30/nanodet/IMG_1376.MOV --save_result
