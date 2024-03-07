#!/bin/bash
flag=0
file=${0%%.*}
name=${file##*/}
# time=$(date +%Y%m%d%H%M%S)
while [ $flag -eq 0 ]
do
    count=0
    for i in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    do
        if [ $i -lt 8000 ]
        then
            echo 'GPU' $count ' is avaiable, start training...'
            CUDA_VISIBLE_DEVICES=$count \
            nohup python train.py \
            --dataset_name 'ICFG-PEDES' \
            --root_dir '/data0/data_ccq/ICFG/' \
            --output_dir '/data1/ccq/multimodality-ICFG' \
            --img_aug \
            --name 'sketch2_add-fusion-twofocal-1-35-fusion-itcloss_05kl-text-label' \
            --fusion_way 'add' \
            --batch_size 64 \
            --pa 0.1 \
            --pretrain_choice 'ViT-B/16' \
            --loss_names 'itc' \
            --lrscheduler 'cosine' \
            --target_lr 0 \
            --num_epoch 60 \
            --al 1.0 \
            --ga 3.5 \
            --klp 0.5 \
            --focal_three_fusion_loss3 \
            > scripts/ICFG-PEDES/ViT/nohup.out 
            _pid=$!
            echo "training pid: $_pid"
            flag=1
            break
        fi
        count=$(($count+1))    
    done
    sleep 20
done

#--name $name \
# --root_dir '/data0/data_ccq/CUHK-PEDES/' \

# 这个Shell脚本主要用于在有NVIDIA GPU的环境中自动化地启动机器学习或深度学习训练任务。让我们逐步分析这个脚本的功能：

# 1. `#!/bin/bash`: 这一行称为shebang，告诉系统这个脚本应该使用bash解释器执行。

# 2. 初始化变量`flag`为0，用于控制外层`while`循环。

# 3. `file=${0%%.*}`和`name=${file##*/}`这两行用于处理脚本文件名，移除文件扩展名，然后从文件路径中提取出文件的基本名字，但实际上这两个变量在脚本中并未使用。

# 4. 注释掉的`time=$(date +%Y%m%d%H%M%S)`可以用来生成一个时间戳，但在此脚本中它被注释掉了，所以没有被使用。

# 5. `while [ $flag -eq 0 ]`：一个循环，只要`flag`等于0就会持续执行。这个循环用来不断检查GPU，直到找到一个可用的GPU。

# 6. `for i in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)`: 这个循环通过`nvidia-smi`命令查询每个GPU当前使用的内存量，返回值没有标题头，也没有单位。

# 7. 在循环内部，如果发现有GPU的使用内存小于8000MB（表示这个GPU相对空闲），则打印一条消息表示该GPU可用于训练，并启动训练过程。训练过程使用的是`nohup python train.py`命令，以便在后台运行，并将输出重定向到指定的文件中。这个命令会启动一个Python训练脚本，传递给它一系列参数，如数据集名称、根目录、输出目录、图像增强开关、训练配置名称、融合方式、批处理大小等。

# 8. 训练启动后，脚本会打印出训练进程的PID，并将`flag`设置为1，这将导致外层循环结束，脚本执行完成。

# 9. 如果所有GPU的内存使用都超过8000MB，则脚本会等待20秒后再次检查，这个过程会不断重复，直到找到一个满足条件的GPU。

# 这个脚本主要用于在多GPU环境下自动监测和利用空闲的GPU资源来启动特定的训练任务，提高资源的利用率。