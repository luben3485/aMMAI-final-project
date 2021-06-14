id=$1
free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)

echo $free_mem

while [ $free_mem -lt 7000 ]; do
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)
    echo $free_mem
    sleep 1
done
python train.py --task csfsl_multi --model ResNet10  --method e_protonet_fc --n_shot 5 --train_aug --stop_epoch 200
