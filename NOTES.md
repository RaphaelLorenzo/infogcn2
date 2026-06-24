python main.py --half=True --batch_size=64 --test_batch_size=64 \
    --step 50 60 --num_epoch=70 --num_worker=4 --dataset=ntu --num_class=60 \
    --datacase=NTU60_CS --weight_decay=0.0003 --num_person=1 --num_point=25 --graph=graph.ntu_rgb_d.Graph \
    --feeder=feeders.feeder_ntu.Feeder --base_lr 1e-1 --base_channel 64 \
    --window_size 52 --lambda_1=1e-0 --lambda_2=1e-1 --lambda_3=1e-3 --n_step 3

python main.py --half=True --batch_size=64 --test_batch_size=64 \
    --step 50 60 --num_epoch=70 --num_worker=4 --dataset=ntu --num_class=60 \
    --datacase=NTU60_CS --weight_decay=0.0003 --num_person=1 --num_point=25 --graph=graph.ntu_rgb_d.Graph \
    --feeder=feeders.feeder_ntu.Feeder --base_lr 1e-1 --base_channel 64 \
    --window_size 52 --lambda_1=1e-0 --lambda_2=1e-1 --lambda_3=1e-3 --n_step 3 --partial_load 0.2 --seed 42

python main.py --half=True --batch_size=64 --test_batch_size=64 \
    --step 50 60 --num_epoch=70 --num_worker=4 --dataset=ntu --num_class=60 \
    --datacase=NTU60_CS --weight_decay=0.0003 --num_person=1 --num_point=25 --graph=graph.ntu_rgb_d.Graph \
    --feeder=feeders.feeder_ntu.Feeder --base_lr 1e-1 --base_channel 64 \
    --window_size 52 --lambda_1=1e-0 --lambda_2=1e-1 --lambda_3=1e-3 --n_step 3 --partial_load 0.2 --seed 43

python main.py --half=True --batch_size=64 --test_batch_size=64 \
    --step 50 60 --num_epoch=70 --num_worker=4 --dataset=ntu --num_class=60 \
    --datacase=NTU60_CS --weight_decay=0.0003 --num_person=1 --num_point=25 --graph=graph.ntu_rgb_d.Graph \
    --feeder=feeders.feeder_ntu.Feeder --base_lr 1e-1 --base_channel 64 \
    --window_size 52 --lambda_1=1e-0 --lambda_2=1e-1 --lambda_3=1e-3 --n_step 3 --partial_load 0.2 --seed 44

python main.py --half=True --batch_size=64 --test_batch_size=64 \
    --step 50 60 --num_epoch=70 --num_worker=4 --dataset=ntu --num_class=60 \
    --datacase=NTU60_CS --weight_decay=0.0003 --num_person=1 --num_point=25 --graph=graph.ntu_rgb_d.Graph \
    --feeder=feeders.feeder_ntu.Feeder --base_lr 1e-1 --base_channel 64 \
    --window_size 52 --lambda_1=1e-0 --lambda_2=1e-1 --lambda_3=1e-3 --n_step 3 --partial_load 0.2 --seed 45

python main.py --half=True --batch_size=64 --test_batch_size=64 \
    --step 50 60 --num_epoch=70 --num_worker=4 --dataset=ntu --num_class=60 \
    --datacase=NTU60_CS --weight_decay=0.0003 --num_person=1 --num_point=25 --graph=graph.ntu_rgb_d.Graph \
    --feeder=feeders.feeder_ntu.Feeder --base_lr 1e-1 --base_channel 64 \
    --window_size 52 --lambda_1=1e-0 --lambda_2=1e-1 --lambda_3=1e-3 --n_step 3 --partial_load 0.2 --seed 46