python -W ingore train.py --cuda_devices "0" --epochs 100 -- batch_size 24 --physio_net --dataset_path "data/training2017" --network_config "ECGCNN_S"
python -W ingore train.py --cuda_devices "0" --epochs 100 -- batch_size 24 --physio_net --dataset_path "data/training2017" --network_config "ECGCNN_M"
python -W ingore train.py --cuda_devices "0" --epochs 100 -- batch_size 24 --physio_net --dataset_path "data/training2017" --network_config "ECGCNN_L"
python -W ingore train.py --cuda_devices "0" --epochs 100 -- batch_size 24 --physio_net --dataset_path "data/training2017" --network_config "ECGAttNet_S"
python -W ingore train.py --cuda_devices "0" --epochs 100 -- batch_size 24 --physio_net --dataset_path "data/training2017" --network_config "ECGAttNet_M"
python -W ingore train.py --cuda_devices "0" --epochs 100 -- batch_size 24 --physio_net --dataset_path "data/training2017" --network_config "ECGAttNet_L"

