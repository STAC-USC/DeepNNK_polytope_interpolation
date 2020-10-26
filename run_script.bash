##### Neighbors experiment
#python main.py --experiment neighbor --use_gpu --mode train --regularize #--layer_size 16
python main.py --experiment neighbor --use_gpu --mode test --regularize #--layer_size 16
#python main.py --experiment neighbor --nouse_gpu --mode calibrate --data_type train  --knn_param 25 #--layer_size 16
#python main.py --experiment neighbor --nouse_gpu --mode calibrate --data_type test  --knn_param 25 #--layer_size 16
#python main.py --experiment neighbor --nouse_gpu --mode calibrate --data_type train  --knn_param 50 #--layer_size 16
#python main.py --experiment neighbor --nouse_gpu --mode calibrate --data_type test  --knn_param 50 --layer_size 16
#python main.py --experiment neighbor --nouse_gpu --mode calibrate --data_type train  --knn_param 75 #--layer_size 16
#python main.py --experiment neighbor --nouse_gpu --mode calibrate --data_type test  --knn_param 75 #--layer_size 16
#python main.py --experiment neighbor --use_gpu --mode plot --data_type test
#python main.py --experiment neighbor --use_gpu --mode calibrate  --cross_validation 5 #--layer_size 16
