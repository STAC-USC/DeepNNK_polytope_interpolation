##### Neighbors experiment
python main.py --experiment neighbor --use_gpu --mode train --regularize #--layer_size 16
python main.py --experiment neighbor --use_gpu --mode test --regularize #--layer_size 16
#python main.py --experiment neighbor --nouse_gpu --mode calibrate --data_type train --regularize --knn_param 25 #--layer_size 16
#python main.py --experiment neighbor --nouse_gpu --mode calibrate --data_type test --regularize --knn_param 25 #--layer_size 16
#python main.py --experiment neighbor --nouse_gpu --mode calibrate --data_type train --regularize --knn_param 50 #--layer_size 16
#python main.py --experiment neighbor --nouse_gpu --mode calibrate --data_type test --regularize --knn_param 50 --layer_size 16
#python main.py --experiment neighbor --nouse_gpu --mode calibrate --data_type train --regularize --knn_param 75 #--layer_size 16
#python main.py --experiment neighbor --nouse_gpu --mode calibrate --data_type test --regularize --knn_param 75 #--layer_size 16
#python main.py --experiment neighbor --use_gpu --mode plot --data_type test --regularize
#python main.py --experiment neighbor --use_gpu --mode SVM  --cross_validation 5 --regularize #--layer_size 16
