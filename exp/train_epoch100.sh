# python exp/train.py --epochs 100 --loss_func bce  
# python exp/train.py --epochs 100 --loss_func tversky
 


# python exp/trainer/train_bolivia.py --epochs 100 --loss_func bce 
# python exp/trainer/train_bolivia.py --epochs 100 --loss_func focal
# python exp/trainer/train_bolivia.py --epochs 100 --loss_func tversky

python exp/multimodal_trainer.py --epochs 100 --loss_func tversky --batch_size 4 --prithvi_finetune_ratio 1
python exp/multimodal_trainer.py --epochs 50 --loss_func tversky --batch_size 4 --prithvi_finetune_ratio 1
python exp/multimodal_trainer.py --epochs 20 --loss_func tversky --batch_size 4 --prithvi_finetune_ratio 1

 