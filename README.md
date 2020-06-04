# ACAL
Implementation of ACAL method -- https://arxiv.org/pdf/1807.00374.pdf

CycleGAN code reference from [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. 

To train: 

``` 
python train.py --dataroot ./datasets/numbers --name numbers_cyclegan --model acal --source_model /home/avinash/avinash/src/saved_models/source_model.pth --dataset_mode acal --batch_size 64 --ngf 16 --ndf 16 --netD n_layers --n_layers_D 2 --netG resnet_6blocks --load_size 32 --crop_size 32 --num_threads 16 --preprocess none --no_flip --n_epochs 5 --n_epochs_decay 0 --output_nc 1
 ```
