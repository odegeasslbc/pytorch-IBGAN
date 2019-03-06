# pytorch-IBGAN
A pytorch implementation of Information Bottleneck GAN  
Original paper can be find here: https://openreview.net/forum?id=ryljV2A5KX


## Training

- run 'python main.py', for now, please see in the code for variables to adjust different training configurations

## TODO
-[] add argparser for easier configeration  
-[] validate the MSE loss term on z and z_hat, as not directly stated in paper but showed in fig-1  
-[] run on celebA dataset and demo qualitative results  

## Acknowledgement
@misc{
jeon2019ibgan,
title={{IB}-{GAN}: Disentangled Representation Learning with Information Bottleneck {GAN}},
author={Insu Jeon and Wonkwang Lee and Gunhee Kim},
year={2019},
url={https://openreview.net/forum?id=ryljV2A5KX},
}
