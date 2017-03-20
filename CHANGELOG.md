## 1.0.0.2 ( 2017-03-20 )

Bugfixes:
    - fix a running mean and variance update rule in batch normal 
    - fix a gru bug

Features :
    - support multiple GPU support
    - add regularizer related functions  
    - add summary on/off context

## 1.0.0.1 ( 2017-02-28 )

Bugfixes:
    - fix a missed tensorflow 1.0 refactoring at sg_upconv1d()
    - fix a missed tensorflow 1.0 refactoring at sg_ctc_loss()
    
Features :
    - support VALID padding at sg_upconv() and sg_upconv1d()
    - add zero_out parameter at sg_rnn(), sg_gru() and sg_lstm() 

## 1.0.0.0 ( 2017-02-22 )

Features :
    - scope support at batch normalization

Refactored :
    - adapted to tensorflow 1.0.0
    
     
