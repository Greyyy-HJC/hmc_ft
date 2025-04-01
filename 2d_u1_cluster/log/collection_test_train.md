## Test train on 8x8 lattice

```bash
python train.py --lattice_size 8 --min_beta 3.0 --max_beta 3.0 --n_epochs 4 --batch_size 128 --n_subsets 8 --n_workers 32 --if_check_jac True
```

### No jit, 3 CNN, No optimization of torch.roll

```
Training data shape: torch.Size([102, 2, 8, 8])
Testing data shape: torch.Size([26, 2, 8, 8])

>>> Training the model at beta =  3

Training epochs:   0%|          | 0/4 [00:00<?, ?it/s]

Epoch 1/4:   0%|          | 0/1 [00:00<?, ?it/s]

Epoch 1/4: 100%|██████████| 1/1 [01:27<00:00, 87.70s/it]

Jacobian log determinant by hand is -1.17e-01
Jacobian log determinant by autograd is -1.17e-01
Jacobian is all good


Evaluating:   0%|          | 0/1 [00:00<?, ?it/s]

Evaluating: 100%|██████████| 1/1 [00:24<00:00, 24.53s/it]

Training epochs:  25%|██▌       | 1/4 [01:53<05:40, 113.65s/it]
Jacobian log determinant by hand is -2.51e-01
Jacobian log determinant by autograd is -2.51e-01
Jacobian is all good
Epoch 1/4 - Train Loss: 37.726879 - Test Loss: 22.078512
```

### No jit, 1 CNN, No optimization of torch.roll

```
Training data shape: torch.Size([102, 2, 8, 8])
Testing data shape: torch.Size([26, 2, 8, 8])

>>> Training the model at beta =  3

Training epochs:   0%|          | 0/4 [00:00<?, ?it/s]

Epoch 1/4:   0%|          | 0/1 [00:00<?, ?it/s]

Epoch 1/4: 100%|██████████| 1/1 [08:31<00:00, 511.15s/it]

Jacobian log determinant by hand is -1.48e-01
Jacobian log determinant by autograd is -1.48e-01
Jacobian is all good
Evaluating:   0%|          | 0/1 [00:00<?, ?it/s]

Evaluating: 100%|██████████| 1/1 [07:06<00:00, 426.71s/it]

Training epochs:  25%|██▌       | 1/4 [15:44<47:12, 944.00s/it]
Jacobian log determinant by hand is -4.34e+00
Jacobian log determinant by autograd is -4.34e+00
Jacobian is all good
Epoch 1/4 - Train Loss: 37.576542 - Test Loss: 17.625326
```

### Use jit, 3 CNN, No optimization of torch.roll

```
Training data shape: torch.Size([102, 2, 8, 8])
Testing data shape: torch.Size([26, 2, 8, 8])

>>> Training the model at beta =  3

Training epochs:   0%|          | 0/4 [00:00<?, ?it/s]

Epoch 1/4:   0%|          | 0/1 [00:00<?, ?it/s]

Epoch 1/4: 100%|██████████| 1/1 [01:24<00:00, 84.96s/it]

Jacobian log determinant by hand is -3.30e-01
Jacobian log determinant by autograd is -3.30e-01
Jacobian is all good


Evaluating:   0%|          | 0/1 [00:00<?, ?it/s]

Evaluating: 100%|██████████| 1/1 [00:23<00:00, 23.47s/it]

Jacobian log determinant by hand is -5.29e-01
Jacobian log determinant by autograd is -5.29e-01
Jacobian is all good
Epoch 1/4 - Train Loss: 37.515778 - Test Loss: 21.776953


Epoch 2/4:   0%|          | 0/1 [00:00<?, ?it/s]

Epoch 2/4: 100%|██████████| 1/1 [03:14<00:00, 194.70s/it]

Jacobian log determinant by hand is -6.67e-01
Jacobian log determinant by autograd is -6.67e-01
Jacobian is all good


Evaluating:   0%|          | 0/1 [00:00<?, ?it/s]   

Evaluating: 100%|██████████| 1/1 [00:23<00:00, 23.14s/it]

Training epochs:  50%|█████     | 2/4 [05:31<05:51, 175.52s/it]
Jacobian log determinant by hand is -7.27e-01
Jacobian log determinant by autograd is -7.27e-01
Jacobian is all good
Epoch 2/4 - Train Loss: 37.149242 - Test Loss: 21.540144


Epoch 3/4:   0%|          | 0/1 [00:00<?, ?it/s]

Epoch 3/4: 100%|██████████| 1/1 [03:17<00:00, 197.14s/it]

Jacobian log determinant by hand is -7.72e-01
Jacobian log determinant by autograd is -7.72e-01
Jacobian is all good


Evaluating:   0%|          | 0/1 [00:00<?, ?it/s]

Evaluating: 100%|██████████| 1/1 [00:24<00:00, 24.25s/it]

Jacobian log determinant by hand is -9.29e-01
Training epochs:  75%|███████▌  | 3/4 [09:16<03:18, 198.16s/it]Jacobian log determinant by hand is -9.29e-01
Jacobian log determinant by autograd is -9.29e-01
Jacobian is all good
Epoch 3/4 - Train Loss: 36.778904 - Test Loss: 21.300198
```