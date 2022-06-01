import os

import torch 

import pytorch_lightning as pl


from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers


from hdc2021.utils.blurred_dataset import MultipleBlurredDataModule
from hdc2021.deblurrer.MultiScale_GD import MultiScaleReconstructor

blurring_step = 19
dataset = MultipleBlurredDataModule(batch_size=2, blurring_step=blurring_step)
dataset.prepare_data()
dataset.setup()

step_to_radius = {4 : 0.015,
                  9 : 0.03,
                  14: 0.043,  
                  19: 0.0875}

checkpoint_callback = ModelCheckpoint(
    save_last=True,
    dirpath=None,
    save_top_k=1,
    verbose=True,
    monitor='val_ocr_acc',
    mode='max',
)

base_path = '/localdata/AlexanderDenker/deblurring_experiments'
experiment_name = 'multi_scale'
path_parts = [base_path, experiment_name, "step_" + str(blurring_step)]
log_dir = os.path.join(*path_parts)
tb_logger = pl_loggers.TensorBoardLogger(log_dir)

trainer_args = {'accelerator': 'ddp',
                'gpus': [0],
                'default_root_dir': log_dir,
                'callbacks': [checkpoint_callback],
                'benchmark': False,
                'fast_dev_run': False,
                'gradient_clip_val': 1.0,
                'logger': tb_logger,
                'log_every_n_steps': 10,
                'accumulate_grad_batches': 3, 
                'multiple_trainloader_mode': 'min_size',
                #'limit_train_batches': 0.1,
                'auto_scale_batch_size': 'binsearch'}#,
                #'accumulate_grad_batches': 6}#,}
                # 'log_gpu_memory': 'all'} # might slow down performance (unnecessary uses only the output of nvidia-smi)


reconstructor = MultiScaleReconstructor(blurring_step=blurring_step, radius=step_to_radius[blurring_step], sigmas=[5e-3, 6e-3, 4e-3, 5e-3,4e-3],
                                        step_sizes=[1.1, 0.8, 0.8, 0.7, 0.3],
                                        channels = [[16, 64, 128, 256],
                                                       [16, 64, 128, 128, 256],
                                                       [16, 64, 64, 128, 256],
                                                       [8, 32, 32, 64, 128, 128],
                                                       [4, 8, 64, 64, 64, 128,128,128]],
                                        skip_channels =  [[16, 64, 128, 256],
                                                       [16, 64, 128, 128, 256],
                                                       [16, 64, 64, 128, 256],
                                                       [8, 32, 32, 64, 128, 128],
                                                       [4, 8, 64, 64, 64, 128,128,128]],
                                        kernel_size = [3, 3, 3, 3, 3],
                                        init_x=True,
                                        use_sigmoid=False,
                                        n_memory=3)

trainer = pl.Trainer(max_epochs=300, 
                    **trainer_args)
trainer.fit(reconstructor, datamodule=dataset)






