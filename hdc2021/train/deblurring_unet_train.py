
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from pathlib import Path

import pytorch_lightning as pl
import time

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers


#from hdc2021_challenge.utils.blurred_dataset import BlurredDataModule
#from hdc2021_challenge.deblurrer.UNet_deblurrer import UNetDeblurrer

from hdc2021.utils.blurred_dataset import MultipleBlurredDataModule, SimulatedDataModule, BlurredDataModule
from hdc2021.deblurrer.UNet_deblurrer import UNetDeblurrer


blurring_step = 14

# Training on synthetic data
dataset_synthetic = SimulatedDataModule(batch_size=1,
                                        blurring_step=blurring_step,
                                        num_data_loader_workers=8,
                                        create_measurements=True,
                                        num_data=(500, 100),
                                        noise_gt = 10*255,
                                        noise_measurement = 255)
dataset_synthetic.prepare_data()
dataset_synthetic.setup()

checkpoint_callback = ModelCheckpoint(
    dirpath=None,
    filename='synthetic-data-{epoch}-{val_loss:.2f}',
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
    #prefix=''
)

base_path = '/localdata/junickel/hdc2021'
experiment_name = 'unet_deblurring'
#blurring_step = "step_" + str(blurring_step)
path_parts = [base_path, experiment_name, "step_" + str(blurring_step)]
log_dir = os.path.join(*path_parts)
tb_logger = pl_loggers.TensorBoardLogger(log_dir)

trainer_args = {'accelerator': 'ddp',
                'gpus': 2,
                'default_root_dir': log_dir,
                'callbacks': [checkpoint_callback],
                'benchmark': False,
                'fast_dev_run': False,
                'gradient_clip_val': 1.0,
                'logger': tb_logger,
                'log_every_n_steps': 20,
                'auto_scale_batch_size': 'binsearch'}#,
                #'accumulate_grad_batches': 6}#,}
                # 'log_gpu_memory': 'all'} # might slow down performance (unnecessary uses only the output of nvidia-smi)

reconstructor = UNetDeblurrer(blurring_step=blurring_step)

trainer = pl.Trainer(max_epochs=150, **trainer_args)

trainer.fit(reconstructor, datamodule=dataset_synthetic)

# Training on real data
dataset_real = BlurredDataModule(batch_size=1,
                                 blurring_step=blurring_step, 
                                 num_data_loader_workers=8,
                                 shift_bg=False)
dataset_real.prepare_data()
dataset_real.setup()

time.sleep(30)

# initialize pretrained network

pathCheckpoint = checkpoint_callback.best_model_path
print(pathCheckpoint)
reconstructor_pretrained = reconstructor.load_from_checkpoint(pathCheckpoint,blurring_step=blurring_step) 

'''
reconstructor = UNetDeblurrer(blurring_step=blurring_step)
reconstructor_pretrained = reconstructor.load_from_checkpoint(Path('/localdata/junickel/hdc2021/unet_deblurring/step_14/default/version_2/checkpoints/synthetic-data-epoch=9-val_loss=0.01.ckpt'),blurring_step=blurring_step) 

base_path = '/localdata/junickel/hdc2021'
experiment_name = 'unet_deblurring'
#blurring_step = "step_" + str(blurring_step)
path_parts = [base_path, experiment_name, "step_" + str(blurring_step)]
log_dir = os.path.join(*path_parts)
tb_logger = pl_loggers.TensorBoardLogger(log_dir)
'''

checkpoint_callback = ModelCheckpoint(
    dirpath=None,
    filename='real-data-{epoch}-{val_loss:.2f}',
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
    #prefix=''
)


base_path = '/localdata/junickel/hdc2021'
experiment_name = 'unet_deblurring'
#blurring_step = "step_" + str(blurring_step)
path_parts = [base_path, experiment_name, "step_" + str(blurring_step)]
log_dir = os.path.join(*path_parts)
tb_logger = pl_loggers.TensorBoardLogger(log_dir)


trainer_args = {'accelerator': 'ddp',
                'gpus': 2,
                'default_root_dir': log_dir,
                'callbacks': [checkpoint_callback],
                'benchmark': False,
                'fast_dev_run': False,
                'gradient_clip_val': 1.0,
                'logger': tb_logger,
                'log_every_n_steps': 20,
                'auto_scale_batch_size': 'binsearch'}#,
                #'accumulate_grad_batches': 6}#,}
                # 'log_gpu_memory': 'all'} # might slow down performance (unnecessary uses only the output of nvidia-smi)

trainer = pl.Trainer(max_epochs=150, **trainer_args)

trainer.fit(reconstructor_pretrained, datamodule=dataset_real)