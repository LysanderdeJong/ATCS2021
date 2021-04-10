import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchtext.legacy.data import BucketIterator
from dataloader import create_dataset
from utils import get_model
from pl_model import SentenceEmbeddings


class LogCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, trainer, pl_module):
        for name, params in pl_module.named_parameters():
            trainer.logger.experiment.add_histogram(name, params, trainer.current_epoch)
            
class LearningRateCallback(pl.Callback):
    def __init__(self, init_lr=1e-1, decay_rate=0.99):
        super().__init__()
        self.currect_lr = init_lr
        self.current_val_acc = 0.0
        self.decay_weight = decay_rate
        
    def apply_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.currect_lr

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        trainer.logger.log_metrics({'SGD':self.currect_lr}, step=trainer.global_step)
        self.currect_lr = trainer.optimizers[0].param_groups[0]['lr']
        if trainer.current_epoch > 0 and self.currect_lr > 1e-5:
            self.currect_lr *= self.decay_weight
            self.apply_lr(trainer.optimizers[0])
            
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch > 0 and self.currect_lr > 1e-5:
            if self.current_val_acc > trainer.callback_metrics['val_acc']:
                self.currect_lr /= 5
                self.apply_lr(trainer.optimizers[0])
            else:
                self.current_val_acc = trainer.callback_metrics['val_acc']
                
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if trainer.current_epoch > 0 and self.currect_lr < 1e-5:
            trainer.should_stop = True
            
class PrintCallback(pl.Callback):
    def __init__(self):
        super().__init__()
            
    def on_train_epoch_end(self, trainer, pl_module, outputs):
        print(f"Epoch {trainer.current_epoch} finished.")

def train(args):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    pl.seed_everything(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # create dataloaders
    train, val, test, TEXT, LABEL = create_dataset()
    train_loader, val_loader, test_loader = BucketIterator.splits((train, val, test),
                                                                  sort_key=lambda x: len(x.premise),
                                                                  batch_size=args.batch_size)

    # Create a PyTorch Lightning trainer
    callbacks = []
    modelcheckpoint = ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1,
                                      save_last=True, filename='{epoch}-{val_loss:.4f}-{val_acc:.3f}')
    callbacks.append(modelcheckpoint)
    callbacks.append(LogCallback())
    callbacks.append(LearningRateCallback())
    if not args.progress_bar:
        callbacks.append(PrintCallback())
        
    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         auto_select_gpus=True,
                         gpus=None if args.gpus == "None" else int(args.gpus),
                         max_epochs=args.epochs,
                         callbacks=callbacks,
                         auto_scale_batch_size='binsearch' if args.auto_batch else None,
                         auto_lr_find=True if args.auto_lr else False,
                         precision=args.precision,
                         progress_bar_refresh_rate=1 if args.progress_bar else 0,
                         limit_train_batches=args.train_limit,
                         limit_val_batches=args.val_limit,
                         limit_test_batches=args.test_limit,
                         accumulate_grad_batches=args.grad_batch,
                         resume_from_checkpoint=args.checkpoint_path,
                         gradient_clip_val=args.grad_clip,
                         benchmark=True if args.benchmark else False,
                         plugins=args.plugins,
                         profiler=args.profiler if args.profiler else None)
    trainer.logger._default_hp_metric = None
    trainer.logger._log_graph = False
    
    # Create model
    model = get_model(TEXT, args)
        
    if not args.progress_bar:
        print("\nThe progress bar has been surpressed. For updates on the training progress, " + \
              "check the TensorBoard file at " + trainer.logger.log_dir + ". If you " + \
              "want to see the progress bar, use the argparse option \"progress_bar\".\n")

    # Training
    # with torch.autograd.set_detect_anomaly(True):
    trainer.tune(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    print(modelcheckpoint.best_model_path)
    trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    print("Check the Tensorboard to monitor training progress.")
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser = SentenceEmbeddings.add_model_specific_args(parser)
    
    # trainer hyperparameters
    parser.add_argument('--epochs', default=30, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for training.')
    
    parser.add_argument('--precision', default=32, type=int,
                        choices=[16, 32],
                        help='At what precision the model should train.')
    parser.add_argument('--grad_batch', default=1, type=int,
                        help='Accumulate gradient to simulate larger batch sizes.')
    parser.add_argument('--grad_clip', default=5., type=float,
                        help='Clip the gradient norm.')
    parser.add_argument('--plugins', default=None, type=str,
                        help='Modify the multi-gpu training path. See docs lightning docs for details.')
    
    parser.add_argument('--gpus', default=1, type=str,
                        help='Which gpus to use.')
    
    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='Continue training from this checkpoint.')
    
    # other hyperparameters
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--log_dir', default='logs/', type=str,
                        help='Directory where the PyTorch Lightning logs ' + \
                             'should be created.')
    
    parser.add_argument('--train_limit', default=1.0, type=float,
                        help='Percentage of data to train on.')
    parser.add_argument('--val_limit', default=1.0, type=float,
                        help='Percentage of data to validate with.')
    parser.add_argument('--test_limit', default=1.0, type=float,
                        help='Percentage of data to test with.')
    
    parser.add_argument('--progress_bar', action='store_true',
                        help='Use a progress bar indicator for interactive experimentation. '+ \
                             'Not to be used in conjuction with SLURM jobs.')
    parser.add_argument('--auto_lr', action='store_true',
                        help='When used tries to automatically set an appropriate learning rate.')
    parser.add_argument('--auto_batch', action='store_true',
                        help='When used tries to automatically set an appropriate batch size.')
    parser.add_argument('--benchmark', action='store_true',
                        help='Enables cudnn auto-tuner.')
    
    parser.add_argument('--profiler', default=None, type=str,
                        choices=['simple', 'advanced', 'pytorch'],
                        help='Code profiler.')
    
    args = parser.parse_args()
    train(args)