import torch
import pytorch_lightning as pl

isonnx_export = False #训练时不打开，导出onnx模型时打开，且调小batchsize
class ModelModule(pl.LightningModule):
    def __init__(self, backbone, loss_func, metrics, optimizer_args, scheduler_args=None, cfg=None):
        super().__init__()

        self.save_hyperparameters(
            cfg,
            ignore=['backbone', 'loss_func', 'metrics', 'optimizer_args', 'scheduler_args'])

        self.backbone = backbone #此处backbone即为cvt，而不是efficientnet
        self.loss_func = loss_func
        self.metrics = metrics
        self.batch = None
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args
        self.idx = 0

    def forward(self, batch):
        return self.backbone(batch)

    def shared_step(self, batch, prefix='', on_step=False, return_output=True):
        pred = self(batch)
        loss, loss_details = self.loss_func(pred, batch)

        self.metrics.update(pred, batch)

        if self.trainer is not None:
            self.log(f'{prefix}/loss', loss.detach(), on_step=on_step, on_epoch=True)
            self.log_dict({f'{prefix}/loss/{k}': v.detach() for k, v in loss_details.items()},
                          on_step=on_step, on_epoch=True)

        # Used for visualizations
        if return_output:
            return {'loss': loss, 'batch': batch, 'pred': pred}

        return {'loss': loss}
    
    def onnx_export(self, batch):
        self.backbone.eval()
        torch.onnx.export(self.backbone,                     # model being run
                  ##since model is in the cuda mode, input also need to be
                  batch,              # model input (or a tuple for multiple inputs)
                  "model_troch_export.onnx", # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
        print('--------export finish--------')


    def training_step(self, batch, batch_idx):
        self.batch = batch
        if(isonnx_export):
            self.onnx_export(self.batch)
        return self.shared_step(batch, 'train', True,
                                batch_idx % self.hparams.experiment.log_image_interval == 0)

    # def training_epoch_end(self, training_step_outputs):
    #     self.onnx_export(self.batch)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val', False,
                                batch_idx % self.hparams.experiment.log_image_interval == 0)

    def on_validation_start(self) -> None:
        self._log_epoch_metrics('train')
        self._enable_dataloader_shuffle(self.trainer.val_dataloaders)

    def validation_epoch_end(self, outputs):
        self._log_epoch_metrics('val')

    def _log_epoch_metrics(self, prefix: str):
        """
        lightning is a little odd - it goes

        on_train_start
        ... does all the training steps ...
        on_validation_start
        ... does all the validation steps ...
        on_validation_epoch_end
        on_train_epoch_end
        """
        metrics = self.metrics.compute()

        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, val in value.items():
                    self.log(f'{prefix}/metrics/{key}{subkey}', val)
            else:
                self.log(f'{prefix}/metrics/{key}', value)

        self.metrics.reset()

    def _enable_dataloader_shuffle(self, dataloaders):
        """
        HACK for https://github.com/PyTorchLightning/pytorch-lightning/issues/11054
        """
        for v in dataloaders:
            v.sampler.shuffle = True
            v.sampler.set_epoch(self.current_epoch)

    def configure_optimizers(self, disable_scheduler=False):
        parameters = [x for x in self.backbone.parameters() if x.requires_grad]
        optimizer = torch.optim.AdamW(parameters, **self.optimizer_args)

        if disable_scheduler or self.scheduler_args is None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda lr: 1)
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **self.scheduler_args)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
