import copy
import io
from typing import Dict, List, Optional, Union

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import search_space.cell_options
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from pytorch_lightning import Trainer

from pymoo.core.problem import Problem
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping, BatchSizeFinder
from pytorch_lightning.loggers import TensorBoardLogger
from search_space.cgpnas.CGPDecoder_new import CGPDecoder
from search_space.cgpnas.CGPDecoder import CGPDecoder as CGPDecoder_original
from torch.nn import functional as F
from torch.nn import init
from torchmetrics import Accuracy
from torchprofile import profile_macs

CALLBACKS_MAP = {
    "EarlyStopping": EarlyStopping,
    "BatchSizeFinder": BatchSizeFinder
}


class IntHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
        handlebox.add_artist(text)
        return text


def weights_init_kaiming(m: nn.Module):
    """Initialize weights using Kaiming Normal initialization."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):  # Apply only to relevant layers
        if m.weight is not None and m.weight.numel() > 0:  # Check if weight exists and is non-empty
            init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None and m.bias.numel() > 0:  # Check if bias exists and is non-empty
            init.constant_(m.bias, 0.0)

    elif isinstance(m, nn.BatchNorm2d):  # Handle BatchNorm separately
        if m.weight is not None and m.weight.numel() > 0:
            init.uniform_(m.weight, 0.02, 1.0)
        if m.bias is not None and m.bias.numel() > 0:
            init.constant_(m.bias, 0.0)


def init_weights(net: nn.Module, init_type: str = "kaiming"):
    """Apply weight initialization to the network."""
    if init_type == "kaiming":
        net.apply(weights_init_kaiming)
    else:
        raise ValueError(f"Unsupported initialization method: {init_type}")


class MO_evaluation(Problem):
    def __init__(
            self,
            max_epochs: int = 1,
            nvar_real: int = 0,
            objectives: Dict = None,
            datamodule: Optional[LightningDataModule] = None,
            train_dataloaders: Optional[LightningDataModule] = None,
            val_dataloaders: Optional[LightningDataModule] = None,
            layer_options=None,
            num_classes: Optional[int] = None,
            devices: Union[List[int], str, int] = "auto",
            accelerator: str = 'cpu',
            callbacks: Dict = None,
            logger_params=None,
            generations=None,
            decoder_style='original',
            **evaluation_kwargs

    ):
        super().__init__(n_var=1, n_obj=len(objectives), n_constr=0, requires_kwargs=True)

        self.epochs = max_epochs
        self.datamodule = datamodule
        self.trainloaders = train_dataloaders
        self.valloaders = val_dataloaders
        self.devices = devices
        self.accelerator = accelerator
        self.logger_params = logger_params
        self.max_gen = generations
        self.decoder_style = decoder_style
        self.problem_type = 'nas'

        # Bounds for real-valued variables
        self.xl = np.zeros(nvar_real)
        self.xu = 0.999999 * np.ones(nvar_real)

        self.class_args = objectives['Classification_Error'] if 'Classification_Error' in objectives else None
        self.macs_args = objectives['MAC'] if 'MAC' in objectives else None
        self.params_args = objectives['Parameters'] if 'Parameters' in objectives else None
        self.objectives = objectives

        self.logger = self.setup_logger()
        self.global_step = 0
        self.layer_types = search_space.cell_options.__all__ if search_space is None else layer_options
        self.layer_types.append('full')

        self.generation = 0
        self.n_evals = []

        self.f = {
            'Accuracy': [],
            'Error': [],
            'MAC': [],
            'Params': []
        }

        self.f_gen = [{
            'Accuracy': [],
            'Error': [],
            'MAC': [],
            'Params': []
        } for _ in range(self.max_gen)]

        self.callbacks = []
        if callbacks is not None:
            for c, kwargs in callbacks.items():
                if c in CALLBACKS_MAP:  # Ensure it's a valid callback
                    self.callbacks.append(CALLBACKS_MAP[c](**kwargs))
                else:
                    raise ValueError(f"Unknown callback: {c}")

    def reset(self):
        self.global_step = 0
        self.generation = 0
        self.n_evals = []

        self.f = {
            'Accuracy': [],
            'Error': [],
            'MAC': [],
            'Params': []
        }

        self.f_gen = [{
            'Accuracy': [],
            'Error': [],
            'MAC': [],
            'Params': []
        } for _ in range(self.max_gen)]

    def debug(self, string):
        print(string)
        copy.deepcopy(self)

    def count_parameters(self, model):
        """
        Count the number of trainable parameters in a model.

        Args:
            model (torch.nn.Module): PyTorch model.

        Returns:
            int: Number of trainable parameters.
        """
        scale = self.params_args['scale'] if self.params_args is not None else 1e6
        params = sum(p.numel() for p in model.parameters() if p.requires_grad) / scale

        # Normalize
        norm_params = np.log10(params) / 3
        return params

    def count_macs(self, model):
        scale = self.macs_args['scale'] if self.macs_args is not None else 1e6
        in_tensor_size = self.datamodule.input_tensor
        device = torch.device('cpu')
        input_tensor = torch.zeros(1, in_tensor_size[2], in_tensor_size[0], in_tensor_size[1], dtype=torch.float,
                                   device=device, requires_grad=False)
        macs = profile_macs(model.eval(), input_tensor) / scale  # Convert to millions

        # Normalize
        norm_macs = np.log10(macs) / 3
        return macs

    def setup_logger(self, generation=None, individual=None):
        # The folder for the tensorboard event files
        save_dir = self.logger_params.get('save_dir', 'lightning_logs')

        # This should be the name of the type of experiment
        name = self.logger_params.get('name', 'CGP')

        # This should be the name of the version of the experiment
        version = self.logger_params.get('version', None)

        logger = TensorBoardLogger(save_dir,
                                   name,
                                   version)
        return logger

    def log_metrics(self, m, validation_accuracy, macs, params):
        print(f"{m}: Metrics - [Error: {100 - validation_accuracy}, MACs: {macs}, Parameters: {params}]")

        self.f['Accuracy'].append(validation_accuracy)
        self.f['Error'].append(100 - validation_accuracy)
        self.f['MAC'].append(macs)
        self.f['Params'].append(params)

        self.f_gen[self.generation]['Accuracy'].append(validation_accuracy)
        self.f_gen[self.generation]['Error'].append(100 - validation_accuracy)
        self.f_gen[self.generation]['MAC'].append(macs)
        self.f_gen[self.generation]['Params'].append(params)

    def log_gen_metrics(self):
        F = np.column_stack([self.f['Error'], self.f['MAC']])
        F_new = np.column_stack([self.f_gen[self.generation]['Error'], self.f_gen[self.generation]['MAC']])
        pF = F[fast_non_dominated_sort(F)[0]]

        hv_metric = Hypervolume(ref_point=np.array([1.1, 1.1]),
                                norm_ref_point=False,
                                zero_to_one=True,
                                ideal=F.min(axis=0),
                                nadir=F.max(axis=0))

        self.n_evals.append(self.global_step)
        hv = [hv_metric.do(_F) for _F in F_new]

        self.logger.experiment.add_scalar('Generational Results MOO/Hypervolume', np.mean(hv), self.generation)

        plt.figure()
        plt.scatter(self.f['Error'], self.f['MAC'], alpha=0.3, label='Historic')
        plt.scatter(self.f_gen[self.generation]['Error'], self.f_gen[self.generation]['MAC'], alpha=0.7,
                    label=f'Gen {self.generation}')
        plt.scatter(pF[:, 0], pF[:, 1], label='Pareto Front')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.xlabel('% Classification Error')
        plt.ylabel('MAdds')
        plt.title("Objective Space")

        buf = io.BytesIO()

        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        plt.close()
        self.logger.experiment.add_image(f"Objective Space", im, self.generation)

        val_avg = np.mean(self.f_gen[self.generation]['Accuracy'])
        error_avg = np.mean(self.f_gen[self.generation]['Error'])
        mac_avg = np.mean(self.f_gen[self.generation]['MAC'])
        param_avg = np.mean(self.f_gen[self.generation]['Params'])

        val_max = np.max(self.f_gen[self.generation]['Accuracy'])
        error_min = np.min(self.f_gen[self.generation]['Error'])
        mac_min = np.min(self.f_gen[self.generation]['MAC'])
        param_min = np.min(self.f_gen[self.generation]['Params'])

        self.logger.experiment.add_scalar('Generational Results Mean/Val Accuracy', val_avg, self.generation)
        self.logger.experiment.add_scalar('Generational Results Mean/Val Error', error_avg, self.generation)
        self.logger.experiment.add_scalar('Generational Results Mean/#MACs', mac_avg, self.generation)
        self.logger.experiment.add_scalar('Generational Results Mean/#Parameters', param_avg, self.generation)

        self.logger.experiment.add_scalar('Generational Results Best/Val Accuracy', val_max, self.generation)
        self.logger.experiment.add_scalar('Generational Results Best/Val Error', error_min, self.generation)
        self.logger.experiment.add_scalar('Generational Results Best/#MACs', mac_min, self.generation)
        self.logger.experiment.add_scalar('Generational Results Best/#Parameters', param_min, self.generation)

    def _calc_pareto_front(self, *args, **kwargs):
        pass

    def plot_genotype(self, genotypes):
        # Extract unique names and assign colors
        color_map = {name: i for i, name in enumerate(self.layer_types)}

        # Convert mods to a matrix (row = sequence, col = elements, value = color index)
        max_length = max(len(seq) - 1 for seq in genotypes)  # Find longest sequence

        # Genotypes is a list of 100 elements long
        max_height = min(len(genotypes), 24)  # Max height plot
        matrix = np.full((max_height, max_length), np.nan)

        # print('\ngenotypes', genotypes)
        for i, seq in enumerate(genotypes[:max_height]):
            for j, item in enumerate(seq[1:]):
                matrix[i, j] = color_map[item[0]]

            df = pd.DataFrame(
                matrix,
                index=np.arange(max_height),
                columns=np.arange(max_length)
            )

            fig, ax = plt.subplots(figsize=(max_length // 2, max(max_height, 4) // 2))
            fig.subplots_adjust(left=0.05, right=.65)
            sns.set_theme(font_scale=1.2)
            sns.heatmap(df,
                        annot=True,
                        annot_kws={"size": 16},
                        ax=ax)
            ax.legend(
                color_map.values(),
                color_map.keys(),
                handler_map={int: IntHandler()},
                loc='upper left',
                bbox_to_anchor=(1.2, 1)
            )
            ax.set_xlabel('Depth Model')
            ax.set_ylabel('Genotype')

            buf = io.BytesIO()

            plt.savefig(buf, format='jpeg', bbox_inches='tight')
            buf.seek(0)
            im = Image.open(buf)
            im = torchvision.transforms.ToTensor()(im)
            plt.close()
            self.logger.experiment.add_image(f"Genotype Representation Generation {self.generation}", im,
                                             self.generation)

    def _evaluate(self, x, out, *args, **kwargs):
        all_solutions = []

        # self.plot_genotype([model[0].active_net_list() for model in x])
        for m, model_instance in enumerate(x):
            self.aux_head = kwargs.get('aux_head', False)
            epochs = kwargs.get('epochs', self.epochs)
            task = Classification_Module(model_instance[0].active_net_list(),
                                         input_tensor=self.datamodule.input_tensor,
                                         n_classes=len(self.datamodule.classes),
                                         aux_head=self.aux_head,
                                         max_epochs=epochs,
                                         decoder_style=self.decoder_style,
                                         **self.class_args)
            trainer = Trainer(accelerator=self.accelerator,
                              devices=self.devices,
                              max_epochs=epochs,
                              logger=self.logger,
                              enable_progress_bar=False,
                              enable_checkpointing=False,
                              )
            trainer.fit(task, datamodule=self.datamodule)
            validation_accuracy = float(task.val_acc) * 100

            macs = self.count_macs(task.model)

            params = self.count_parameters(task.model)

            metric_list = [100 - validation_accuracy]

            if self.macs_args is not None:
                metric_list.append(macs)
            if self.params_args is not None:
                metric_list.append(params)
            self.log_metrics(m, validation_accuracy, macs, params)
            self.global_step += 1
            all_solutions.append(metric_list)

        # Store results
        self.log_gen_metrics()
        out["F"] = np.array(all_solutions, dtype=float)
        print("Final Evaluation Results:", out["F"])
        self.generation += 1


class Classification_Module(pl.LightningModule):
    def __init__(self,
                 model_description,
                 input_tensor,
                 n_classes,
                 aux_head=False,
                 learning_rate=None,
                 weight_decay=None,
                 max_epochs=None,
                 decoder_style='original',
                 **kwargs):
        super().__init__(**kwargs)
        self.model_description = model_description
        self.input_tensor = input_tensor
        self.n_classes = n_classes
        self.aux_head = aux_head
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = F.cross_entropy
        self.max_epochs = max_epochs
        self.decoder_style = decoder_style

        self.accuracy = Accuracy(task='multiclass', num_classes=n_classes)
        self.model = None
        self.validation_step_outputs = []
        self.val_acc = 0
        # try:
        if self.decoder_style == 'original':
            self.model = CGPDecoder_original(model_description, self.input_tensor, self.n_classes, self.aux_head)
        else:
            self.model = CGPDecoder(model_description, self.input_tensor, self.n_classes, self.aux_head)

        init_weights(self.model, "kaiming")

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        """Customized optimizer with momentum, as well as a scheduler."""
        optimizer = torch.optim.SGD(
            self.parameters(),
            momentum=0.9,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=1e-3)
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)[0]
        loss = self.criterion(logits, y)
        # print(loss)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)[0]
        loss = self.criterion(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.validation_step_outputs.append(acc.item())
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)[0]
        loss = self.criterion(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.test_acc.append(acc)
        return loss

    def on_train_epoch_start(self):
        # Logging learning rate at the beginning of every epoch
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])

    def on_validation_epoch_end(self) -> None:
        self.val_acc = np.mean(self.validation_step_outputs)
        print(f"Validation Accuracy: {self.val_acc:.4f}")
        self.validation_step_outputs.clear()

