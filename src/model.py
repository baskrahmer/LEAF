from typing import Tuple, List, Optional, Set

import lightning
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import Accuracy, F1Score, MeanAbsoluteError, MeanSquaredError
from torchmetrics.text import Perplexity
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from src.config import Config


class RegressionHead(nn.Module):
    """
    Model head to predict a continuous, non-negative target variable.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(in_features=hidden_dim, out_features=1)
        self.activation = nn.Softplus()
        self.loss = nn.MSELoss()

    def __call__(self, activations, regressands, **kwargs) -> dict:
        predicted_values = self.activation(self.linear(activations))
        loss = self.loss(predicted_values, regressands)
        return {
            "predicted_values": predicted_values,
            "loss": loss,
        }


class ClassificationHead(nn.Module):
    """
    Model head to predict a categorical target variable.
    """

    def __init__(self, hidden_dim: int, num_classes: int, idx_to_co2e: dict):
        super().__init__()
        self.linear = nn.Linear(in_features=hidden_dim, out_features=num_classes)
        self.loss = nn.CrossEntropyLoss()

        # Turn dict into lookup table
        self.idx_to_co2e = idx_to_co2e

    def __call__(self, activations, classes, **kwargs) -> dict:
        logits = self.linear(activations)
        loss = self.loss(logits, classes)
        _, predicted_classes = torch.max(F.softmax(logits, dim=1), dim=1)
        return {
            "predicted_values": torch.tensor([self.idx_to_co2e[p] for p in predicted_classes.numpy()]).unsqueeze(-1),
            "logits": logits,
            "loss": loss,
        }


class HybridHead(nn.Module):
    """
    Hybrid model head with predictions based on both categorical and continuous target variables.
    """

    def __init__(self, hidden_dim: int, num_classes: int, alpha: float):
        super().__init__()
        self.classification_linear = nn.Linear(in_features=hidden_dim, out_features=num_classes)
        self.regression_linear = nn.Linear(in_features=num_classes, out_features=1)
        self.regression_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.alpha = alpha

    def __call__(self, activations, classes, regressands, **kwargs) -> dict:
        logits = self.classification_linear(activations)
        classification_loss = self.classification_loss(logits, classes)
        predicted_values = self.regression_linear(logits)
        regression_loss = self.regression_loss(predicted_values, regressands)
        return {
            "predicted_values": predicted_values,
            "logits": logits,
            "loss": self.alpha * classification_loss + (1 - self.alpha) * regression_loss,
        }


class LEAFModel(nn.Module):

    def __init__(self, c, num_classes: int, base_model: Optional[PreTrainedModel] = None,
                 idx_to_co2e: Optional[dict] = None):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(c.model_name)

        # Transfer pretrained model weights to the new model
        if base_model is not None:
            transfer_weights = {k.replace("bert.", ""): v for k, v in base_model.state_dict().items()}
            result = self.base_model.load_state_dict(transfer_weights, strict=False)
            if len(result.unexpected_keys) == len(base_model.state_dict()):
                raise ValueError("No weights were transferred from the base model to the new model.")

        hidden_dim = self.base_model.config.hidden_size
        if c.objective == "classification":
            self.head = ClassificationHead(hidden_dim=hidden_dim, num_classes=num_classes, idx_to_co2e=idx_to_co2e)
        elif c.objective == "regression":
            self.head = RegressionHead(hidden_dim=hidden_dim)
        elif c.objective == "hybrid":
            self.head = HybridHead(hidden_dim=hidden_dim, num_classes=num_classes, alpha=c.alpha)
        else:
            raise ValueError

    def forward(self, input_ids, attention_mask, **kwargs) -> dict:
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(outputs.last_hidden_state[:, 0], **kwargs)


class LightningWrapper(lightning.LightningModule):
    """
    Wrapper class for the model to be used with PyTorch Lightning.
    """

    def __init__(
            self, c: Config, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, num_classes: int, mlm: bool,
            languages: Set[str], classes: Set[str]
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.classes = classes
        self.languages = languages

        self.trainable_parameters = self.model.parameters() if mlm else self.model.head.parameters()

        self._device = "cuda" if (c.use_gpu and torch.cuda.is_available()) else "cpu"

        self.learning_rate = c.learning_rate
        self.num_steps = c.train_steps if not mlm else c.mlm_train_steps
        self.train_batch_size = c.train_batch_size
        self.test_batch_size = c.test_batch_size
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        def make_lang_filter(key, value):
            return lambda x: np.array(x[key]) == value

        def make_class_filter(key, value):
            return lambda x: x[key].cpu().numpy() == value

        language_filters = {l: make_lang_filter("lang", l) for l in languages}
        class_filters = {c: make_class_filter("classes", c) for c in classes}
        self.metric_filters = language_filters | class_filters | {"all": lambda x: [True] * len(x["lang"])}

        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}

        for filter in self.metric_filters.keys():
            self.train_metrics |= self.get_metric_dict(c, filter, num_classes=num_classes,
                                                       objective="mlm" if mlm else c.objective)
            self.val_metrics |= self.get_metric_dict(c, filter, num_classes=num_classes,
                                                     objective="mlm" if mlm else c.objective)
            self.test_metrics |= self.get_metric_dict(c, filter, num_classes=num_classes,
                                                      objective="mlm" if mlm else c.objective)

    def get_metric_dict(self, c: Config, split: str, num_classes: int, objective: str) -> dict[str, torch.nn.Module]:
        metric_dict = {}
        if objective == "mlm":
            metric_dict[f"{split}_perplexity"] = Perplexity(ignore_index=self.tokenizer.pad_token_id).to(self._device)
        else:
            metric_dict[f"{split}_mae"] = MeanAbsoluteError().to(self._device)
            metric_dict[f"{split}_mse"] = MeanSquaredError().to(self._device)
            if objective == "classification" or objective == "hybrid":
                metric_dict[f"{split}_accuracy"] = Accuracy("multiclass", num_classes=num_classes).to(self._device)
                metric_dict[f"{split}_f1"] = F1Score("multiclass", num_classes=num_classes).to(self._device)
        return metric_dict

    def update_metrics(self, batch, outputs: dict, metrics: dict[str, torch.nn.Module], data_split: str) -> None:
        for metric_key, metric in metrics.items():
            filter_fn = self.metric_filters["_".join(metric_key.split("_")[:-1])]
            filter_idx = filter_fn(batch)
            if not any(filter_idx):
                continue

            if metric_key.endswith("f1") or metric_key.endswith("accuracy"):
                logits = outputs["logits"][filter_idx]
                classes = batch['classes'][filter_idx]
                value = metric(preds=logits, target=classes)
            elif metric_key.endswith("perplexity"):
                ids = batch["input_ids"][filter_idx]
                logits = outputs["logits"][filter_idx]
                value = metric(preds=logits, target=ids)
            else:
                pred_values = outputs["predicted_values"][filter_idx]
                regressands = batch['regressands'][filter_idx]
                value = metric(preds=pred_values, target=regressands)
            self.log(f"{data_split}_{metric_key}", value=value, on_step=data_split == "train", on_epoch=True,
                     prog_bar=True,
                     batch_size=self.train_batch_size if data_split == "train" else self.test_batch_size)

    @staticmethod
    def clear_metrics(metrics: dict[str, torch.nn.Module]):
        for metric_key, metric in metrics.items():
            metric.reset()

    def macro_average_metrics(self, metrics: dict[str, torch.nn.Module], data_split: str) -> dict[str, float]:
        macro_averages = {}
        for metric_key, metric in metrics.items():
            identifier = metric_key.split("_")[0]
            base_metric = "_".join(metric_key.split("_")[1:])
            if identifier in self.languages:
                macro_key = f"lang_{base_metric}"
            elif identifier in self.classes:
                macro_key = f"class_{base_metric}"
            else:
                continue
            macro_averages[macro_key] = macro_averages.get(macro_key, []) + [metric.compute()]
        for key, values in macro_averages.items():
            self.log(f"{data_split}_{key}", value=torch.tensor(values).mean(),
                     on_step=False, on_epoch=True, prog_bar=True,
                     batch_size=self.train_batch_size if data_split == "train" else self.test_batch_size)

    def training_step(self, batch: dict):
        outputs = self.forward(batch)
        self.log("train_loss", outputs["loss"], on_step=True, on_epoch=True, batch_size=self.train_batch_size)
        self.update_metrics(batch, outputs, self.train_metrics, "train")
        return outputs["loss"]

    def validation_step(self, batch: dict):
        outputs = self.forward(batch)
        self.log("val_loss", outputs["loss"], on_step=False, on_epoch=True, batch_size=self.test_batch_size)
        self.update_metrics(batch, outputs, self.val_metrics, "val")
        return outputs["loss"]

    def test_step(self, batch: dict):
        outputs = self.forward(batch)
        self.log("test_loss", outputs["loss"], on_step=False, on_epoch=True, batch_size=self.test_batch_size)
        self.update_metrics(batch, outputs, self.test_metrics, "test")
        return outputs["loss"]

    def on_train_epoch_end(self) -> None:
        self.macro_average_metrics(self.train_metrics, "train")
        self.clear_metrics(self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        self.macro_average_metrics(self.val_metrics, "val")
        self.clear_metrics(self.val_metrics)

    def on_test_epoch_end(self) -> None:
        self.macro_average_metrics(self.test_metrics, "test")
        self.clear_metrics(self.test_metrics)

    def forward(self, batch: dict) -> dict:
        return self.model(**{k: v for k, v in batch.items() if k != "lang"})

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[dict]]:
        optimizer = torch.optim.AdamW(
            params=[p for p in self.trainable_parameters],
            lr=self.learning_rate,
            betas=(0.9, 0.999),
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_training_steps=self.num_steps,
            num_warmup_steps=int(0.1 * self.num_steps)
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


def get_tokenizer(c: Config) -> Tuple[PreTrainedTokenizerBase, dict]:
    tokenizer = AutoTokenizer.from_pretrained(c.model_name)
    tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": c.max_length}
    return tokenizer, tokenizer_kwargs
