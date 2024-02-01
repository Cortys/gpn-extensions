import json
import os

from gpn.utils.storage import create_storage

os.environ["MKL_THREADING_LAYER"] = "GNU"

from typing import Optional, Dict, Any
from sacred import Experiment
import pyblaze.nn as xnn
from pyblaze.nn.engine._history import History
from pyblaze.utils.torch import gpu_device

import gpn.nn as unn
from gpn.models import create_model
from gpn.models import EnergyScoring, DropoutEnsemble, Ensemble
from gpn.utils import set_seed, ModelNotFoundError
from gpn.nn import TransductiveGraphEngine
from gpn.nn import get_callbacks_from_config
from gpn.utils import RunConfiguration, DataConfiguration
from gpn.utils import ModelConfiguration, TrainingConfiguration
from .dataset import ExperimentDataset


class TransductiveExperiment:
    """base experiment which works for default models and default GraphEngine"""

    def __init__(
        self,
        run_cfg: RunConfiguration,
        data_cfg: DataConfiguration,
        model_cfg: ModelConfiguration,
        train_cfg: TrainingConfiguration,
        ex: Optional[Experiment] = None,
        dataset: Optional[ExperimentDataset] = None,
    ):
        self.run_cfg = run_cfg
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.evaluation_results = None
        self.training_completed = False
        self.storage = None
        self.storage_params = None

        self.model = None
        self.ex = ex

        # metrics for evaluation of default graph
        # and id+ood splits combined for ood
        self.metrics = [
            "accuracy",
            "brier_score",
            "ece",
            "confidence_aleatoric_apr",
            "confidence_epistemic_apr",
            "confidence_structure_apr",
            "confidence_aleatoric_auroc",
            "confidence_epistemic_auroc",
            "confidence_structure_auroc",
            "ce",
            "avg_prediction_confidence_aleatoric",
            "avg_prediction_confidence_epistemic",
            "avg_sample_confidence_aleatoric",
            "avg_sample_confidence_aleatoric_entropy",
            "avg_sample_confidence_epistemic",
            "avg_sample_confidence_epistemic_entropy",
            "avg_sample_confidence_features",
            "avg_sample_confidence_neighborhood",
            "average_entropy",
            "accuracy_rejection_prediction_confidence_aleatoric",
            "accuracy_rejection_prediction_confidence_epistemic",
            "accuracy_rejection_sample_confidence_aleatoric",
            "accuracy_rejection_sample_confidence_aleatoric_entropy",
            "accuracy_rejection_sample_confidence_epistemic",
            "accuracy_rejection_sample_confidence_epistemic_entropy",
        ]

        if self.run_cfg.reduced_training_metrics:
            self.train_metrics = ["accuracy", "ce"]
        else:
            # Leave out accuracy rejections curves for training
            self.train_metrics = self.metrics[:-6]

        self.ood_metrics = [
            # metrics for ood detection (id vs ood)
            "ood_detection_aleatoric_apr",
            "ood_detection_aleatoric_auroc",
            "ood_detection_epistemic_apr",
            "ood_detection_epistemic_auroc",
            "ood_detection_epistemic_entropy_apr",
            "ood_detection_epistemic_entropy_auroc",
            "ood_detection_features_apr",
            "ood_detection_features_auroc",
            "ood_detection_neighborhood_apr",
            "ood_detection_neighborhood_auroc",
            "ood_detection_structure_apr",
            "ood_detection_structure_auroc",
            # id metrics
            "ood_accuracy",
            "ood_avg_prediction_confidence_aleatoric",
            "ood_avg_prediction_confidence_epistemic",
            "ood_avg_sample_confidence_aleatoric",
            "ood_avg_sample_confidence_aleatoric_entropy",
            "ood_avg_sample_confidence_epistemic",
            "ood_avg_sample_confidence_epistemic_entropy",
            "ood_avg_sample_confidence_neighborhood",
            "ood_avg_sample_confidence_features",
            "ood_average_entropy",
            # ood metrics
            "id_accuracy",
            "id_avg_prediction_confidence_aleatoric",
            "id_avg_prediction_confidence_epistemic",
            "id_avg_sample_confidence_aleatoric",
            "id_avg_sample_confidence_aleatoric_entropy",
            "id_avg_sample_confidence_epistemic",
            "id_avg_sample_confidence_epistemic_entropy",
            "id_avg_sample_confidence_features",
            "id_average_entropy",
        ]

        # base dataset
        set_seed(self.model_cfg.seed)

        self.setup_storage()

        if dataset is None:
            if self.evaluation_results is None:
                self.dataset = ExperimentDataset(
                    self.data_cfg, to_sparse=self.data_cfg.to_sparse
                )
            else:
                self.dataset = None
        else:
            self.dataset = dataset

        if self.dataset is not None:
            self.model_cfg.set_values(
                dim_features=self.dataset.dim_features,
                num_classes=self.dataset.num_classes,
            )
            self.setup_model()
            self.setup_engine()

    def setup_storage(self):
        storage, storage_params = create_storage(
            self.run_cfg, self.data_cfg, self.model_cfg, self.train_cfg, ex=self.ex
        )
        self.storage = storage
        self.storage_params = storage_params

        results_file_path = storage.create_results_file_path(
            self.model_cfg.model_name,
            storage_params,
            init_no=self.model_cfg.init_no,
        )
        self.results_file_path = results_file_path

        if not self.run_cfg.reeval and os.path.exists(results_file_path):
            with open(results_file_path, "r") as f:
                self.evaluation_results: dict[str, Any] | None = json.load(f)
                if self.evaluation_results is not None:
                    # If cached eval results don't contain all metrics, re-evaluate
                    eval_res_keys = list(self.evaluation_results.keys())
                    if self.data_cfg.ood_flag:
                        metrics = self.metrics + self.ood_metrics
                    else:
                        metrics = self.metrics
                    for metric in metrics:
                        found = False
                        for res_key in eval_res_keys:
                            if metric in res_key:
                                found = True
                                break
                        if not found:
                            self.evaluation_results = None
                            break

    def setup_engine(self) -> None:
        assert self.dataset is not None
        self.engine = TransductiveGraphEngine(self.model, splits=self.dataset.splits)

    def setup_model(self) -> None:
        assert self.storage is not None
        assert self.storage_params is not None

        if self.run_cfg.eval_mode == "ensemble":
            self.run_cfg.set_values(save_model=False)

            # only allow creation of an ensemble when evaluating
            if self.run_cfg.job == "train":
                raise AssertionError

            if self.run_cfg.job == "evaluate":
                model = Ensemble(self.model_cfg, models=None)
                model.set_storage(self.storage, self.storage_params)
                model.load_from_storage()

            else:
                raise AssertionError

        else:
            model = create_model(self.model_cfg)
            model.set_storage(self.storage, self.storage_params)

            if not self.run_cfg.retrain:
                try:
                    # if it is possible to load model: skip training
                    model.load_from_storage()
                    self.run_cfg.set_values(job="evaluate")
                    model.set_expects_training(False)
                    self.run_cfg.set_values(save_model=False)

                    if self.run_cfg.eval_mode == "dropout":
                        assert self.run_cfg.job == "evaluate"
                        model = DropoutEnsemble(
                            model, num_samples=self.model_cfg.num_samples_dropout
                        )

                    elif self.run_cfg.eval_mode == "energy_scoring":
                        assert self.run_cfg.job == "evaluate"
                        model = EnergyScoring(
                            model, temperature=self.model_cfg.temperature
                        )

                except ModelNotFoundError:
                    pass

        self.model = model

    def evaluate(self) -> Dict[str, Any]:
        if self.evaluation_results is not None:
            return self.evaluation_results
        assert self.model is not None
        assert self.dataset is not None

        metrics = unn.get_metrics(self.metrics)
        eval_res = self.engine.evaluate(
            data=self.dataset.val_loader, metrics=metrics, gpu=self.run_cfg.gpu
        )
        eval_val = eval_res["val"]
        eval_test = eval_res["test"]
        results = {f"test_{k}": v for k, v in eval_test.items()}
        results = {**results, **{f"val_{k}": v for k, v in eval_val.items()}}

        if "all" in eval_res:
            eval_all = eval_res["all"]
            results = {**results, **{f"all_{k}": v for k, v in eval_all.items()}}

        self.model.write_results(results)

        return results

    def evaluate_ood(self) -> Dict[str, Any]:
        if self.evaluation_results is not None:
            return self.evaluation_results
        assert self.model is not None
        assert self.dataset is not None

        metrics = unn.get_metrics(self.metrics)
        ood_metrics = unn.get_metrics(self.ood_metrics)

        # for isolated evaluation and poisoning experiments
        # target values are uses as ID values
        # for other cases, target usually represents both ID and OOD combined
        target_as_id = (self.data_cfg.ood_setting == "poisoning") or (
            self.data_cfg.ood_dataset_type == "isolated"
        )

        eval_res = self.engine.evaluate_target_and_ood(
            data=self.dataset.val_loader,
            data_ood=self.dataset.ood_loader,
            target_as_id=target_as_id,
            metrics=metrics,
            metrics_ood=ood_metrics,
            gpu=self.run_cfg.gpu,
        )

        eval_val = eval_res["val"]
        eval_test = eval_res["test"]
        results = {f"test_{k}": v for k, v in eval_test.items()}
        results = {**results, **{f"val_{k}": v for k, v in eval_val.items()}}

        if "all" in eval_res:
            eval_all = eval_res["all"]
            results = {**results, **{f"all_{k}": v for k, v in eval_all.items()}}

        self.model.write_results(results)

        return results

    def train(self) -> History | None:
        if (
            self.model is None
            or self.dataset is None
            or not self.model.expects_training()
        ):
            return None

        callbacks = []
        warmup_callbacks = []

        if self.run_cfg.log:
            batch_progress_logger = xnn.callbacks.BatchProgressLogger()
            callbacks.append(batch_progress_logger)
            warmup_callbacks.append(batch_progress_logger)

        metrics = unn.get_metrics(self.train_metrics)

        callbacks.extend(get_callbacks_from_config(self.train_cfg))

        # move training datasets to gpu before training
        gpu = self.engine._gpu_descriptor(self.run_cfg.gpu)
        device = gpu_device(gpu[0] if isinstance(gpu, list) else gpu)

        self.dataset.train_dataset.to(device)
        self.dataset.train_val_dataset.to(device)
        self.dataset.warmup_dataset.to(device)
        self.dataset.finetune_dataset.to(device)

        eval_every = self.train_cfg.eval_every

        if eval_every is None:
            eval_every = 1

        # ------------------------------------------------------------------------------------------------
        # warmup training
        warmup_epochs = (
            0 if self.train_cfg.warmup_epochs is None else self.train_cfg.warmup_epochs
        )
        if warmup_epochs > 0:
            # set-up optimizer
            optimizer = self.model.get_warmup_optimizer(
                self.train_cfg.lr, self.train_cfg.weight_decay
            )

            self.engine.model.set_warming_up(True)
            _ = self.engine.train(
                train_data=self.dataset.warmup_loader,
                val_data=self.dataset.train_val_loader,
                optimizer=optimizer,
                likelihood_optimizer=None,
                loss=self.model.warmup_loss,
                epochs=self.train_cfg.warmup_epochs,
                eval_every=eval_every,
                eval_train=True,
                callbacks=warmup_callbacks,
                metrics=metrics,
                gpu=self.run_cfg.gpu,
            )

            self.engine.model.set_warming_up(False)

        # ------------------------------------------------------------------------------------------------
        # main training loop training
        # set-up optimizer
        optimizer = self.model.get_optimizer(
            self.train_cfg.lr, self.train_cfg.weight_decay
        )
        likelihood_optimizer = None

        if isinstance(optimizer, (tuple, list)):
            likelihood_optimizer = optimizer[1]
            optimizer = optimizer[0]

        # default training
        history = self.engine.train(
            train_data=self.dataset.train_loader,
            val_data=self.dataset.train_val_loader,
            optimizer=optimizer,
            likelihood_optimizer=likelihood_optimizer,
            loss=self.model.loss,
            epochs=self.train_cfg.epochs,
            eval_every=eval_every,
            eval_train=True,
            callbacks=callbacks,
            metrics=metrics,
            gpu=self.run_cfg.gpu,
        )

        # ------------------------------------------------------------------------------------------------
        # finetuning
        finetune_epochs = (
            0
            if self.train_cfg.finetune_epochs is None
            else self.train_cfg.finetune_epochs
        )
        if finetune_epochs > 0:
            # set-up optimizer
            optimizer = self.model.get_finetune_optimizer(
                self.train_cfg.lr, self.train_cfg.weight_decay
            )
            likelihood_optimizer = None

            self.engine.model.set_finetuning(True)
            _ = self.engine.train(
                train_data=self.dataset.finetune_loader,
                val_data=self.dataset.train_val_loader,
                optimizer=optimizer,
                likelihood_optimizer=None,
                loss=self.model.finetune_loss,
                epochs=self.train_cfg.finetune_epochs,
                eval_every=eval_every,
                eval_train=True,
                callbacks=warmup_callbacks,
                metrics=metrics,
                gpu=self.run_cfg.gpu,
            )
            self.engine.model.set_finetuning(False)

        self.dataset.train_dataset.to("cpu")
        self.dataset.train_val_dataset.to("cpu")
        self.dataset.warmup_dataset.to("cpu")
        self.dataset.finetune_dataset.to("cpu")
        self.training_completed = True

        return history

    def run(self) -> Dict[str, Any]:
        exp_params = ", ".join(
            [
                f"model={self.model_cfg.model_name}",
                f"dataset={self.data_cfg.dataset}",
                f"ood_type={self.data_cfg.ood_type}",
                f"split={self.data_cfg.split_no}",
                f"init={self.model_cfg.init_no}",
                f"results={self.results_file_path}"
            ]
        )
        print(f"Starting experiment ({exp_params}).")
        if self.run_cfg.job == "train":
            self.train()

        if self.data_cfg.ood_flag:
            results = self.evaluate_ood()
        else:
            results = self.evaluate()

        # save trained model
        # or potential values to be cached
        # e.g. alpha_prior of y_soft
        if (
            self.model is not None
            and self.run_cfg.save_model
            and self.training_completed
        ):
            self.model.save_to_storage()

        print(f"Completed experiment ({exp_params}).")

        return results
