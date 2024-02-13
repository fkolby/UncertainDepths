import time

import torch
from pytorch_laplace.hessian.mse import MSEHessianCalculator
from pytorch_laplace.laplace.online_diag import OnlineDiagLaplace
from torch.nn import MSELoss
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import numpy as np


class OnlineLaplace:
    def __init__(self, net, dataset_size, loss_function, cfg, device, **kwargs):
        self.cfg = cfg
        self.net = net
        self.dataset_size = dataset_size
        self.loss_function = loss_function
        self.prior_prec = cfg.models.prior_prec
        self.hessian_scale = cfg.models.hessian_scale
        self.hessian_calculator = MSEHessianCalculator(
            hessian_shape="diag", approximation_accuracy="approx", backend="nnj"
        )
        self.n_samples = cfg.models.train_n_net_samples
        self.sampler = OnlineDiagLaplace()
        self.hessian = self.sampler.init_hessian(
            net=self.net,
            data_size=dataset_size * cfg.models.hessian_initial_multiplication_factor,
            device=device,
        )  # initializes the precision of n parameters at (\theta_1, \theta_2, ...,\theta_n) = dataset_size * (1,1,1,1,...,1)

        self.sigma_n = 1.0  # ask about this parameter
        self.constant = 1 / (2 * self.sigma_n**2)  # and this
        self.time_forward = 0.0
        self.time_hessian = 0.0
        self.time_rest = 0.0
        self.time_total = 1.0
        self.time_expected_tough_calc = 0.0
        self.time_append = 0.0
        self.time_second_hess_start = 0.0
        self.hessian_change_abs = 0.0
        self.hessian_change_signed = 0.0
        if isinstance(self.hessian_calculator, MSEHessianCalculator) and not isinstance(
            self.loss_function, MSELoss
        ):
            print("Hessian calc and loss function are not referring to same loss type")
            raise (NotImplementedError)

    def step(self, img, depth, epoch, train=True, **kwargs):
        hessian_memory_factor = kwargs.pop(
            "hessian_memory_factor", self.cfg.models.hessian_memory_factor
        )
        print(
            "epoch",
            epoch,
            str(self.cfg.trainer_args.max_epochs - (epoch + 1)),
            self.cfg.models.sample_last_n_epochs,
            "train",
            train,
            "dont_sample",
            self.cfg.models.dont_sample_parameters_during_training,
            flush=True,
        )

        if (
            self.cfg.models.update_hessian_probabilistically
        ):  # if updating every 10th time, to ensure hessian mem factor is a bound on |theta_t-theta_t-10|, it must be 10*learning_rate.
            alpha = 1 - hessian_memory_factor
            hessian_memory_factor = 1 - alpha

        total_time_start = time.time()
        sigma_q = self.sampler.posterior_scale(
            hessian=self.hessian, scale=self.hessian_scale, prior_prec=self.prior_prec
        )
        if self.cfg.models.dont_sample_parameters_during_training:
            mask = mask = torch.logical_and(
                depth >= self.cfg.dataset_params.min_depth,
                depth <= self.cfg.dataset_params.max_depth,
            )
            preds = self.net(img)
            loss = self.loss_function(preds[mask], depth[mask])
            preds_all_samples = [preds]
            hessian_tenth_qt = 13
            hessian_ninetieth_qt = 13
            hessian_size = 13
            hessian_median = 13

        else:
            mu_q = parameters_to_vector(self.net.parameters())

            if self.cfg.models.dont_sample_parameters_during_training and train:
                net_samples = mu_q.view(1, len(mu_q))

            elif (
                self.cfg.trainer_args.max_epochs - (epoch + 1)
                <= self.cfg.models.sample_last_n_epochs
            ) or not train:
                print("sampling", flush=True)
                net_samples = self.sampler.sample_from_normal(
                    mu=mu_q, scale=sigma_q, n_samples=self.n_samples
                )
            else:
                net_samples = mu_q.view(1, len(mu_q))

            loss_running_sum = 0

            mask = mask = torch.logical_and(
                depth >= self.cfg.dataset_params.min_depth,
                depth <= self.cfg.dataset_params.max_depth,
            )

            preds_all_samples = []

            temp_hessians = []
            for sample in net_samples:
                print("samplingactually", flush=True)
                vector_to_parameters(sample, self.net.parameters())
                forward_time_start = time.time()
                preds = self.net(img)
                self.time_forward += time.time() - forward_time_start
                preds_all_samples.append(
                    preds.unsqueeze(dim=0)
                )  # logging purposes, now dim=0 is model-dim, so preds has size (n_samples X batch X height X width)
                loss_running_sum += self.loss_function(preds[mask], depth[mask])

                hess_time_start = time.time()

                if train:
                    update_this_step = (not self.cfg.models.update_hessian_probabilistically) or (
                        np.random.rand() <= (1 / self.cfg.models.update_hessian_every)
                    )
                    print(
                        (1 / self.cfg.models.update_hessian_every),
                        (not self.cfg.models.update_hessian_probabilistically)
                        or (
                            np.random.rand() <= (1 / self.cfg.models.update_hessian_every),
                            np.random.rand(),
                            (np.random.rand() <= (1 / self.cfg.models.update_hessian_every)),
                        ),
                        flush=True,
                    )
                    print("samplingactually", flush=True)
                    if update_this_step:
                        print("updatehessian!", flush=True)
                        expected_tough_calc = time.time()
                        temp_hess = self.hessian_calculator.compute_hessian(
                            x=img, model=self.net, target=depth
                        )
                        self.time_expected_tough_calc += time.time() - expected_tough_calc
                        temp_hessians.append(temp_hess)
                        temp_hess = self.sampler.scale(
                            hessian_batch=temp_hess,
                            batch_size=img.shape[0],
                            data_size=self.dataset_size,
                        )  # temp_hess*dataset_size/batchsize # ask about this

                self.time_hessian += time.time() - hess_time_start

            hess_time_start = time.time()
            if train and update_this_step:
                temp_hessian = self.sampler.average_hessian_samples(
                    hessian=temp_hessians, constant=self.constant
                )  # mean(hessians)/constant #ask about this

                if self.cfg.models.use_exp_average_instead:
                    self.hessian = (
                        temp_hessian * (1 - hessian_memory_factor)
                        + hessian_memory_factor * self.hessian
                    )
                    self.hessian_change_abs = torch.mean(
                        torch.abs(
                            temp_hessian + hessian_memory_factor * self.hessian - self.hessian
                        )
                    )
                    self.hessian_change_signed = torch.mean(
                        temp_hessian + hessian_memory_factor * self.hessian - self.hessian
                    )

                else:
                    self.hessian_change_abs = torch.mean(
                        torch.abs(
                            temp_hessian + hessian_memory_factor * self.hessian - self.hessian
                        )
                    )
                    self.hessian_change_signed = torch.mean(
                        temp_hessian + hessian_memory_factor * self.hessian - self.hessian
                    )
                    self.hessian = torch.clamp(
                        temp_hessian + hessian_memory_factor * self.hessian,
                        min=self.hessian * 0.5 ** (1 / 1000) + 1e-8,
                        max=self.hessian * 2 ** (1 / 1000) + 1e-8,
                    )

            else:
                self.hessian_change_abs = torch.zeros_like(mu_q)
                self.change_signed = torch.zeros_like(mu_q)
            hessian_size = torch.mean(self.hessian)
            hessian_median = torch.median(self.hessian)
            hessian_tenth_qt = torch.quantile(
                self.hessian, q=torch.tensor([0.10], device=self.hessian.device)
            )
            hessian_ninetieth_qt = torch.quantile(
                self.hessian, q=torch.tensor([0.90], device=self.hessian.device)
            )

            self.time_hessian += time.time() - hess_time_start
            loss = loss_running_sum / self.n_samples * self.constant

            self.time_total += time.time() - total_time_start
            self.time_rest = self.time_total - self.time_hessian - self.time_forward

        return {
            "loss": loss,
            "preds": torch.stack(preds_all_samples, dim=0).mean(dim=0).squeeze(dim=0),
            "variance": torch.stack(preds_all_samples, dim=0).var(dim=0).squeeze(dim=0),
            "time_hessian": self.time_hessian / self.time_total,
            "time_forward": self.time_forward / self.time_total,
            "time_total": self.time_total,
            "time_rest": self.time_rest / self.time_total,
            "time_tough": self.time_expected_tough_calc / self.time_total,
            "size_of_change": self.hessian_change_signed,
            "abs_size_of_change": self.hessian_change_abs,
            "hessian_size": hessian_size,
            "hessian_median": hessian_median,
            "hessian_tenth_qt": hessian_tenth_qt,
            "hessian_ninetieth_qt": hessian_ninetieth_qt,
        }

    def save_hessian(self, path):
        torch.save(self.hessian, path)
