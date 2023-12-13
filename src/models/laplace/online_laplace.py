from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch
import time
from torch.nn import functional as F

from pytorch_laplace import MSEHessianCalculator
from pytorch_laplace.laplace.diag import DiagLaplace


class OnlineLaplace:
    # Heavily inspired by Laplacian AE class (https://github.com/FrederikWarburg/LaplaceAE/blob/3ea27491a0ce3363186a80cc9cd88687d5688dea/src/laplace/onlinelaplace.py#L77)
    def __init__(self, net, dataset_size, cfg, register_forward_hook=False):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.cfg = cfg
        self.net = net.to(device)

        self.prior_prec = torch.tensor(float(cfg.models.online_laplace.prior_prec)).to(device)
        self.hessian_scale = torch.tensor(float(cfg.models.online_laplace.hessian_scale)).to(device)
        self.dataset_size = dataset_size
        self.n_samples = torch.tensor(float(cfg.models.online_laplace.train_samples)).to(device)
        self.one_hessian_per_sampling = torch.tensor(
            float(cfg.models.online_laplace.one_hessian_per_sampling)
        ).to(device)
        self.update_hessian = cfg.models.online_laplace.update_hessian
        self.weight_reg_factor = cfg.models.online_laplace.weight_reg_factor
        if cfg.models.online_laplace.use_learning_rate_for_memory_factor:
            self.hessian_memory_factor = 1 - cfg.hyperparameters.learning_rate
        else:
            self.hessian_memory_factor = cfg.models.online_laplace.hessian_memory_factor

        self.sigma_n = 1.0
        self.constant = 1.0 / (2 * self.sigma_n**2)

        if register_forward_hook:
            self.feature_maps = []

            def fw_hook_get_latent(module, input, output):
                self.feature_maps.append(output.detach())

            for k in range(len(self.net)):
                self.net[k].register_forward_hook(fw_hook_get_latent)

        self.hessian_calculator = MSEHessianCalculator(
            hessian_shape="diag", approximation_accuracy="approx"
        )
        self.hessian = torch.zeros_like(torch.nn.utils.parameters_to_vector(net.parameters()))
        self.laplace = DiagLaplace()

        # logging of time:
        self.timings = {
            "forward_nn": 0,
            "compute_hessian": 0,
            "entire_training_step": 0,
        }

    def __call__(self, x, train=True):
        self.timings["forward_nn"] = 0
        self.timings["compute_hessian"] = 0
        self.timings["entire_training_step"] = time.time()

        sigma_q = self.laplace.posterior_scale(self.hessian, self.hessian_scale, self.prior_prec)
        mu_q = parameters_to_vector(self.net.parameters())  # .unsqueeze(1)
        regularizer = weight_decay(mu_q, self.prior_prec)

        mse_running_sum = 0
        temp_hess = []
        preds = []

        # draw samples from the nn (sample nn)
        samples = self.laplace.sample_from_normal(mu_q, sigma_q, self.n_samples)
        for net_sample in samples:
            # replace the network parameters with the sampled parameters
            vector_to_parameters(net_sample, self.net.parameters())

            # reset or init
            self.feature_maps = []

            # predict with the sampled weights
            start = time.time()
            pred_sample = self.net(x)

            self.timings["forward_nn"] += time.time() - start

            assert pred_sample.shape[0, 2, 3] == x.shape[0, 2, 3]  # BxC/DxHxW
            # compute mse for sample net
            mse_running_sum += F.mse_loss(pred_sample.view(*x.shape), x)

            """ if (not self.one_hessian_per_sampling) and train:
                # compute hessian for sample net
                start = time.time()

                # H = J^T J
                h_s = self.hessian_calculator.__call__(self.net, self.feature_maps, x)
                h_s = self.laplace.scale(h_s, x.shape[0], self.dataset_size)

                self.timings["compute_hessian"] += time.time() - start

                # append results
                hessian.append(h_s) """
            preds.append(pred_sample)

        # reset the network parameters with the mean parameter (MAP estimate parameters)
        vector_to_parameters(mu_q, self.net.parameters())
        mse = mse_running_sum / self.n_samples

        if self.one_hessian_per_sampling and train:
            # reset or init
            self.feature_maps = []
            # predict with the sampled weights
            pred_sample = self.net(x)
            # compute hessian for sample net
            start = time.time()

            # H = J^T J
            h_s = self.hessian_calculator.compute_hessian(
                x=x,
                model=self.net,
            )
            temp_hess = [self.laplace.scale(h_s, x.shape[0], self.dataset_size)]

            self.timings["compute_hessian"] += time.time() - start

        if train:
            # take mean over hessian compute for different sampled NN
            temp_hess = self.average_hessian_samples(temp_hess, self.constant)

            if self.update_hessian:
                self.hessian = self.hessian_memory_factor * self.hessian + temp_hess
            else:
                self.hessian = (
                    1 - self.hessian_memory_factor
                ) * temp_hess + self.hessian_memory_factor * self.hessian

        loss = self.constant * mse + self.weight_reg_factor * regularizer

        # store some stuff for loggin purposes
        self.mse_loss = mse
        self.regularizer_loss = self.weight_reg_factor * regularizer
        self.x_recs = x_recs

        return loss

    def average_hessian_samples(self, hessian, constant):
        # average over samples
        hessian = torch.stack(hessian).mean(dim=0) if len(hessian) > 1 else hessian[0]

        # get posterior_precision
        return constant * hessian

    def load_hessian(self, path):
        self.hessian = torch.load(path)

    def save_hessian(self, path):
        torch.save(self.hessian, path)


def weight_decay(mu_q, prior_prec):
    return 0.5 * (torch.matmul(mu_q.T, mu_q) / prior_prec + torch.log(prior_prec))
