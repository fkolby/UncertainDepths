import torch
import torch.nn.functional as F
from torch import nn, Tensor
from nnj import AbstractJacobian
from typing import Optional, Tuple, List, Union, Literal


def compute_reversed_padding(padding, kernel_size=1):
    return kernel_size - 1 - padding


class Conv2d(AbstractJacobian, nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode
        )
        
        self._n_params = sum([torch.numel(w) for w in list(self.parameters())])
        dw_padding_h = compute_reversed_padding(self.padding[0], kernel_size=self.kernel_size[0])
        dw_padding_w = compute_reversed_padding(self.padding[1], kernel_size=self.kernel_size[1])
        self.dw_padding = (dw_padding_h, dw_padding_w)

    @torch.no_grad()
    def jacobian(self,x:Tensor, val: Union[None, Tensor], wrt= Literal["weight", "input"]):
        if wrt == "input":
            return _jacobian_wrt_input(x, val)
        
        elif wrt == "weight":
            if val is None:
                return self.forward(x)
            else:
                return _jacobian_wrt_weight(x)

    @torch.no_grad()
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        return (
            F.conv2d(
                jac_in.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c1, h1, w1),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            .reshape(b, *jac_in.shape[4:], c2, h2, w2)
            .movedim((-3, -2, -1), (1, 2, 3))
        )

    @torch.no_grad()
    def _jacobian_wrt_input(self, x: Tensor, val: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        output_identity = torch.eye(c1 * h1 * w1).unsqueeze(0).expand(b, -1, -1)
        output_identity = output_identity.reshape(b, c1, h1, w1, c1 * h1 * w1)

        # convolve each column
        jacobian = self._jacobian_wrt_input_mult_left_vec(x, val, output_identity)

        # reshape as a (num of output)x(num of input) matrix, one for each batch size
        jacobian = jacobian.reshape(b, c2 * h2 * w2, c1 * h1 * w1)

        return jacobian
    
    @torch.no_grad()
    def _jacobian_wrt_weight(self, x: Tensor, val: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        kernel_h, kernel_w = self.kernel_size

        output_identity = torch.eye(c2 * c1 * kernel_h * kernel_w)
        # expand rows as [(input channels)x(kernel height)x(kernel width)] cubes, one for each output channel
        output_identity = output_identity.reshape(c2, c1, kernel_h, kernel_w, c2 * c1 * kernel_h * kernel_w)

        reversed_inputs = torch.flip(x, [-2, -1]).movedim(0, 1)

        # convolve each base element and compute the jacobian
        jacobian = (
            F.conv_transpose2d(
                output_identity.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c1, kernel_h, kernel_w),
                weight=reversed_inputs,
                bias=None,
                stride=self.stride,
                padding=self.dw_padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=0,
            )
            .reshape(c2, *output_identity.shape[4:], b, h2, w2)
            .movedim((-3, -2, -1), (1, 2, 3))
        )

        # transpose the result in (output height)x(output width)
        jacobian = torch.flip(jacobian, [-3, -2])
        # switch batch size and output channel
        jacobian = jacobian.movedim(0, 1)
        # reshape as a (num of output)x(num of weights) matrix, one for each batch size
        jacobian = jacobian.reshape(b, c2 * h2 * w2, c2 * c1 * kernel_h * kernel_w)
        return jacobian

    @torch.no_grad()
    def _jvp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        jacobian vector product
        """
        if wrt == "input":
            raise NotImplementedError
        elif wrt == "weight":
            raise NotImplementedError

    @torch.no_grad()
    def _vjp(self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input") -> Tensor:
        """
        vector jacobian product
        """
        if wrt == "input":
            return self._jacobian_wrt_input_mult_left(x, val, vector.unsqueeze(1)).squeeze(1)
        elif wrt == "weight":
            if self.bias is None:
                return self._jacobian_wrt_weight_mult_left(x, val, vector.unsqueeze(1)).squeeze(1)
            else:
                b_term = torch.einsum("bchw->bc", vector.reshape(val.shape))
                return torch.cat(
                    [self._jacobian_wrt_weight_mult_left(x, val, vector.unsqueeze(1)).squeeze(1), b_term],
                    dim=1,
                )

    @torch.no_grad()
    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        if wrt == "input":
            b, c1, h1, w1 = x.shape
            c2, h2, w2 = val.shape[1:]
            assert matrix.shape[1] == c1 * h1 * w1
            n_col = matrix.shape[2]
            return (
                F.conv2d(
                    matrix.movedim((1), (-1)).reshape(-1, c1, h1, w1),
                    weight=self.weight,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                .reshape(b, n_col, c2 * h2 * w2)
                .movedim((1), (-1))
            )
        elif wrt == "weight":
            raise NotImplementedError

    @torch.no_grad()
    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        if wrt == "input":
            return self._jacobian_wrt_input_mult_left(x, val, matrix)
        elif wrt == "weight":
            if self.bias is None:
                return self._jacobian_wrt_weight_mult_left(x, val, matrix)
            else:
                b, c, h, w = val.shape
                b_term = torch.einsum("bvchw->bvc", matrix.reshape(b, -1, c, h, w))
                return torch.cat([self._jacobian_wrt_weight_mult_left(x, val, matrix), b_term], dim=2)

    @torch.no_grad()
    def _jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Tensor:
        """
        jacobian.T matrix jacobian product
        """
        if val is None:
            val = self.forward(x)
        if matrix is None:
            matrix = torch.ones_like(val)
            from_diag = True
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return self._jacobian_wrt_input_sandwich_full_to_full(x, val, matrix)
            elif from_diag and not to_diag:
                # diag -> full
                raise NotImplementedError
            elif not from_diag and to_diag:
                # full -> diag
                raise NotImplementedError
            elif from_diag and to_diag:
                # diag -> diag
                return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, matrix)
        elif wrt == "weight":
            if not from_diag and not to_diag:
                # full -> full
                if self.bias is None:
                    return self._jacobian_wrt_weight_sandwich_full_to_full(x, val, matrix)
                else:
                    matrix = self._mjp(x, val, matrix, wrt=wrt)
                    matrix = matrix.movedim(-2, -1)
                    matrix = self._mjp(x, val, matrix, wrt=wrt)
                    matrix = matrix.movedim(-2, -1)
                    return matrix
            elif from_diag and not to_diag:
                raise NotImplementedError
                # diag -> full