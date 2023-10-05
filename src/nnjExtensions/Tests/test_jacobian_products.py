"""
Test that the methods 
.jvp() .vjp() .jmp() .mjp() .jmjTp() .jTmjp()
return the correct tensor

NOTE: this tests assumes that each layer has the correct implementation of .jacobian()
"""

import torch

import nnj

# define some input data
xs = [
    torch.randn(7, 3),
    torch.ones(7, 3),
    torch.randn(7, 3) + torch.ones(7, 3),
    10 * torch.rand(7, 3),
]

# get all the layers
to_test_easy = [
    nnj.Linear(3, 5, bias=False),
    nnj.Linear(3, 5),
    nnj.Tanh(),
    nnj.ReLU(),
    nnj.Sigmoid(),
    nnj.Sinusoidal(),
    nnj.TruncExp(),
    nnj.Softplus(),
    nnj.nnjExtensions.flatten.flatten(),
    nnj.nnjExtensions.Conv2d.Conv2d(3,5,1,stride=2, padding=1),
    nnj.nnjExtensions.Conv2d.Conv2d(3,5,1),
]
to_test_advanced = [
    nnj.Sequential(nnj.Linear(3, 5), nnj.Tanh(), nnj.Linear(5, 13)),
    nnj.Sequential(nnj.Linear(3, 5), nnj.ReLU(), nnj.Linear(5, 13)),

    nnj.nnjExtensions.SkipConnect.SkipConnect(
        nnj.linear(3,10),
        nnj.nnjExtensions.Conv2d.Conv2d(10,4,1),
        nnj.nnjExtensions.Conv2d.Conv2d(10,4,1,strid=2,padding=1),
    )
    nnj.Sequential(
        nnj.Linear(3, 5),
        nnj.Tanh(),
        nnj.Linear(5, 2),
        nnj.Tanh(),
        nnj.Sinusoidal(),
        nnj.Linear(2, 12),
        nnj.Reshape(3, 4),
        nnj.Tanh(),
        nnj.Reshape(12),
        nnj.Softplus(),
        nnj.Tanh(),
    ),
    nnj.Sequential(
        nnj.Linear(3, 5),
        nnj.Tanh(),
        nnj.Sequential(
            nnj.Linear(5, 5),
            nnj.Tanh(),
            nnj.TruncExp(),
            nnj.Linear(5, 2),
        ),
        nnj.ReLU(),
        nnj.Linear(2, 13),
    ),
    nnj.Sequential(
        nnj.Linear(3, 5),
        nnj.Tanh(),
        nnj.Sequential(
            nnj.Sequential(
                nnj.Linear(5, 5),
                nnj.Tanh(),
                nnj.Linear(5, 3),
            ),
            nnj.Tanh(),
            nnj.Linear(3, 2),
        ),
        nnj.ReLU(),
        nnj.Sequential(
            nnj.Linear(2, 5),
            nnj.Tanh(),
            nnj.Linear(5, 2),
            nnj.TruncExp(),
            nnj.Softplus(),
        ),
        nnj.Linear(2, 13),
    ),
]


###################
# vector products #
###################


def test_jvp_wrt_input():
    for layer in to_test_easy + to_test_advanced:
        for x in xs:
            tangent_vector_input = torch.randn(*x.shape)

            jacobian = layer.jacobian(x, None, wrt="input")
            jvp_slow = torch.einsum("bij, bj -> bi", jacobian, tangent_vector_input)
            jvp_fast = layer.jvp(x, None, tangent_vector_input, wrt="input")

            assert jvp_fast.shape == jvp_slow.shape
            assert torch.isclose(jvp_fast, jvp_slow, atol=1e-4).all()


def test_jvp_wrt_weight():
    for layer in to_test_easy + to_test_advanced:
        for x in xs:
            batch_size = x.shape[0]
            tangent_vector_params = torch.randn((batch_size, layer._n_params))

            jvp_fast = layer.jvp(x, None, tangent_vector_params, wrt="weight")
            if layer._n_params == 0:
                assert jvp_fast is None
            else:
                jacobian = layer.jacobian(x, None, wrt="weight")
                jvp_slow = torch.einsum("bij, bj -> bi", jacobian, tangent_vector_params)

                assert jvp_fast.shape == jvp_slow.shape
                assert torch.isclose(jvp_fast, jvp_slow, atol=1e-4).all()

test_jvp_wrt_weight()
def test_vjp_wrt_input():