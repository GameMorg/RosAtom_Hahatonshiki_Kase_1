"""Тесты: явный якобиан = autograd; согласованность обёртки; валидация входа.

Запуск: pytest -q
"""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eos.data import Normalizer  # noqa: E402
from eos.model import EOSInference, EOSNet, Thermo  # noqa: E402


@pytest.mark.parametrize("fourier", [0, 8])
def test_explicit_jacobian_matches_autograd(fourier):
    net = EOSNet(32, 3, "silu", fourier=fourier).double()
    x = (torch.rand(256, 2, dtype=torch.float64) * 2 - 1).requires_grad_()
    y, J = net.forward_jac(x)
    J_ag = torch.stack([torch.autograd.grad(y[:, i].sum(), x, retain_graph=True)[0] for i in range(2)], 1)
    assert torch.allclose(J, J_ag, atol=1e-12)
    assert torch.allclose(net(x), y)


def test_physical_derivatives_match_autograd():
    norm = Normalizer.fit(np.array([1e-3, 1e4]), np.array([0.1, 1e7]), np.array([1.0, 1e3]), np.array([-5.0, 1e3]))
    net, th = EOSNet(32, 3, fourier=4).double(), Thermo(norm).double()
    rho = torch.exp(torch.empty(200, dtype=torch.float64).uniform_(-6, 9)).requires_grad_()
    T = torch.exp(torch.empty(200, dtype=torch.float64).uniform_(-2, 16)).requires_grad_()
    out = EOSInference(net, th)(rho, T)
    P, E = th.values(net(th.x(rho, T)))[:2]
    gP = torch.autograd.grad(P.sum(), (rho, T), retain_graph=True)
    gE = torch.autograd.grad(E.sum(), (rho, T))
    for col, g in zip((2, 3, 4, 5), (*gP, *gE)):
        assert torch.allclose(out[:, col], g, rtol=1e-8, atol=1e-12)


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "artifacts/models/pinn.onnx")), reason="нет модели")
def test_wrapper_onnx_vs_torch_and_validation():
    from eos.wrapper import EOSModel
    m1 = EOSModel(os.path.join(ROOT, "artifacts/models/pinn.onnx"))
    m2 = EOSModel(os.path.join(ROOT, "artifacts/models/pinn.pt"))
    rho, T = np.array([0.01, 2.7, 300.0]), np.array([10.0, 3e3, 1e6])
    assert np.allclose(m1.predict(rho, T)[0], m2.predict(rho, T)[0], rtol=1e-4)
    with pytest.raises(ValueError):
        m1.predict([0.0], [1.0])
    with pytest.raises(ValueError):
        m1.predict([1.0], [-5.0])


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "artifacts/models/pinn.onnx")), reason="нет модели")
def test_out_of_domain_flag_and_nan_mode():
    import warnings
    from eos.wrapper import EOSModel
    m = EOSModel(os.path.join(ROOT, "artifacts/models/pinn.onnx"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = m.check_physics([2.7, 1e5], [300.0, 300.0])
        assert not f["out_of_domain"][0] and f["out_of_domain"][1]
        P, E = EOSModel(os.path.join(ROOT, "artifacts/models/pinn.onnx"), ood="nan").predict([2.7, 1e5], [300.0, 300.0])
    assert np.isfinite(P[0]) and np.isnan(P[1]) and np.isnan(E[1])


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "artifacts/models/pinn.pt")), reason="нет модели")
def test_wrapper_autograd_matches_explicit():
    from eos.wrapper import EOSModel
    m = EOSModel(os.path.join(ROOT, "artifacts/models/pinn.pt"))
    rho, T = np.array([0.01, 2.7, 300.0]), np.array([10.0, 3e3, 1e6])
    a, e = m.derivatives_autograd(rho, T), m.derivatives(rho, T)
    for k in ("dPdr", "dPdT", "dEdr", "dEdT"):
        assert np.allclose(a[k], e[k], rtol=5e-3)  # float32


def test_training_step_with_autograd_derivatives():
    from eos.data import Normalizer
    from eos.train import TrainConfig, train
    g = np.random.default_rng(0)
    n = 512
    rho = np.exp(g.uniform(np.log(1e-3), np.log(1e4), n)); T = np.exp(g.uniform(np.log(0.1), np.log(1e7), n))
    P = rho * (1 + T); E = np.log1p(T)
    d = dict(rho=rho, T=T, P=P, E=E, region=np.ones(n, int))
    norm = Normalizer.fit(rho, T, P, E)
    cfg = TrainConfig(width=16, depth=2, fourier=2, epochs=1, batch=256, n_colloc=64, lam_mech=1, lam_therm=1, lam_maxwell=1e-3, deriv="autograd")
    _, _, h = train(cfg, {"train": d, "val": d}, norm, log_every=1)
    assert np.isfinite(h[-1]["loss"])
