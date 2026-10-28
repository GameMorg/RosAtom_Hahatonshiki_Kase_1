"""Пример использования обёртки EOSModel (ONNX — torch не требуется)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eos.wrapper import EOSModel  # noqa: E402

eos = EOSModel("artifacts/models/pinn.onnx")
rho = np.array([0.01, 2.7, 2.7, 100.0])     # г/см³
T = np.array([1e4, 300.0, 1e5, 1e6])        # K

P, E = eos.predict(rho, T)
cs = eos.sound_speed(rho, T)
flags = eos.check_physics(rho, T)
d = eos.derivatives(rho, T)
for i in range(len(rho)):
    print(f"ρ={rho[i]:8.3g} г/см³  T={T[i]:8.3g} K  →  P={P[i]:10.4g} ГПа  E={E[i]:10.4g} кДж/г  "
          f"c_s={cs[i]:8.3g} км/с  C_v={d['dEdT'][i]:.3g} кДж/(г·K)  "
          f"устойчиво={not flags['any_stability'][i]}  r_M={flags['r_maxwell'][i]:+.3f}")
try:
    eos.predict([-1.0], [300.0])
except ValueError as e:
    print("ValueError:", e)

# вне области обучения: флаг out_of_domain и режим ood="nan"
f = eos.check_physics([1e5], [1e3])
print("ρ=1e5 г/см³ вне области:", bool(f["out_of_domain"][0]))
print("ood='nan':", EOSModel("artifacts/models/pinn.onnx", ood="nan").predict([1e5], [1e3]))
