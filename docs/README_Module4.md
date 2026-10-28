# Модуль 4. Инференс и упаковка

Скрипт: `scripts/m4_evaluate.py`. Код: `eos/model.py` (`EOSInference`, `EOSPredict`), `eos/wrapper.py` (`EOSModel`).

```bash
python scripts/m4_evaluate.py --models vanilla pinn
```

## Что делает скрипт
1. Оценивает бейзлайн (2 варианта), Vanilla и PINN на test (центры ячеек, 278 537 точек) и по областям.
2. Проверяет физику на test и на 200 000 случайных точек, равномерных по ln ρ × ln T во всей области.
3. Сверяет явные производные с `torch.autograd.grad` на 20 000 test-точек (п. 2.3):
   медианное относительное расхождение 5·10⁻⁷ – 9·10⁻⁶ (точность float32).
4. Экспортирует (п. 4.3):
   * `artifacts/models/pinn.ts.pt` — TorchScript `EOSInference`: (ρ, T) → [P, E, ∂P/∂ρ, ∂P/∂T, ∂E/∂ρ, ∂E/∂T, c_s²];
   * `artifacts/models/pinn.onnx` — то же в ONNX (opset 17), производные считаются внутри графа явным якобианом (эквивалент autograd, т.к. сам autograd в ONNX не экспортируется);
   * `artifacts/models/pinn_predict.onnx` — только (P, E);
   * `artifacts/models/pinn_state_dict.pt` — state_dict (п. 4.1).
   Максимальное расхождение ONNX vs PyTorch по P — 0 (до float32).
5. Меряет размер (п. 4.1) и время на 10⁵ случайных запросов (п. 4.2).
6. Строит графики и итоговую таблицу `artifacts/tables/comparison.{csv,md}` (п. 4.5), полные результаты — `results.json`.

## Размер и скорость (PINN, 2 ядра CPU, 10⁵ запросов)

| | время |
|---|---|
| P, E — PyTorch eager | ≈ 290–440 мс |
| P, E — ONNX Runtime | ≈ 120 мс |
| + все производные и c_s² — явный якобиан, eager | ≈ 0.9–1.0 с |
| + производные — autograd | ≈ 1.0–1.3 с |
| + производные — TorchScript | ≈ 0.6–0.75 с |
| + производные — ONNX Runtime | ≈ 0.7 с |

Размер: state_dict 0.29 МБ (74 882 параметра), ONNX 0.3–0.6 МБ — в 15–35 раз меньше лимита 10 МБ и
в 15 раз меньше прореженной таблицы бейзлайна (4.4 МБ; полная таблица ≈ 17 МБ).

## Класс-обёртка `EOSModel` (п. 4.4)

```python
from eos.wrapper import EOSModel
eos = EOSModel("artifacts/models/pinn.onnx")        # .onnx — без torch; .ts.pt / .pt — через torch
P, E   = eos.predict(rho, T)                         # ГПа, кДж/г
d      = eos.derivatives(rho, T)                     # dPdr, dPdT, dEdr, dEdT (=C_v), cs2
cs     = eos.sound_speed(rho, T)                     # км/с (NaN, если c_s² ≤ 0)
flags  = eos.check_physics(rho, T)                   # dict булевых массивов, True = нарушение
```

Проверки входа: `ValueError` при ρ ≤ 0, T ≤ 0, NaN/inf или разной форме массивов; `RuntimeWarning`
при выходе за область обучения (ρ ∈ [10⁻³, 10⁴] г/см³, T ∈ [0.1, 10⁷] K). Батчирование — по 2·10⁵ точек.

`check_physics` возвращает: `P_le0`, `dPdrho_le0`, `Cv_lt0`, `cs2_le0`, `cs_ge_c`, `maxwell` (|r_M| > 5 %),
`any_stability` и саму невязку `r_maxwell`.

Пример запуска — `python examples/api_example.py`:

```
ρ=    0.01 г/см³  T=   1e+04 K  →  P=     50.23 ГПа  E=    -8.392 кДж/г  c_s=     741 км/с  ...  устойчиво=True  r_M=-0.027
ρ=     2.7 г/см³  T=     300 K  →  P=       136 ГПа  E=    0.2407 кДж/г  c_s=    10.6 км/с  ...  устойчиво=True  r_M=-0.986
ρ=     2.7 г/см³  T=   1e+05 K  →  P=     956.8 ГПа  E=     258.5 кДж/г  c_s=      26 км/с  ...  устойчиво=True  r_M=+0.074
ρ=     100 г/см³  T=   1e+06 K  →  P= 8.354e+05 ГПа  E=      4685 кДж/г  c_s=     125 км/с  ...  устойчиво=True  r_M=-0.505
ValueError: rho и T должны быть конечными и > 0
```

## Графики
`m2_learning_curves.png`, `m2_error_heatmaps.png` (тепловые карты ошибок P и E), `m3_violation_maps.png`
(карта нарушений устойчивости), `m3_maxwell_maps.png`, `m3_pareto_maxwell.png`, `m3_{vanilla,pinn}_sections.png`
(сечения таблица vs модель).


## Обновления
* **Размеры:** state_dict и `*_predict.onnx` (только P, E) — 0.29 МБ; `pinn.onnx` / `pinn_thermo.onnx` с производными и c_s — 0.59 МБ
  (в граф развёрнут якобиан, веса хранятся и в транспонированном виде).
* **10⁸ запросов** (`scripts/m4_bench_1e8.py`, батчи по 10⁶, CPU 2 ядра): PINN 184 с, компактная PINN 65 с, таблица 45 с.
* **Вне области обучения** сеть экстраполирует без гарантий физики (при ρ = 10⁵ г/см³ давление меньше, чем при 10⁴):
  `check_physics` возвращает флаг `out_of_domain`, а `EOSModel(path, ood="nan")` возвращает NaN, как табличная интерполяция.
* **r_M = −0.986 при ρ = 2.7, T = 300 K** — свойство данных: у самой таблицы там r_M = −0.988 (минимум холодной энергии при P ≈ 136 ГПа).
