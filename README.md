# PINN-аппроксимация широкодиапазонного УРС алюминия с термодинамическими ограничениями

ИТ КоР 5.0, трек Machine Learning, кейс 02.

Нейросетевой суррогат уравнения состояния: (ρ, T) → (P, E) и все частные производные, c_s,
для ρ ∈ [10⁻³, 10⁴] г/см³, T ∈ [0.1, 10⁷] K — одна модель 0.3 МБ вместо 20 таблиц,
с практически полным выполнением условий термодинамической устойчивости во всей области.

## Команда и роли
| № | ФИО | Роль |
|---|---|---|
| 1 | Бронникова Екатерина Юрьевна | Капитан команды. Архитектура PINN, физический лосс; документация, презентация, финальная сборка решения |
| 2 | Ярутов Алексей Николаевич | Загрузка данных, EDA, baseline (билинейная интерполяция) |
| 3 | Мылицын Николай Романович | Обучение Vanilla-модели, подбор гиперпараметров |
| 4 | Боева Анастасия Владимировна | Физический блок: автоград производных, скорость звука |
| 5 | Семенова Анжелика Романовна | Инференс, упаковка (ONNX, TorchScript), класс-обёртка |

## Главные результаты

| модель | P, ср. отн. ошибка | E, ср. отн. ошибка | нарушения устойчивости* | невязка Максвелла (медиана) | 𝒢 < 0 | размер | 10⁵ запросов P,E / +производные |
|---|---|---|---|---|---|---|---|
| Билинейная интерполяция (ln P) | **0.012 %** | **0.027 %** | 1.9 % | 0.49 | 2.2 % | 4.4 МБ | 47 мс / 230 мс |
| Vanilla MLP | 0.22 % | 0.25 % | 28.4 % | 0.48 | 13.2 % | 0.29 МБ | 122 мс / 673 мс (ONNX) |
| **PINN-T «термодинамическая» (рекомендуемая)** | 11.9 % | 11.9 % | 0.082 % | **0.050** | 14.0 % | 0.29 / 0.59 МБ | 113 мс / 806 мс (ONNX) |
| PINN-A «точная» | 0.36 % | 0.59 % | **0.018 %** | 0.47 | 15.9 % | 0.29 / 0.59 МБ | 126 мс / 778 мс (ONNX) |
| Компактная PINN (модуль 5) | 1.0 % | 1.4 % | 0.048 % | — | — | **0.06 МБ** | **45 мс** / 203 мс |

\* доля из 200 000 случайных точек по всей области, где нарушено хотя бы одно из: (∂P/∂ρ)_T > 0, C_v ≥ 0,
c_s² > 0, c_s < c. Ошибки — на test-точках (центры ячеек сетки, не виденные ни бейзлайном, ни сетями).
Размер: 0.29 МБ — state_dict и ONNX «только P, E» (`*_predict.onnx`); 0.59 МБ — ONNX с производными и c_s.
𝒢 = 1 + (ρ/c_s)(∂c_s/∂ρ)_S — выпуклость изоэнтроп (доп. проверка, `scripts/m3_extra_physics.py`).
10⁸ запросов (CPU, 2 ядра): PINN 184 с, компактная PINN 65 с, таблица 45 с (`artifacts/tables/bench_1e8.json`).
Полная таблица: `artifacts/tables/comparison.md`, сжатие: `artifacts/tables/compression.md`, доп. физика: `artifacts/tables/extra_physics.json`.

**Какую модель брать.** Организаторы указали, что выполнение термодинамических ограничений важнее всего,
поэтому рекомендуемая модель — **PINN-T** (`artifacts/models/pinn_thermo.*`): она в 10 раз согласованнее самих таблиц
по Максвеллу и почти не нарушает устойчивость, ценой ~12 % отклонения от таблиц (сильнее всего — в холодных областях,
где таблицы P и E несовместимы). Если важнее близость к таблицам — **PINN-A** (`artifacts/models/pinn.*`).
Открытая проблема обеих: выпуклость изоэнтроп 𝒢 > 0 нарушена в 14–16 % точек (у таблиц 2.2 %).

**Ключевая находка EDA:** сами таблицы не удовлетворяют соотношению Максвелла ρ²∂E/∂ρ = P − T∂P/∂T
(медиана невязки 29 %; на холодной кривой — ровно в 2 раза). Поэтому устойчивость в PINN навязана полностью,
а согласованность по Максвеллу — штрафом с явно измеренным компромиссом (модуль 3, `m3_pareto_maxwell.png`).

## Архитектура решения

```
data/raw/Obl01..20.txt
   │  eos/data.py       загрузка, восстановление сеток, train/val/test, Normalizer (ln ρ, ln T → [-1,1]; ln P, asinh E)
   ▼
scripts/m1_eda_baseline.py   EDA, сечения, физика в таблицах, BilinearEOS (eos/baseline.py)
   ▼
scripts/m2_m3_train.py       EOSNet (Фурье-признаки + MLP SiLU) — Vanilla → PINN (дообучение с физикой)
   │  eos/model.py      forward_jac: значения + якобиан за один проход; Thermo: P,E,производные,c_s²,r_M
   │  eos/train.py      цикл обучения, коллокационные точки, physics_losses
   ▼
scripts/m4_evaluate.py       метрики, проверка autograd, экспорт TorchScript/ONNX, графики, comparison.md
scripts/m5_compress.py       TorchScript/ONNX/fp16/int8/компактная модель, compression.md
   ▼
eos/wrapper.py  EOSModel: predict / derivatives / sound_speed / check_physics
```

| модуль | документация |
|---|---|
| 1. Данные, EDA, бейзлайн | [docs/README_Module1.md](docs/README_Module1.md) |
| 2. Нейросетевая модель (Vanilla) | [docs/README_Module2.md](docs/README_Module2.md) |
| 3. Физические ограничения (PINN) | [docs/README_Module3.md](docs/README_Module3.md) |
| 4. Инференс и упаковка | [docs/README_Module4.md](docs/README_Module4.md) |
| 5. Сжатие и скорость | [docs/README_Module5.md](docs/README_Module5.md) |
| 6. Презентация | [docs/README_Module6.md](docs/README_Module6.md) |

Итоговая документация: [docs/documentation.docx](docs/documentation.docx) / [PDF](docs/documentation.pdf), презентация: [docs/presentation.pptx](docs/presentation.pptx) / [PDF](docs/presentation.pdf).
Скрипт итоговых визуализаций — `scripts/m4_evaluate.py` (графики EDA — `scripts/m1_eda_baseline.py`).

## Установка и запуск

```bash
pip install -r requirements.txt
# положить Obl01.txt … Obl20.txt в data/raw/
python run.py                 # полный пайплайн (CPU, ≈ 1.5–2 ч)
python run.py --skip-train    # только оценка/экспорт уже обученных моделей (≈ 15 мин)
python run.py --fast          # дымовой прогон (по 5 эпох)
pytest -q                     # тесты: якобиан = autograd, ONNX = PyTorch, валидация входа
```

## Использование API

```python
from eos.wrapper import EOSModel

eos = EOSModel("artifacts/models/pinn_thermo.onnx")     # PINN-T; ONNX Runtime, PyTorch не нужен
# eos = EOSModel("artifacts/models/pinn.onnx")          # PINN-A; ood="nan" — NaN вне области обучения
P, E = eos.predict(rho=[2.7, 100.0], T=[300.0, 1e6])     # ГПа, кДж/г
cs = eos.sound_speed([2.7], [300.0])                     # км/с
d = eos.derivatives([2.7], [300.0])                      # dPdr, dPdT, dEdr, dEdT (= C_v), cs2 (явный якобиан)
# EOSModel("artifacts/models/pinn.pt").derivatives_autograd(rho, T)   # те же производные через torch.autograd (п. 2.3)
flags = eos.check_physics([2.7], [300.0])                # {'Cv_lt0': [False], ..., 'out_of_domain': [False], 'r_maxwell': [...]}
```

Пример целиком: `python examples/api_example.py`.

## Артефакты
* `artifacts/models/pinn_thermo.*` (PINN-T) и `pinn.*` (PINN-A): `.pt` (чекпойнт), `_state_dict.pt`, `.ts.pt`, `.onnx`, `_predict.onnx` — модели ≤ 10 МБ;
* `artifacts/models/normalization.json` — коэффициенты нормирования;
* `artifacts/models/vanilla*.pt|onnx`, `pinn_small*`, `sweep/` — Vanilla, компактная PINN, серия по λ_Maxwell;
* `artifacts/tables/` — comparison, compression, pareto_maxwell, results.json, физика данных по областям;
* `artifacts/figures/` — все графики (скрипты m1 и m4 пересоздают их).

## Структура репозитория
```
eos/            библиотека (data, baseline, physics, metrics, model, train, wrapper, viz)
scripts/        m1_eda_baseline, m2_m3_train, m2_autograd_derivatives, m3_extra_physics, m4_evaluate, m4_bench_1e8, m5_compress
configs/        vanilla, pinn_finetune_sweep, pinn (PINN-A), pinn_thermo (PINN-T), pinn_small, pinn_consistent (эксперимент с 𝒢)
docs/           README_Module1..6, documentation.docx/pdf, presentation.pptx/pdf
examples/       api_example.py
tests/          test_core.py
artifacts/      модели, таблицы, графики
run.py, requirements.txt
```
