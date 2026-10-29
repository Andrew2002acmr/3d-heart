# Learnable Feature Representation, Stage 03

Запуск существующего CT Lab: `.venv/Scripts/python.exe -m ct_lab.learnable_launch` из `repository`, затем http://127.0.0.1:8510 → Learnable Feature Explorer. GUI только читает encoder checkpoints, обучение автоматически не запускается. Source intensity не HU, spacing не мм.

Полный протокол и команды — `../LEARNABLE_FEATURE_EXPERIMENT_03.md`; результаты — `../outputs/learnable_feature_03/`. `learnable_train` фиксирует план, выполняет legacy A–G абляцию, извлекает 48 train/10 validation, повторяет A–G и обучает 36 небольших MLP на C/G. Существующие receipts/cache переиспользуются. Для независимого воспроизведения используйте отдельную копию, не удаляйте предыдущие результаты. CPU-only, float32 patches/halo, без dense full-volume feature tensor.

`learnable_diagnostics` — фиксированные perturbations/permutation; `learnable_artifacts` — train-fit PCA/UMAP, кривые; `learnable_verify_ui` — Streamlit AppTest; `learnable_browser.cjs` — isolated headless Chrome (использует existing web/node_modules/playwright); `learnable_package` — повторная проверка source/production/prior hashes, таблицы, отчёт и ZIP без numba cache.

Validation ранее использовалась и служит early stopping/model selection. Sealed test не открывается. Это representation learning на выбранных GT-labelled voxels, не whole-volume segmentation, не Dice и не клиническая оценка. Предыдущие CHD68 модели и production preprocessing остаются без изменений.
