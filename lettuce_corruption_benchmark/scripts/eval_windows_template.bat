@echo off
REM Windows template. Edit model/config/checkpoint paths manually.
REM It is usually easier to run each model-condition command after editing.

set RESULT_DIR=results\raw_json
if not exist %RESULT_DIR% mkdir %RESULT_DIR%

REM Example:
REM python tools\test.py configs\maskrcnn_r50_lettuce.py work_dirs\maskrcnn_r50\best.pth --cfg-options test_dataloader.dataset.data_prefix.img=stress\clean\ > %RESULT_DIR%\maskrcnn_r50__clean.log 2>&1

echo Edit this file with your actual config and checkpoint paths.
