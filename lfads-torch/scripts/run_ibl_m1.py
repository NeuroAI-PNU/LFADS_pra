import os
import shutil
from datetime import datetime
from pathlib import Path

from lfads_torch.run_model import run_model

# ---------- OPTIONS -----------
PROJECT_STR = "lfads-torch-example"
DATASET_STR = "ibl_m1"
RUN_TAG = datetime.now().strftime("%y%m%d_%H%M%S") + "_ibl_m1"
RUN_DIR = Path("runs") / PROJECT_STR / DATASET_STR / RUN_TAG
OVERWRITE = False
# ------------------------------

# Overwrite the directory if necessary
if RUN_DIR.exists() and OVERWRITE:
    shutil.rmtree(RUN_DIR)
RUN_DIR.mkdir(parents=True)
# Copy this script into the run directory
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
# Switch to the `RUN_DIR` and train the model
os.chdir(RUN_DIR)
run_model(
    overrides={
        "datamodule": DATASET_STR,
        "model": DATASET_STR,
        "trainer.max_epochs": 10,
    },
    config_path="../configs/single.yaml",
)
#우선 max_epoch: 10은 학습이 정상적으로 실행되는지 확인하기 위한 테스트 설정(실제 분석에선 학습 과정을 확인한 뒤 epoch 수를 늘리면 됨.)
