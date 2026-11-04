import sys
from src.data import make_dataset
from src.models import build_baseline, build_transfer, unfreeze_last_block
from src.train import set_seed, run_training

which = sys.argv[1] if len(sys.argv) > 1 else "all"
train_ds = make_dataset("data/splits/train.csv", training=True)
val_ds = make_dataset("data/splits/val.csv")

if which in ("baseline", "all"):
    set_seed()
    run_training(build_baseline(), "baseline", train_ds, val_ds, epochs=30)

if which in ("transfer", "all"):
    set_seed()
    model = build_transfer()
    run_training(model, "transfer_stage1", train_ds, val_ds, epochs=20)
    unfreeze_last_block(model)                 # stage 2 continues from stage 1's best weights
    run_training(model, "transfer_stage2", train_ds, val_ds, epochs=20)