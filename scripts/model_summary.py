import pandas as pd
from src.models import build_baseline

model = build_baseline()
model.summary()

rows = [(l.name, l.__class__.__name__, str(l.output.shape), l.count_params())
        for l in model.layers]
table = pd.DataFrame(rows, columns=["layer", "type", "output_shape", "params"])
print(table.to_string(index=False))
print("Total params:", model.count_params())
table.to_csv("reports/baseline_layers.csv", index=False)