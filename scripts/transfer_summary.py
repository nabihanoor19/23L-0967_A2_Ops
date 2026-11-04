from src.models import build_transfer, unfreeze_last_block


def trainable_report(model, title):
    base = model.get_layer("backbone")
    trainable = [l.name for l in base.layers if l.trainable]
    lines = [f"=== {title} ===",
             f"Trainable parameters (whole model): {sum(w.numpy().size for w in model.trainable_weights)}",
             f"Total parameters: {model.count_params()}",
             f"Trainable backbone layers: {len(trainable)} of {len(base.layers)}"]
    lines += [f"  - {n}" for n in trainable]
    lines += ["Trainable head layers: gap(no weights), dropout(no weights), prob_ai"]
    return "\n".join(lines)


model = build_transfer()
model.summary()
s1 = trainable_report(model, "Stage 1: backbone frozen, head only (lr = 1e-3)")
print(s1)

unfreeze_last_block(model)
s2 = trainable_report(model, "Stage 2: last block unfrozen (lr = 1e-5)")
print(s2)

open("reports/trainable_layers.txt", "w").write(s1 + "\n\n" + s2)