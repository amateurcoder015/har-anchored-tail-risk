import pandas as pd

from hart.compare import loss_frames
from hart.hypotheses import evaluate_hypotheses, one_sided_p
from volgate.config import REPO_ROOT, load_config
from volgate.evaluate.stats import dm_test

ALPHA = 0.025
SECONDARY = [("r2_mlp", "r0"), ("r2_linear", "r0"), ("r1", "r0")]
cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
P["tables"].mkdir(parents=True, exist_ok=True)
L = loss_frames(P["processed"], cfg["assets"], ALPHA)["ALL"]
primary = evaluate_hypotheses(L)
primary.to_csv(P["tables"] / "hypotheses.csv", index_label="hypothesis")
rows = []
for lower, higher in SECONDARY:
    s, p = dm_test(L[lower], L[higher])
    rows.append({"lower": lower, "higher": higher, "dm_stat": s, "p_one_sided": one_sided_p(s, p),
                 "mean_lower": L[lower].mean(), "mean_higher": L[higher].mean()})
pd.DataFrame(rows).to_csv(P["tables"] / "secondary.csv", index=False)
print(primary.round(4).to_string())
print(pd.DataFrame(rows).round(4).to_string(index=False))
