import numpy as np
import pandas as pd


_ESCAPES = {"\\": "\\textbackslash{}", "_": "\\_", "%": "\\%", "&": "\\&", "#": "\\#", "$": "\\$"}


def _cell(v, digits):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        return "--" if np.isnan(v) else f"{v:.{digits}f}"
    return "".join(_ESCAPES.get(ch, ch) for ch in str(v))


def to_latex(df: pd.DataFrame, caption: str, label: str, digits: int = 3,
             note: str | None = None) -> str:
    """Booktabs table; NaN shows as --, underscores escaped, index is the first column."""
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", "\\begin{tabular}{l" + "r" * len(df.columns) + "}",
             "\\toprule", " & ".join(_cell(c, digits) for c in cols) + " \\\\", "\\midrule"]
    for idx, *row in df.itertuples(name=None):
        lines.append(" & ".join([_cell(idx, digits)] + [_cell(v, digits) for v in row]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines.append(f"\\par\\footnotesize {note}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"
