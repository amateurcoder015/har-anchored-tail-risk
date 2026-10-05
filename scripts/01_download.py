from volgate.config import REPO_ROOT, load_config
from volgate.data.download import download_raw

cfg = load_config()
tickers = {**cfg["assets"], "india_vix": cfg["vix"]}
manifest = download_raw(tickers, cfg["start"], cfg["end"], REPO_ROOT / cfg["paths"]["raw"])
print(manifest[["key", "rows", "first", "last"]].to_string(index=False))
