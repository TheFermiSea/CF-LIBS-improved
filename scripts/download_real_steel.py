import os

os.makedirs("data/real_steel", exist_ok=True)
OUT = "data/real_steel/steel_266.parquet"


def summarize(rownames, row):
    print("columns:", rownames)
    for k in rownames:
        v = row[k]
        if isinstance(v, str):
            s = v[:60]
        elif hasattr(v, "__len__"):
            s = f"<array len {len(v)}> head={list(v[:3])}"
        else:
            s = v
        print(f"  {k}: {str(s)[:90]}")


try:
    from datasets import load_dataset

    ds = load_dataset("PhdYoda/steel_266_LIBS", split="train")
    print(f"loaded via datasets: {len(ds)} rows")
    ds.to_parquet(OUT)
    print(f"saved -> {OUT}")
    summarize(list(ds.features.keys()), ds[0])
except Exception as e:  # noqa: BLE001
    print("datasets path FAILED:", repr(e)[:200])
    print("NEXT: pip install datasets, or fall back to direct parquet download")
