import argparse
import sqlite3
from pathlib import Path

import matplotlib
import pandas as pd


# Use a headless backend so plotting works in remote/server environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(db_path: Path, table_name: str = "frame_telemetry") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()
    return df


def detect_latency_column(df: pd.DataFrame) -> str:
    for candidate in ["latency_ms", "latency"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No latency column found. Columns: {list(df.columns)}")


def build_summary(df: pd.DataFrame, latency_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["width", "height"], as_index=False)
        .agg(
            count=(latency_col, "count"),
            mean_latency_ms=(latency_col, "mean"),
            median_latency_ms=(latency_col, "median"),
            min_latency_ms=(latency_col, "min"),
            max_latency_ms=(latency_col, "max"),
            std_latency_ms=(latency_col, "std"),
        )
    )

    p95 = (
        df.groupby(["width", "height"])[latency_col]
        .quantile(0.95)
        .reset_index(name="p95_latency_ms")
    )

    grouped = grouped.merge(p95, on=["width", "height"], how="left")
    grouped["pixel_count"] = grouped["width"] * grouped["height"]
    grouped["resolution"] = grouped["width"].astype(str) + "x" + grouped["height"].astype(str)

    return grouped.sort_values("mean_latency_ms", ascending=False)


def save_plots(df: pd.DataFrame, summary: pd.DataFrame, latency_col: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_df = df.copy()
    plot_df["resolution"] = plot_df["width"].astype(str) + "x" + plot_df["height"].astype(str)

    order = (
        summary.sort_values("pixel_count", ascending=True)["resolution"].tolist()
    )

    # Boxplot of latency by resolution
    fig, ax = plt.subplots(figsize=(9, 5))
    data_by_res = [
        plot_df.loc[plot_df["resolution"] == res, latency_col].values for res in order
    ]
    ax.boxplot(data_by_res, tick_labels=order, showfliers=True)
    ax.set_title("Latency Distribution by Resolution")
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Latency (ms)")
    plt.tight_layout()
    fig.savefig(out_dir / "latency_boxplot_by_resolution.png", dpi=160)
    plt.close(fig)

    # Mean + p95 latency vs pixel count
    chart = summary.sort_values("pixel_count")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        chart["pixel_count"],
        chart["mean_latency_ms"],
        marker="o",
        label="Mean latency",
    )
    ax.plot(
        chart["pixel_count"],
        chart["p95_latency_ms"],
        marker="s",
        label="P95 latency",
    )
    for _, row in chart.iterrows():
        ax.annotate(
            row["resolution"],
            (row["pixel_count"], row["mean_latency_ms"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax.set_title("Latency vs Pixel Count")
    ax.set_xlabel("Pixel count (width x height)")
    ax.set_ylabel("Latency (ms)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "latency_vs_pixel_count.png", dpi=160)
    plt.close(fig)

    # Payload vs latency scatter by resolution
    fig, ax = plt.subplots(figsize=(9, 5))
    for res in order:
        subset = plot_df[plot_df["resolution"] == res]
        ax.scatter(subset["payload_kb"], subset[latency_col], label=res, alpha=0.75)
    ax.set_title("Payload Size vs Latency")
    ax.set_xlabel("Payload (KB)")
    ax.set_ylabel("Latency (ms)")
    ax.legend(title="Resolution")
    plt.tight_layout()
    fig.savefig(out_dir / "payload_vs_latency.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze frame telemetry data")
    parser.add_argument("--db", default="telemetry.db", help="Path to telemetry.db")
    parser.add_argument(
        "--out", default="analysis_outputs", help="Directory for tables and plots"
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    out_dir = Path(args.out)

    df = load_data(db_path)
    latency_col = detect_latency_column(df)

    summary = build_summary(df, latency_col)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary_by_resolution.csv", index=False)

    overall = {
        "rows": int(len(df)),
        "resolutions": int(df[["width", "height"]].drop_duplicates().shape[0]),
        "mean_latency_ms": float(df[latency_col].mean()),
        "median_latency_ms": float(df[latency_col].median()),
        "p95_latency_ms": float(df[latency_col].quantile(0.95)),
    }

    summary_text = [
        "Telemetry Summary",
        f"rows: {overall['rows']}",
        f"distinct resolutions: {overall['resolutions']}",
        f"overall mean latency (ms): {overall['mean_latency_ms']:.3f}",
        f"overall median latency (ms): {overall['median_latency_ms']:.3f}",
        f"overall p95 latency (ms): {overall['p95_latency_ms']:.3f}",
        "",
        "Per-resolution summary:",
        summary.to_string(index=False),
    ]

    (out_dir / "summary.txt").write_text("\n".join(summary_text), encoding="utf-8")

    save_plots(df, summary, latency_col, out_dir)

    print("Analysis complete.")
    print(f"Wrote: {out_dir / 'summary_by_resolution.csv'}")
    print(f"Wrote: {out_dir / 'summary.txt'}")
    print(f"Wrote: {out_dir / 'latency_boxplot_by_resolution.png'}")
    print(f"Wrote: {out_dir / 'latency_vs_pixel_count.png'}")
    print(f"Wrote: {out_dir / 'payload_vs_latency.png'}")


if __name__ == "__main__":
    main()
