"""plots.py — Vẽ đồ thị huấn luyện và so sánh thí nghiệm.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    """
    cfg = result["cfg"]
    history = result["history"]
    summary = result.get("summary", {})

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    epochs = history.get("epoch", [])
    if not epochs:
        print(f"Cảnh báo: không có dữ liệu epoch để vẽ cho {cfg.get('exp_id', 'unknown')}")
        return

    best_epoch = summary.get("best_epoch", 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    opt_name = cfg.get("optimizer", "")
    lr = cfg.get("lr", "")
    batch = cfg.get("batch", "")
    loss_name = cfg.get("loss", "")
    title_str = (
        f"Thí nghiệm: {cfg.get('exp_id', '')} ({cfg.get('group', '')})\n"
        f"Cấu hình: opt={opt_name}, lr={lr}, batch={batch}, loss={loss_name}, init={cfg.get('init', '')}"
    )
    fig.suptitle(title_str, fontsize=12, fontweight="bold")

    # Ô 1: Train Loss vs Val Loss
    ax1 = axes[0]
    ax1.plot(epochs, history["train_loss"], label="Train Loss (eval mode)", color="tab:blue", marker="o", markersize=4)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="tab:orange", marker="s", markersize=4)
    ax1.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Đường cong Loss (Train vs Val)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Ô 2: Val Accuracy & Macro-F1
    ax2 = axes[1]
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="tab:green", marker="^", markersize=4)
    if "val_macro_f1" in history:
        ax2.plot(epochs, history["val_macro_f1"], label="Val Macro-F1", color="tab:red", marker="d", markersize=4)
    ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Hiệu năng trên Validation (Acc & Macro-F1)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Ô 3: Gradient Norm (trước khi clip)
    ax3 = axes[2]
    ax3.plot(epochs, history["grad_norm"], label="Grad Norm (L2 trước clip)", color="tab:purple", marker="v", markersize=4)
    clip_val = cfg.get("clip_norm")
    if clip_val is not None:
        ax3.axhline(clip_val, color="red", linestyle=":", label=f"Clip threshold (c={clip_val})")
    ax3.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Gradient L2 Norm")
    ax3.set_title("Chuẩn Gradient theo Epoch")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu biểu đồ: {p}")


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ 'val_loss', 'val_macro_f1', 'grad_norm') của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    if not results:
        return

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    metric_labels = {
        "val_loss": "Validation Loss",
        "train_loss": "Train Loss",
        "val_acc": "Validation Accuracy",
        "val_macro_f1": "Validation Macro-F1",
        "grad_norm": "Gradient L2 Norm",
    }
    y_label = metric_labels.get(metric, metric)

    for r in results:
        exp_id = r["cfg"].get("exp_id", "exp")
        hist = r.get("history", {})
        epochs = hist.get("epoch", [])
        values = hist.get(metric, [])
        if epochs and values:
            ax.plot(epochs, values, marker="o", markersize=4, label=exp_id)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title if title else f"So sánh {y_label} giữa các thí nghiệm", fontsize=12, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu ảnh so sánh: {p}")
