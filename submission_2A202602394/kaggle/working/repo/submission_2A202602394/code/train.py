"""train.py — Vòng lặp huấn luyện, đánh giá, dự đoán và ghi nhận thí nghiệm.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base).
DEFAULT_CFG = dict(
    exp_id="base-s1",
    group="baseline",
    description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # Đã chọn qua quét LR trên val (0.01 -> 0.1)
    weight_decay=0.0,
    momentum=0.9,
    batch=512,
    epochs=20,
    hidden=(256, 128),
    dropout=0.0,
    init="he",
    clip_norm=None,            # None = không clip; hoặc số thực, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    model.eval()
    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds.append(torch.argmax(logits, dim=1))
    return torch.cat(preds, dim=0)


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (lấy trung bình trên toàn bộ phần tử).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        y_one_hot = F.one_hot(y, num_classes=logits.shape[1]).float()
        return F.mse_loss(logits, y_one_hot)
    else:
        raise ValueError(f"Hàm mất mát không hợp lệ: '{loss_name}'. Chọn 'ce' hoặc 'mse'.")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    model.eval()
    n = len(X)
    total_loss = 0.0
    preds_list = []
    y_list = []

    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        loss = compute_loss(logits, yb, loss_name)
        total_loss += loss.item() * len(yb)
        preds_list.append(torch.argmax(logits, dim=1))
        y_list.append(yb)

    all_preds = torch.cat(preds_list, dim=0).cpu().numpy()
    all_y = torch.cat(y_list, dim=0).cpu().numpy()

    mean_loss = total_loss / n
    acc = float((all_preds == all_y).mean())

    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (all_y, all_preds), 1)
    macro_f1 = macro_f1_from_confusion(cm)

    return {"loss": mean_loss, "acc": acc, "macro_f1": macro_f1}


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    """
    set_seed(cfg["seed"])
    device = data["X_tr"].device
    is_cuda = device.type == "cuda"

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # 1. Khởi tạo mô hình
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init = str(cfg.get("init", "he"))
    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)

    # Kiểm tra số tham số khớp quy định
    n_params = count_params(model)
    if hidden in EXPECTED_PARAMS:
        assert n_params == EXPECTED_PARAMS[hidden], (
            f"Số tham số không khớp: nhận được {n_params}, kỳ vọng {EXPECTED_PARAMS[hidden]}"
        )

    # 2. Khởi tạo optimizer & scheduler
    optimizer = build_optimizer(
        name=cfg["optimizer"],
        params=model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
        momentum=cfg.get("momentum", 0.9),
    )

    precision = cfg.get("precision", "fp32")
    use_scaler = (precision == "fp16" and is_cuda)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    # Đo loss bước 0 trước khi cập nhật
    step0_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
    step0_loss = float(step0_res["loss"])

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = None
    diverged = False

    # Để đo train_loss chuẩn xác và nhanh: dùng toàn bộ train nếu có GPU, hoặc 50.000 mẫu nếu CPU
    X_tr_eval = data["X_tr"] if (is_cuda or len(data["X_tr"]) <= 50_000) else data["X_tr"][:50_000]
    y_tr_eval = data["y_tr"] if (is_cuda or len(data["y_tr"]) <= 50_000) else data["y_tr"][:50_000]

    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch", 512))
    clip_norm = cfg.get("clip_norm")

    for epoch in range(1, epochs + 1):
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        model.train()
        grad_norms_epoch = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size=batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if precision == "fp16" and is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
            elif precision == "bf16" and is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])

            # Kiểm tra NaN/inf
            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break

            if use_scaler and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                optimizer.step()

            grad_norms_epoch.append(gn)

        if is_cuda:
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - t0

        if diverged:
            print(f"[{cfg.get('exp_id', 'exp')}] Bị phân kỳ (NaN/Inf loss) tại epoch {epoch}!")
            break

        # Đánh giá cuối epoch ở chế độ eval
        train_res = evaluate(model, X_tr_eval, y_tr_eval, loss_name=cfg["loss"])
        val_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])

        mean_gn = float(np.mean(grad_norms_epoch)) if grad_norms_epoch else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_res["loss"]))
        history["val_loss"].append(float(val_res["loss"]))
        history["val_acc"].append(float(val_res["acc"]))
        history["val_macro_f1"].append(float(val_res["macro_f1"]))
        history["grad_norm"].append(mean_gn)
        history["epoch_time_s"].append(float(epoch_time))

        if val_res["loss"] < best_val_loss:
            best_val_loss = float(val_res["loss"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    # Thu thập bộ nhớ GPU cực đại (MB)
    peak_mem_MB = 0.0
    if is_cuda:
        peak_mem_MB = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    # Lấy metric tại best_epoch (1-indexed)
    if not diverged and best_state is not None:
        best_idx = best_epoch - 1
        best_val_acc = history["val_acc"][best_idx]
        best_val_f1 = history["val_macro_f1"][best_idx]
        final_tr_loss = history["train_loss"][-1]
        final_val_loss = history["val_loss"][-1]
        avg_time = float(np.mean(history["epoch_time_s"]))
    else:
        best_val_acc = 0.0
        best_val_f1 = 0.0
        final_tr_loss = float("nan")
        final_val_loss = float("nan")
        avg_time = float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0

    summary = {
        "step0_loss": step0_loss,
        "best_val_loss": best_val_loss if not diverged else float("nan"),
        "best_epoch": best_epoch,
        "final_train_loss": final_tr_loss,
        "final_val_loss": final_val_loss,
        "val_acc": best_val_acc,
        "val_macro_f1": best_val_f1,
        "time_per_epoch_s": avg_time,
        "peak_mem_MB": peak_mem_MB,
        "diverged": diverged,
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"row_id": row_id.astype(int), "pred": preds.astype(int)})
    df.to_csv(p, index=False)
    print(f"Đã ghi file dự đoán: {p} (số dòng: {len(df)})")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions."""
    device = data["X_eval"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init = str(cfg.get("init", "he"))

    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)
    model.load_state_dict(result["best_state"])
    model.eval()

    preds = predict(model, data["X_eval"])
    preds_np = preds.cpu().numpy()
    write_predictions(data["eval_row_id"], preds_np, pred_path)
