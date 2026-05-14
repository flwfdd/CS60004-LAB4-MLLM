from transformers import AutoModel


def set_trainable_modules(
    model, train_vision=False, train_connector=True, train_language=False
):
    """
    根据模块名称控制 vision_model、mlp1、language_model 是否参与训练。
    """
    # ============================== 代码开始 ==============================
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "vision_model" in name:
            param.requires_grad = train_vision
        elif "mlp1" in name:
            param.requires_grad = train_connector
        elif "language_model" in name:
            param.requires_grad = train_language
    # ============================== 代码结束 ==============================
    return model


def count_parameters(model):
    """
    统计模型总参数量、可训练参数量和可训练参数占比。
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = (trainable_params / total_params) * 100 if total_params > 0 else 0
    return total_params, trainable_params, ratio


if __name__ == "__main__":
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    model = AutoModel.from_pretrained(
        str(root / "data/models/InternVL2-2B"),
        trust_remote_code=True,
    )

    configs = {
        "A_connector_only": dict(
            train_vision=False, train_connector=True, train_language=False
        ),
        "B_connector_language": dict(
            train_vision=False, train_connector=True, train_language=True
        ),
        "C_vision_connector": dict(
            train_vision=True, train_connector=True, train_language=False
        ),
        "D_full": dict(train_vision=True, train_connector=True, train_language=True),
    }

    for config_name, kwargs in configs.items():
        set_trainable_modules(model, **kwargs)
        total, trainable, ratio = count_parameters(model)
        print(f"--- Configuration: {config_name} ---")
        print(f"Total params: {total:,}")
        print(f"Trainable params: {trainable:,}")
        print(f"Trainable ratio: {ratio:.4f}%")
        print("-" * 40)
