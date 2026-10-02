def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]],
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
    updated = []
    for params, grads, decay in zip(parameters, gradients, apply_to_all):
        if decay:
            new = [p * (1 - lr * weight_decay) - lr * g for p, g in zip(params, grads)]
        else:
            new = [p - lr * g for p, g in zip(params, grads)]
        updated.append(new)
    return updated