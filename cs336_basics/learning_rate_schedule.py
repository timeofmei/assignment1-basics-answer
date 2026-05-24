import math

def lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int):
    if it < warmup_iters:
        lr_t = it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        lr_t = min_learning_rate + (1 + math.cos(math.pi / (cosine_cycle_iters - warmup_iters) * (it - warmup_iters))) / 2 * (max_learning_rate - min_learning_rate)
    else:
        lr_t = min_learning_rate
    return lr_t
