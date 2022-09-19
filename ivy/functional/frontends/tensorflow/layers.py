import ivy


def MAE(a, b):
    differeneces = abs(a - b)
    mae = ivy.mean(differeneces)
    return mae
