import ivy.functional.frontends.torch as torch_frontend


def lstm(*args, **kwargs):
    return torch_frontend.lstm(*args, **kwargs)
