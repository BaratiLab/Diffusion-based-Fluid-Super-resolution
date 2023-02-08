import torch


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def ddim_steps(x, seq, model, b, **kwargs):
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    dx_func = kwargs.get('dx_func', None)
    clamp_func = kwargs.get('clamp_func', None)
    cache = kwargs.get('cache', False)

    logger = kwargs.get('logger', None)
    if logger is not None:
        logger.update(x=xs[-1])

    for i, j in zip(reversed(seq), reversed(seq_next)):
        with torch.no_grad():
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c2 = (1 - at_next).sqrt()
        if dx_func is not None:
            dx = dx_func(xt)
        else:
            dx = 0
        with torch.no_grad():
            xt_next = at_next.sqrt() * x0_t + c2 * et - dx
            if clamp_func is not None:
                xt_next = clamp_func(xt_next)
            xs.append(xt_next.to('cpu'))

        if logger is not None:
            logger.update(x=xs[-1])

        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]

    return xs, x0_preds


def ddpm_steps(x, seq, model, b,  **kwargs):
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    xs = [x]
    x0_preds = []
    betas = b
    dx_func = kwargs.get('dx_func', None)
    cache = kwargs.get('cache', False)
    clamp_func = kwargs.get('clamp_func', None)

    for i, j in zip(reversed(seq), reversed(seq_next)):
        with torch.no_grad():

            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()

        if dx_func is not None:
            dx = dx_func(x)
        else:
            dx = 0
        with torch.no_grad():
            sample = mean + mask * torch.exp(0.5 * logvar) * noise - dx
            if clamp_func is not None:
                sample = clamp_func(sample)
            xs.append(sample.to('cpu'))
        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]

    return xs, x0_preds


def guided_ddpm_steps(x, seq, model, b,  **kwargs):
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    xs = [x]
    x0_preds = []
    betas = b
    dx_func = kwargs.get('dx_func', None)
    if dx_func is None:
        raise ValueError('dx_func is required for guided denoising')
    clamp_func = kwargs.get('clamp_func', None)
    cache = kwargs.get('cache', False)
    w = kwargs.get('w', 3.0)

    for i, j in zip(reversed(seq), reversed(seq_next)):
        with torch.no_grad():

            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

        dx = dx_func(x)
        with torch.no_grad():

            output = (w+1)*model(x, t.float(), dx)-w*model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()


        with torch.no_grad():
            sample = mean + mask * torch.exp(0.5 * logvar) * noise - dx
            if clamp_func is not None:
                sample = clamp_func(sample)
            xs.append(sample.to('cpu'))
        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]

    return xs, x0_preds


def guided_ddim_steps(x, seq, model, b, **kwargs):
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    dx_func = kwargs.get('dx_func', None)
    if dx_func is None:
        raise ValueError('dx_func is required for guided denoising')
    clamp_func = kwargs.get('clamp_func', None)
    cache = kwargs.get('cache', False)
    w = kwargs.get('w', 3.0)
    logger = kwargs.get('logger', None)
    if logger is not None:
        logger.update(x=xs[-1])

    for i, j in zip(reversed(seq), reversed(seq_next)):
        with torch.no_grad():
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

        dx = dx_func(xt)

        with torch.no_grad():

            et = (w+1)*model(xt, t, dx) - w*model(xt, t)

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c2 = (1 - at_next).sqrt()

        with torch.no_grad():
            xt_next = at_next.sqrt() * x0_t + c2 * et - dx
            if clamp_func is not None:
                xt_next = clamp_func(xt_next)
            xs.append(xt_next.to('cpu'))

        if logger is not None:
            logger.update(x=xs[-1])

        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]

    return xs, x0_preds