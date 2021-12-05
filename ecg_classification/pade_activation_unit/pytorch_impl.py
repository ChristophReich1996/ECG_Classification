import torch
import torch.nn as nn
import numpy as np


def get_constants_for_inits(name, seed=17):
    # (numerator: [x, x.pow(1), x.pow(2), x.pow(3), x.pow(4, x.pow(5)], denominator: (x, x.pow(2), center)

    if name == "pade_sigmoid_3":
        return ((1 / 2, 1 / 4, 1 / 20, 1 / 240),
                (0., 1 / 10),
                (0,))
    elif name == "pade_sigmoid_5":
        return ((1 / 2, 1 / 4, 17 / 336, 1 / 224, 0, - 1 / 40320),
                (0., 1 / 10),
                (0,))
    elif name == "pade_softplus":
        return ((np.log(2), 1 / 2, (15 + 8 * np.log(2)) / 120, 1 / 30, 1 / 320),
                (0.01, 1 / 15),
                (0,))
    elif name == "pade_optimized_avg":
        return [(0.15775171, 0.74704865, 0.82560348, 1.61369449, 0.6371632, 0.10474671),
                (0.38940287, 2.19787666, 0.30977883, 0.15976778),
                (0.,)]
    elif name == "pade_optimized_leakyrelu":
        return [(3.35583603e-02, 5.05000375e-01, 1.65343934e+00, 2.01001052e+00, 9.31901999e-01, 1.52424124e-01),
                (3.30847488e-06, 3.98021568e+00, 5.12471206e-07, 3.01830109e-01),
                (0,)]
    elif name == "pade_optimized_leakyrelu2":
        return [(0.1494, 0.8779, 1.8259, 2.4658, 1.6976, 0.4414),
                (0.0878, 3.3983, 0.0055, 0.3488),
                (0,)]
    elif name == "pade_optmized":
        return [(0.0034586860882628158, -0.41459839329894876, 4.562452712166459, -16.314813244428276,
                 18.091669531543833, 0.23550876048241304),
                (3.0849791873233383e-28, 3.2072596311394997e-27, 1.0781647589819156e-28, 11.493453196161223),
                (0,)]


class PADEACTIVATION(nn.Module):

    def __init__(self, init_coefficients="pade_optimized_leakyrelu"):
        super(PADEACTIVATION, self).__init__()
        constants_for_inits = get_constants_for_inits(init_coefficients)

        self.n_numerator = len(constants_for_inits[0])
        self.n_denominator = len(constants_for_inits[1])

        self.weight_numerator = nn.Parameter(torch.FloatTensor(constants_for_inits[0]), requires_grad=True)
        self.weight_denominator = nn.Parameter(torch.FloatTensor(constants_for_inits[1]), requires_grad=True)

    def forward(self, x):
        raise NotImplementedError()


class PADEACTIVATION_F_python(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight_numerator, weight_denominator):
        ctx.save_for_backward(input, weight_numerator, weight_denominator)

        z = input

        clamped_n = weight_numerator
        clamped_d = weight_denominator.abs()

        numerator = z.mul(clamped_n[1]) + clamped_n[0]
        xps = list()
        # xp = z
        xps.append(z)
        for c_n in clamped_n[2:]:
            xp = xps[-1].mul(z)
            xps.append(xp)
            numerator = numerator + c_n.mul(xp)

        denominator = z.abs() * clamped_d[0] + 1
        for idx, c_d in enumerate(clamped_d[1:]):
            xp = xps[idx + 1].abs()
            denominator = denominator + c_d.mul(xp)

        return numerator.div(denominator)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight_numerator, weight_denominator = ctx.saved_tensors

        clamped_n = weight_numerator  # .clamp(min=0, max=1.)
        clamped_d = weight_denominator.abs()
        numerator = x.mul(clamped_n[1]) + clamped_n[0]
        xps = list()
        # xp = z
        xps.append(x)
        for c_n in clamped_n[2:]:
            xp = xps[-1].mul(x)
            xps.append(xp)
            numerator = numerator + c_n.mul(xp)

        denominator = x.abs() * clamped_d[0] + 1
        for idx, c_d in enumerate(clamped_d[1:]):
            xp = xps[idx + 1].abs()
            denominator = denominator + c_d.mul(xp)

        xps = torch.stack(xps)
        P = numerator
        Q = denominator
        dfdn = torch.cat(((1.0 / Q).unsqueeze(dim=0), xps.div(Q)))

        dfdd_tmp = (-P.div((Q.mul(Q))))
        dfdd = dfdd_tmp.mul(xps[0:clamped_d.size()[0]].abs())

        for idx in range(dfdd.shape[0]):
            dfdd[idx] = dfdd[idx].mul(weight_denominator[idx].sign())

        dfdx1 = 2.0 * clamped_n[2].mul(xps[0]) + clamped_n[1]
        for idx, xp in enumerate(xps[1:clamped_n.size()[0] - 2]):
            i = (idx + 3)
            dfdx1 += i * clamped_n[i].mul(xp)
        dfdx1 = dfdx1.div(Q)

        dfdx2 = 2.0 * clamped_d[1].mul(xps[0].abs()) + clamped_d[0]
        for idx, xp in enumerate(xps[1:clamped_d.size()[0] - 1]):
            i = (idx + 3)
            dfdx2 += i * clamped_d[idx + 2].mul(xp.abs())
        dfdx2_ = dfdx2.mul(xps[0].sign())
        dfdx2 = dfdx2_.mul(dfdd_tmp)

        dfdx = dfdx1 + dfdx2

        rdfdn = torch.mul(grad_output, dfdn)
        rdfdd = torch.mul(grad_output, dfdd)

        dfdn = rdfdn
        dfdd = rdfdd
        for _ in range(len(P.shape)):
            dfdn = dfdn.sum(-1)
            dfdd = dfdd.sum(-1)
        dfdx = grad_output.mul(dfdx)

        return dfdx, dfdn, dfdd


class PADEACTIVATION_Function_based(PADEACTIVATION):

    def __init__(self, init_coefficients="pade_optimized_leakyrelu", act_func_cls=None):
        super(PADEACTIVATION_Function_based, self).__init__(init_coefficients=init_coefficients)

        if act_func_cls is None:
            act_func_cls = PADEACTIVATION_F_python

        self.activation_function = act_func_cls.apply

    def forward(self, x):
        out = self.activation_function(x, self.weight_numerator, self.weight_denominator)
        return out
