from metrics import e_recall, a_recall, dists, rho_spectrum
from metrics import nmi, f1, mAP, mAP_c, mAP_1000


def select(metricname):
    if 'e_recall' in metricname:
        k = int(metricname.split('@')[-1])
        return e_recall.Metric(k)
    elif 'a_recall' in metricname:
        k = int(metricname.split('@')[-1])
        return a_recall.Metric(k)
    elif metricname=='nmi':
        return nmi.Metric()
    elif metricname=='mAP_c':
        return mAP_c.Metric()
    elif metricname=='mAP_1000':
        return mAP_1000.Metric()
    elif metricname=='mAP':
        return mAP.Metric()
    elif metricname=='f1':
        return f1.Metric()
    elif 'dists' in metricname:
        mode = metricname.split('@')[-1]
        return dists.Metric(mode)
    elif 'rho_spectrum' in metricname:
        # mode = int(metricname.split('@')[-1])
        # embed_dim = opt.embed_dim
        # return rho_spectrum.Metric(embed_dim, mode=mode, opt=opt)
        NotImplementedError("Metric {} is currently not supported!".format(metricname))
    else:
        raise NotImplementedError("Metric {} not available!".format(metricname))
