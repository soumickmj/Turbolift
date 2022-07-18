import torch

def dice_loss(pred, target):
    """This definition generalise to real valued pred and target vector.
        This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    nclasses = pred.shape[1]
    loss = 0.
    for c in range(nclasses):
        # have to use contiguous since they may from a torch.view op
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat*iflat)
        B_sum = torch.sum(tflat * tflat)
        loss += 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))
    return loss

def jaccard_loss(pred, target):
    """This definition generalise to real valued pred and target vector.
        This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1

    nclasses = pred.shape[1]
    loss = 0.
    for c in range(nclasses):
        # have to use contiguous since they may from a torch.view op
        iflat = pred[:,c].contiguous().view(-1)
        tflat = target[:,c].contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat*iflat)
        B_sum = torch.sum(tflat*tflat)
        
        jac = (intersection + smooth) / (A_sum + B_sum - intersection + smooth)
        loss += 1 - jac
    return loss

def focal_tversky_loss(pred, target, alpha=0.7, beta=0.3, gamma=4./3.):
    smooth = 1.

    nclasses = pred.shape[1]
    ftl = 0.
    for c in range(nclasses):
        pflat = pred[:,c].contiguous().view(-1)
        gflat = target[:,c].contiguous().view(-1)

        intersection = (pflat*gflat).sum()
        non_p_g = ((1.-pflat)*gflat).sum()
        p_non_g = (pflat*(1.-gflat)).sum()

        ti = (intersection + smooth)/(intersection + alpha*non_p_g + beta*p_non_g + smooth)
        ftl += (1. - ti)**(1./gamma)
    return ftl