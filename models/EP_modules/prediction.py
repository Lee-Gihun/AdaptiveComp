__all__ = ['ep_prediction']

def ep_prediction(outputs):
    out1, out2 = outputs
    if type(out1) == 'list':
        exit, features = outputs
        _, preds = exit[-1].max(dim=1)
    else:
        output, mark = outputs
        _, preds = output.max(dim=1)
    return preds