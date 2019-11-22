__all__ = ['ep_prediction']

def ep_prediction(outputs):
    if len(outputs) == 3:
        exit, features, selection = outputs
        _, preds = exit[-1].max(dim=1)
    
    else:
        output, mark = outputs
        _, preds = output.max(dim=1)
    return preds