__all__ = ['ep_prediction']

def ep_prediction(outputs):
    logits, features, confidence = outputs
    pred = logits[-1].max(dim=1)[1]
         
    return pred