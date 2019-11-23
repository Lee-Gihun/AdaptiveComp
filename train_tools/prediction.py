__all__ = ['ep_prediction']

def ep_prediction(outputs):
    preds = []
    if len(outputs) == 3:
        logits, features, confidence = outputs
        for logit in logits:
            _, pred = logit.max(dim=1)
            preds.append(pred)
            
    return preds