import math


def root_mean_squared_logarithmic_error(y_true, y_pred, a_min=1):
    i = 0
    sum = 0
    n = len(y_true)
    while i < n:
        true = y_true[i]
        pred = y_pred[i]
        if true < 0:
            return
        if pred < a_min:
            sum += (math.log(true) - math.log(a_min))**2
            i += 1
        else:
            sum += (math.log(true) - math.log(pred))**2
            i += 1
    return math.sqrt(sum / n)
