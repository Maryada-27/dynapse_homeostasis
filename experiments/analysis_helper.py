import numpy as np
def spike_counts(spikes, duration = 1, interval_sec = 0.002):
    #return np.diff([np.count_nonzero(spikes < t) for t in np.arange(0, duration, interval_sec)])
    return np.histogram(spikes,bins=np.arange(0, duration ,interval_sec))[0]

def correlation_coefficient(spike1, spike2, duration= 1, interval_sec = 0.002):
    c1 = spike_counts(spike1, duration, interval_sec)
    c2 = spike_counts(spike2, duration, interval_sec)   
    return np.cov(c1,c2) / np.sqrt(np.var(c1) * np.var(c2))

def interspike_intervals(spikes):
    return np.diff(spikes)

def coefficient_variation(spikes, square=False):
    interspike = interspike_intervals(spikes)
    if square == True:
        return np.var(interspike) / np.power(np.mean(interspike),2)
    else:
        return np.std(interspike) / np.mean(interspike)
    
def fano_factor(spikes, duration, counting_intervals):
    ls = []
    for i in counting_intervals:
        counts = spike_counts(spikes, duration, i)
        ls.append(np.var(counts) / np.mean(counts))
    return np.asarray(ls)