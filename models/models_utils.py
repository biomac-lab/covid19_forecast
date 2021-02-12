from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
import jax.numpy as np
import numpy as onp
import numpyro

def getter(f):
    '''
    Utility to define access method for time varying fields
    '''

    def get(self, samples, forecast=False):
        return samples[f + '_future'] if forecast else self.combine_samples(samples, f)
    return get

def nb2(mu=None, k=None):
    conc = 1./k
    rate = conc/mu
    return dist.GammaPoisson(conc, rate)

def observe_nb2(name, latent, det_prob, dispersion, obs=None):

    mask = True
    if obs is not None:
        mask = np.isfinite(obs) & (obs >= 0.0)
        obs = np.where(mask, obs, 0.0)

    det_prob = np.broadcast_to(det_prob, latent.shape)

    mean = det_prob * latent
    numpyro.deterministic("mean_" + name, mean)

    d = nb2(mu=mean, k=dispersion)

    with numpyro.handlers.mask(mask=mask):
        y = numpyro.sample(name, d, obs = obs)

    return y

def clean_daily_obs(obs, radius=2):
    '''Clean daily observations to fix negative elements'''

    # This searches for a small window containing the negative element
    # whose sum is non-negative, then sets each element in the window
    # to the same value to preserve the sum. This ensures that cumulative
    # sums of the daily time series are preserved to the extent possible.
    # (They only change inside the window, over which the cumulative sum is
    # linear.)

    orig_obs = obs

    # Subset to valid indices
    inds = onp.isfinite(obs)
    obs = obs[inds]

    obs = onp.array(obs)
    bad = onp.argwhere(obs < 0)

    for ind in bad:

        ind = ind[0]

        if obs[ind] >= 0:
            # it's conceivable the problem was fixed when
            # we cleaned another bad value
            continue

        left = ind - radius
        right = ind + radius + 1
        tot = onp.sum(obs[left:right])

        while tot < 0 and (left >= 0 or right <= len(obs)):
            left -= 1
            right += 1
            tot = onp.sum(obs[left:right])

        if tot < 0:
            raise ValueError("Couldn't clean data")

        n = len(obs[left:right])

        avg = tot // n
        rem = int(tot % n)

        obs[left:right] = avg
        obs[left:(left+rem)] += 1

    assert(onp.nansum(orig_obs) == onp.nansum(obs))

    orig_obs[inds] = obs
    return orig_obs