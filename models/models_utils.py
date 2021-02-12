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