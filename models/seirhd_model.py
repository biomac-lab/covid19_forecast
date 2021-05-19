from jax.experimental.ode import odeint
from jax.random import PRNGKey
import jax.numpy as np
import jax

from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
import numpy as onp
import numpyro

import matplotlib.pyplot as plt
import pandas as pd

from .models_utils import getter, clean_daily_obs, observe_nb2

class CompartmentModel(object):
    '''
    Base class for compartment models.
    All class methods
    '''
    @classmethod
    def dx_dt(cls, x, *args):
        '''
        Compute time derivative
        '''
        raise NotImplementedError()


    @classmethod
    def run(cls, tspan, x0, theta, **kwargs):
        # Theta is a tuple of parameters. Entries are
        # scalars or vectors of length T-1
        is_scalar = [np.ndim(a)==0 for a in theta]
        if onp.all(is_scalar):
            return cls._run_static(tspan, x0, theta, **kwargs)
        else:
            return cls._run_time_varying(tspan, x0, theta, **kwargs)

    @classmethod
    def _run_static(cls, tmax, x0, theta, rtol=1e-5, atol=1e-3, mxstep=500):
        '''
        x0 is shape (d,)
        theta is shape (nargs,)
        '''
        t = np.arange(tmax, dtype='float') + 0.
        return odeint(cls.dx_dt, x0, t, *theta)

    @classmethod
    def _run_time_varying(cls, tmax, x0, theta, rtol=1e-5, atol=1e-3, mxstep=500):

        theta = tuple(np.broadcast_to(a, (tmax-1,)) for a in theta)

        '''
        x0 is shape (d,)
        theta is shape (nargs, T-1)
        '''
        t_one_step = np.array([0.0, 1.0])

        def advance(x0, theta):
            x1 = odeint(cls.dx_dt, x0, t_one_step, *theta, rtol=rtol, atol=atol, mxstep=mxstep)[1]
            return x1, x1

        # Run T–1 steps of the dynamics starting from the intial distribution
        _, X = jax.lax.scan(advance, x0, theta, tmax-1)
        return np.vstack((x0, X))

    @classmethod
    def R0(cls, theta):
        raise NotImplementedError()

    @classmethod
    def growth_rate(cls, theta):
        raise NotImplementedError()


class Model():

    names = {
        'S': 'susceptible',
        'I': 'infectious',
        'R': 'removed',
        'E': 'exposed',
        'H': 'hospitalized',
        'U': 'cumulative hospitalized',
        'D': 'dead',
        'C': 'cumulative infected',
        'y': 'confirmed',
        'z': 'deaths',
        'h': 'hospitalizations',
        'dy': 'daily confirmed',
        'dz': 'daily deaths',
        'dh': 'daily hospitalizations',
        'mean_dy': 'daily confirmed (mean)',
        'mean_dz': 'daily deaths (mean)',
        'mean_dh': 'daily hospitalizations (mean)'
    }

    def __init__(self, data=None, mcmc_samples=None, **args):
        self.mcmc_samples = mcmc_samples
        self.data = data
        self.args = args

    @property
    def obs(self):
        '''Provide extra arguments for observations

        Used during inference and forecasting
        '''
        return {}

    """
    ***************************************
    Inference and sampling routines
    ***************************************
    """
    def infer(self, num_warmup=1000, num_samples=1000, num_chains=1, rng_key=PRNGKey(1), **args):
        '''Fit using MCMC'''

        args = dict(self.args, **args)

        kernel = NUTS(self, init_strategy = numpyro.infer.initialization.init_to_uniform())

        mcmc = MCMC(kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains)

        mcmc.run(rng_key, **self.obs, **args)
        mcmc.print_summary()

        self.mcmc = mcmc
        self.mcmc_samples = mcmc.get_samples()

        return self.mcmc_samples

    def prior(self, num_samples=1000, rng_key=PRNGKey(2), **args):
        '''Draw samples from prior'''
        predictive = Predictive(self, posterior_samples={}, num_samples=num_samples)

        args = dict(self.args, **args) # passed args take precedence
        self.prior_samples = predictive(rng_key, **args)

        return self.prior_samples

    def predictive(self, rng_key=PRNGKey(3), **args):
        '''Draw samples from in-sample predictive distribution'''

        if self.mcmc_samples is None:
            raise RuntimeError("run inference first")

        predictive = Predictive(self, posterior_samples=self.mcmc_samples)

        args = dict(self.args, **args)
        return predictive(rng_key, **args)

    def forecast(self, num_samples=1000, rng_key=PRNGKey(4), **args):
        '''Draw samples from forecast predictive distribution'''

        if self.mcmc_samples is None:
            raise RuntimeError("run inference first")

        predictive = Predictive(self, posterior_samples=self.mcmc_samples)

        args = dict(self.args, **args)
        return predictive(rng_key, **self.obs, **args)


    def resample(self, low=0, high=90, rw_use_last=1, **kwargs):
        '''Resample MCMC samples by growth rate'''

        # TODO: hard-coded for SEIRDModel. Would also
        # work for SEIR, but not SIR

        beta = self.mcmc_samples['beta']
        gamma = self.mcmc_samples['gamma']
        sigma = self.mcmc_samples['sigma']
        beta_end = beta[:,-rw_use_last:].mean(axis=1)

        growth_rate = SEIRHDModel.growth_rate((beta_end, sigma, gamma))
        growth_rate = onp.array(growth_rate)
        low = int( (low/100) * len(growth_rate))
        high = int( (high/100) * len(growth_rate))

        sorted_inds = onp.argsort(growth_rate)
        selection = onp.random.randint(low, high, size=(1000))
        inds = sorted_inds[selection]

        new_samples = {k: v[inds, ...] for k, v in self.mcmc_samples.items()}

        self.mcmc_samples = new_samples
        return new_samples


    """
    ***************************************
    Data access and plotting
    ***************************************
    """
    def combine_samples(self, samples, f, use_future=False):
        '''Combine fields like x0, x, x_future into a single array'''

        f0, f_future = f + '0', f + '_future'
        data = np.concatenate((samples[f0][:,None], samples[f]), axis=1)
        if f_future in samples and use_future:
            data = np.concatenate((data, samples[f_future]), axis=1)
        return data

    def get(self, samples, c, **kwargs):

        forecast = kwargs.get('forecast', False)

        if c in self.compartments:
            x = samples['x_future'] if forecast else self.combine_samples(samples, 'x')
            j = self.compartments.index(c)
            return x[:,:,j]

        else:
            return getattr(self, c)(samples, **kwargs)  # call method named c


    def horizon(self, samples, **kwargs):
        '''Get time horizon'''
        y = self.y(samples, **kwargs)
        return y.shape[1]

    '''These are methods e.g., call self.z(samples) to get z'''
    #z = getter('z')
    #y = getter('y')
    mean_y = getter('mean_y')
    mean_z = getter('mean_z')
    mean_h = getter('mean_h')

    z = mean_z
    y = mean_y
    h = mean_h

    # There are only available in some models but easier to define here
    dz = getter('dz')
    dy = getter('dy')
    dh = getter('dh')
    mean_dy = getter('mean_dy')
    mean_dz = getter('mean_dz')
    mean_dh = getter('mean_dh')

class SEIRModel(CompartmentModel):

    @classmethod
    def dx_dt(cls, x, t, beta, sigma, gamma):
        """
        SEIR equations
        """
        S, E, I, R, C = x
        N = S + E + I + R

        sdot = - beta * S * I / N
        edot = beta * S * I / N - sigma * E
        idot = sigma * E - gamma * I
        rdot = gamma * I
        cdot = sigma * E  # incidence

        return np.stack([sdot, edot, idot, rdot, cdot])

    @classmethod
    def R0(cls, theta):
        beta, sigma, gamma = theta
        return beta / gamma

    @classmethod
    def growth_rate(cls, theta):
        '''
        Initial rate of exponential growth

        Reference: Junling Ma, Estimating epidemic exponential growth rate
        and basic reproduction number, Infectious Disease Modeling, 2020
        '''
        beta, sigma, gamma = theta
        return (-(sigma + gamma) + np.sqrt((sigma - gamma)**2 + 4 * sigma * beta))/2.


    @classmethod
    def seed(cls, N=1e6, I=100., E=0.):
        '''
        Seed infection. Return state vector for I exposed out of N
        '''
        return np.stack([N-E-I, E, I, 0.0, I])

class SEIRHDBase(Model):

    compartments = ['S', 'E', 'I', 'R', 'H', 'U', 'D', 'C']

    @property
    def obs(self):
        '''Provide extra arguments for observations

        Used during inference and forecasting
        '''
        if self.data is None:
            return {}

        return {
            'hospitalized': self.data['hospitalized'].values,
            'confirmed': self.data['confirmed'].values,
            'death': self.data['death'].values
           }

    def dz_mean(self, samples, **args):
        '''Daily deaths mean'''
        mean_z = self.mean_z(samples, **args)
        if args.get('forecast'):
            first = self.mean_z(samples, forecast=False)[:,-1,None]
        else:
            first = np.nan

        return onp.diff(mean_z, axis=1, prepend=first)

    def dz(self, samples, noise_scale=0.4, **args):
        '''Daily deaths with observation noise'''
        dz_mean = self.dz_mean(samples, **args)
        dz = dist.Normal(dz_mean, noise_scale * dz_mean).sample(PRNGKey(10))
        return dz

    def dh_mean(self, samples, **args):
        '''Daily hospitalizations mean'''
        mean_h = self.mean_h(samples, **args)
        if args.get('forecast'):
            first = self.mean_h(samples, forecast=False)[:,-1,None]
        else:
            first = np.nan

        return onp.diff(mean_h, axis=1, prepend=first)

    def dh(self, samples, noise_scale=0.4, **args):
        '''Daily hospitalizations with observation noise'''
        dh_mean = self.dh_mean(samples, **args)
        dh = dist.Normal(dh_mean, noise_scale * dh_mean).sample(PRNGKey(11))
        return dh

    def dy_mean(self, samples, **args):
        '''Daily confirmed cases mean'''
        mean_y = self.mean_y(samples, **args)
        if args.get('forecast'):
            first = self.mean_y(samples, forecast=False)[:,-1,None]
        else:
            first = np.nan

        return onp.diff(mean_y, axis=1, prepend=first)

    def dy(self, samples, noise_scale=0.4, **args):
        '''Daily confirmed cases with observation noise'''
        dy_mean = self.dy_mean(samples, **args)
        dy = dist.Normal(dy_mean, noise_scale * dy_mean).sample(PRNGKey(12))
        return dy

class SEIRHDModel(SEIRModel):

    @classmethod
    def dx_dt(cls, x, t, beta, sigma, gamma, hosp_prob, death_prob, death_rate, hosp_rate):
        """
        SEIRD equations
        """
        S, E, I, R, H, U, D, C = x
        N = S + E + I + R + H + D

        # sus -> exp
        s2e = beta * S * I / N
        # exp -> inf
        e2i = sigma * E
        # inf -> rec
        i2r = gamma * (1 - hosp_prob) * I
        # inf -> hosp
        i2h = hosp_rate * hosp_prob * I
        # hosp -> death
        h2d = death_rate * death_prob * H
        # hosp -> rec
        h2r = death_rate * (1 - death_prob) * H

        sdot = - s2e
        edot = s2e - e2i
        idot = e2i - i2r - i2h
        hdot = i2h - h2d - h2r
        rdot = h2r + i2r
        udot = i2h       # Cum hosp
        ddot = h2d       # Cum deaths
        cdot = s2e       # Cum incidence

        return np.stack([sdot, edot, idot, rdot, hdot, udot, ddot, cdot])

    @classmethod
    def seed(cls, N=1e6, I=100., E=0., R=0.0, U=0.0, H=0.0, D=0.0):
        return np.stack([N-E-I-R-H-D, E, I, R, H, U, D, I])


def frozen_random_walk(name, num_steps=100, num_frozen=10):

    # last random value is repeated frozen-1 times
    num_random = min(max(0, num_steps - num_frozen), num_steps)
    num_frozen = num_steps - num_random
    rw = numpyro.sample(name, dist.GaussianRandomWalk(num_steps=num_random))
    rw = np.concatenate((rw, np.repeat(rw[-1], num_frozen)))
    return rw

def LogisticRandomWalk(loc=1., scale=1e-2, drift=0., num_steps=100):
    '''
    Return distrubtion of exponentiated Gaussian random walk

    Variables are x_0, ..., x_{T-1}

    Dynamics in log-space are random walk with drift:
       log(x_0) := log(loc)
       log(x_t) := log(x_{t-1}) + drift + eps_t,    eps_t ~ N(0, scale)

    ==> Dynamics in non-log space are:
        x_0 := loc
        x_t := x_{t-1} * exp(drift + eps_t),    eps_t ~ N(0, scale)
    '''

    logistic_loc = np.log(loc/(1-loc)) + drift * (np.arange(num_steps)+0.)

    return dist.TransformedDistribution(
        dist.GaussianRandomWalk(scale=scale, num_steps=num_steps),
        [
            dist.transforms.AffineTransform(loc = logistic_loc, scale=1.),
            dist.transforms.SigmoidTransform()
        ]
    )


class SEIRHD(SEIRHDBase):

    def __call__(self,
                 T = 50,
                 N = 1e5,
                 T_future = 0,
                 E_duration_est = 4.0,
                 I_duration_est = 3.0,
                 H_duration_est = 15.0,
                 R0_est = 3.0,
                 beta_shape = 1.,
                 sigma_shape = 100.,
                 gamma_shape = 100.,
                 det_prob_est = 0.3,
                 det_prob_conc = 50.,
                 confirmed_dispersion=0.3,
                 death_dispersion=0.3,
                 hosp_prob  = 0.2,
                 death_prob = 0.03,
                 hosp_dispersion=0.3,
                 rw_scale = 2e-1,
                 det_noise_scale=0.15,
                 forecast_rw_scale = 0.,
                 drift_scale = None,
                 num_frozen=0,
                 rw_use_last=1,
                 confirmed=None,
                 death=None,
                 hospitalized=None):
        '''
        Stochastic SEIR model. Draws random parameters and runs dynamics.
        '''

        # Sample initial number of infected individuals
        #I0 = numpyro.sample("I0", dist.Uniform(0, 300))
        #E0 = numpyro.sample("E0", dist.Uniform(0, 300*0.1))
        #H0 = numpyro.sample("H0", dist.Uniform(0, 300*0.1))
        D0 = numpyro.sample("D0", dist.Uniform(0, 100))

        # Sample initial number of infected individuals
        I0 = numpyro.sample("I0", dist.Uniform(0, 1000))
        E0 = numpyro.sample("E0", dist.Uniform(0, 1000))
        H0 = numpyro.sample("H0", dist.Uniform(0, 1000))
        #D0 = numpyro.sample("D0", dist.Uniform(0, 0.02*N))


        # Sample dispersion parameters around specified values
        death_dispersion = numpyro.sample("death_dispersion",
                                           dist.TruncatedNormal(low=0.1,
                                                                loc=death_dispersion,
                                                                scale=0.15))

        hosp_dispersion = numpyro.sample("hospitalized_dispersion",
                                           dist.TruncatedNormal(low=0.1,
                                                                loc=hosp_dispersion,
                                                                scale=0.15))


        confirmed_dispersion = numpyro.sample("confirmed_dispersion",
                                              dist.TruncatedNormal(low=0.1,
                                                                   loc=confirmed_dispersion,
                                                                   scale=0.15))


        # Sample parameters
        sigma = numpyro.sample("sigma",
                               dist.Gamma(sigma_shape, sigma_shape * E_duration_est))

        gamma = numpyro.sample("gamma",
                                dist.Gamma(gamma_shape, gamma_shape * I_duration_est))


        beta0 = numpyro.sample("beta0",
                               dist.Gamma(beta_shape, beta_shape * I_duration_est/R0_est))

        det_prob0 = numpyro.sample("det_prob0",
                                   dist.Beta(det_prob_est * det_prob_conc,
                                            (1-det_prob_est) * det_prob_conc))

        det_prob_d = numpyro.sample("det_prob_d",
                                    dist.Beta(.9 * 100,
                                              (1-.9) * 100))

        det_prob_h = numpyro.sample("det_prob_h",
                                    dist.Beta(.95 * 100,
                                              (1-.95) * 100))

        death_prob = numpyro.sample("death_prob",
                                    dist.Beta(death_prob * 100, (1-death_prob) * 100))

        hosp_prob = numpyro.sample("hosp_prob",
                                    dist.Beta(hosp_prob * 100, (1-hosp_prob) * 100))
                                    # dist.Beta(0.02 * 1000, (1-0.02) * 1000))

        death_rate = numpyro.sample("death_rate",
                                    dist.Gamma(100, 100 * H_duration_est))

        hosp_rate = numpyro.sample("hosp_rate",
                                    dist.Gamma(100, 100 * H_duration_est+7))

        if drift_scale is not None:
            drift = numpyro.sample("drift", dist.Normal(loc=0., scale=drift_scale))
        else:
            drift = 0.


        x0 = SEIRHDModel.seed(N=N, I=I0, E=E0, H=H0, U=H0, D=D0)
        numpyro.deterministic("x0", x0)

        # Split observations into first and rest
        if confirmed is None:
            confirmed0, confirmed = (None, None)
        else:
            confirmed0 = confirmed[0]
            confirmed = clean_daily_obs(onp.diff(confirmed))

        if death is None:
            death0, death = (None, None)
        else:
            death0 = death[0]
            death  = clean_daily_obs(onp.diff(death))

        if hospitalized is None:
            hospitalized0, hospitalized = (None, None)
        else:
            hospitalized0 = hospitalized[0]
            hospitalized  = clean_daily_obs(onp.diff(hospitalized))

        # First observation
        with numpyro.handlers.scale(scale=0.5):
            y0 = observe_nb2("dy0", x0[7], det_prob0, confirmed_dispersion, obs=confirmed0)
            #y0 = observe("dy0", x0[6], det_prob0, det_noise_scale, obs=confirmed0)

        with numpyro.handlers.scale(scale=2.0):
            z0 = observe_nb2("dz0", x0[6], det_prob_d, death_dispersion, obs=death0)
            #z0 = observe("dz0", x0[5], det_prob_d, det_noise_scale, obs=death0)

        with numpyro.handlers.scale(scale=2.0):
            h0 = observe_nb2("dh0", x0[5], det_prob_h, hosp_dispersion, obs=hospitalized0)
            #z0 = observe("dz0", x0[5], det_prob_d, det_noise_scale, obs=death0)


        params = (beta0,
                  sigma,
                  gamma,
                  rw_scale,
                  drift,
                  det_prob0,
                  confirmed_dispersion,
                  death_dispersion,
                  hosp_dispersion,
                  hosp_prob,
                  death_prob,
                  death_rate,
                  hosp_rate,
                  det_prob_d,
                  det_prob_h)

        beta, det_prob, x, y, z, h = self.dynamics(T,
                                                params,
                                                x0,
                                                num_frozen = num_frozen,
                                                confirmed = confirmed,
                                                death = death,
                                                hospitalized = hospitalized)

        x = np.vstack((x0, x))
        y = np.append(y0, y)
        z = np.append(z0, z)
        h = np.append(h0, h)

        if T_future > 0:

            params = (beta[-rw_use_last:].mean(),
                      sigma,
                      gamma,
                      forecast_rw_scale,
                      drift,
                      det_prob[-rw_use_last:].mean(),
                      confirmed_dispersion,
                      death_dispersion,
                      hosp_dispersion,
                      hosp_prob,
                      death_prob,
                      death_rate,
                      hosp_rate,
                      det_prob_d,
                      det_prob_h)

            beta_f, det_rate_rw_f, x_f, y_f, z_f, h_f = self.dynamics(T_future+1,
                                                                 params,
                                                                 x[-1,:],
                                                                 suffix="_future")

            x = np.vstack((x, x_f))
            y = np.append(y, y_f)
            z = np.append(z, z_f)
            h = np.append(h, h_f)

        return beta, x, y, z, h, det_prob, death_prob

    def dynamics(self, T, params, x0, num_frozen=0, confirmed=None, death=None, hospitalized=None, suffix=""):
        '''Run SEIRD dynamics for T time steps'''
        beta0, \
        sigma, \
        gamma, \
        rw_scale, \
        drift, \
        det_prob0, \
        confirmed_dispersion, \
        death_dispersion, \
        hosp_dispersion, \
        hosp_prob, \
        death_prob, \
        death_rate, \
        hosp_rate,  \
        det_prob_d, \
        det_prob_h = params

        rw = frozen_random_walk("rw" + suffix,
                                num_steps=T-1,
                                num_frozen=num_frozen)

        beta = numpyro.deterministic("beta", beta0 * np.exp(rw_scale*rw))

        det_prob = numpyro.sample("det_prob" + suffix,
                                  LogisticRandomWalk(loc=det_prob0,
                                                     scale=rw_scale,
                                                     drift=0.,
                                                     num_steps=T-1))

        # Run ODE
        x = SEIRHDModel.run(T, x0, (beta, sigma, gamma, hosp_prob, death_prob, death_rate, hosp_rate))
        numpyro.deterministic("x" + suffix, x[1:])
        x_diff = np.diff(x, axis=0)

        # Noisy observations
        with numpyro.handlers.scale(scale=0.5):
            y = observe_nb2("dy" + suffix, x_diff[0:,7], det_prob, confirmed_dispersion, obs = confirmed)
            #y = observe("dy" + suffix, x_diff[:,6], det_prob, confirmed_dispersion, obs = confirmed)

        with numpyro.handlers.scale(scale=2.0):
            z = observe_nb2("dz" + suffix, x_diff[0:,6], det_prob_d, death_dispersion, obs = death)
            #z = observe("dz" + suffix, x_diff[:,5], det_prob_d, death_dispersion, obs = death)

        with numpyro.handlers.scale(scale=2.0):
            h = observe_nb2("dh" + suffix, x_diff[0:,5], det_prob_h, hosp_dispersion, obs = hospitalized)
            #h = observe_nb("dh" + suffix, x_diff[0:,4], det_prob_d, hosp_dispersion, obs = death)


        return beta, det_prob, x, y, z, h


    dy = getter('dy')
    dz = getter('dz')
    dh = getter('dh')

    def y0(self, **args):
        return self.y0(**args)


    def y(self, samples, **args):
        '''Get cumulative cases from incident ones'''

        dy = self.dy(samples, **args)

        y0 = np.zeros(dy.shape[0])
        if args.get('forecast'):
            y0 = self.y(samples, forecast=False)[:,-1]

        return y0[:,None] + onp.cumsum(dy, axis=1)


    def z0(self, **args):
        return self.z0(**args)


    def z(self, samples, **args):
        '''Get cumulative deaths from incident ones'''

        dz = self.dz(samples, **args)

        z0 = np.zeros(dz.shape[0])
        if args.get('forecast'):
            z0 = self.z(samples, forecast=False)[:,-1]

        return z0[:,None] + onp.cumsum(dz, axis=1)

    def h0(self, **args):
        return self.h0(**args)

    def h(self, samples, **args):
        '''Get cumulative hospitalizations from incident ones'''

        dh = self.dh(samples, **args)

        h0 = np.zeros(dh.shape[0])
        if args.get('forecast'):
            h0 = self.h(samples, forecast=False)[:,-1]

        return h0[:,None] + onp.cumsum(dh, axis=1)