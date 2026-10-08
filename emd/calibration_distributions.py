"""Explicit calibration densities and conditional sampling in backward-age units.

Bounds are absolute ages, applied after the additive offset. The skew-t is
the Azzalini distribution used by MCMCtree, not SciPy's Jones-Faddy skew-t.
"""
import math

PARAMETERS = {
    'exponential': {'scale'},
    'uniform': {'scale'},
    'lognormal': {'meanlog', 'sdlog'},
    'normal': {'mean', 'sd'},
    'gamma': {'shape', 'scale'},
    'skew-t': {'location', 'scale', 'shape', 'df'},
}


def specification(family, parameters):
    """Validate and normalize a JSON-serializable distribution specification."""
    family = family.lower()
    if family not in PARAMETERS:
        raise ValueError(f'unknown calibration distribution {family!r}; choose from {", ".join(PARAMETERS)}')
    allowed = PARAMETERS[family] | {'offset', 'lower', 'upper'}
    unknown = parameters.keys() - allowed
    if unknown:
        raise ValueError(f'{family}: unknown parameters: {", ".join(sorted(unknown))}')
    missing = PARAMETERS[family] - parameters.keys()
    if missing:
        raise ValueError(f'{family}: missing parameters: {", ".join(sorted(missing))}')
    values = {}
    for key, value in parameters.items():
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ValueError(f'{family}: {key} must be numeric') from None
        if not math.isfinite(value) and not (key == 'upper' and value == math.inf):
            raise ValueError(f'{family}: {key} must be finite (only upper may be inf)')
        values[key] = value
    values.setdefault('offset', 0.)
    values.setdefault('lower', 0.)
    values.setdefault('upper', math.inf)
    for key in ('scale', 'sd', 'sdlog', 'df'):
        if key in values and values[key] <= 0:
            raise ValueError(f'{family}: {key} must be positive')
    if family == 'gamma' and values['shape'] <= 0:
        raise ValueError('gamma: shape must be positive')
    if values['lower'] < 0 or values['upper'] <= values['lower']:
        raise ValueError(f'{family}: bounds require 0 <= lower < upper')
    # Uniform is offset + U(0, scale); bounds always refer to absolute ages.
    if family == 'uniform':
        values['upper'] = min(values['upper'], values['offset']+values['scale'])
        if not math.isfinite(values['upper']):
            raise ValueError('uniform: shifted upper endpoint must be finite')
    if family in ('exponential', 'gamma', 'lognormal', 'uniform'):
        values['lower'] = max(values['lower'], values['offset'])
    if values['lower'] >= values['upper']:
        raise ValueError(f'{family}: truncation has empty support')
    if family == 'lognormal':
        try:
            scale = math.exp(values['meanlog'])
        except OverflowError:
            scale = math.inf
        if not math.isfinite(scale) or scale == 0:
            raise ValueError('lognormal: exp(meanlog) is outside floating-point range')
    return dict(family=family, **values)


def parse_distribution(tokens):
    if not tokens:
        raise ValueError('missing distribution family')
    parameters = {}
    for token in tokens[1:]:
        key, sep, value = token.partition('=')
        if not sep or not value or key in parameters:
            raise ValueError(f'invalid or duplicate distribution parameter {token!r}; use key=value')
        parameters[key] = value
    return specification(tokens[0], parameters)


class CalibrationDensity:
    def __init__(self, spec):
        from scipy import stats
        self.spec = spec
        self.family = spec['family']
        self.lower, self.upper = spec['lower'], spec['upper']
        p = spec
        offset = p['offset']
        if self.family == 'exponential':
            self.base = stats.expon(loc=offset, scale=p['scale'])
        elif self.family == 'uniform':
            self.base = stats.uniform(loc=self.lower, scale=self.upper-self.lower)
        elif self.family == 'lognormal':
            self.base = stats.lognorm(s=p['sdlog'], loc=offset, scale=math.exp(p['meanlog']))
        elif self.family == 'normal':
            self.base = stats.norm(loc=offset+p['mean'], scale=p['sd'])
        elif self.family == 'gamma':
            self.base = stats.gamma(a=p['shape'], loc=offset, scale=p['scale'])
        else:
            self.base = stats.t(df=p['df'], loc=offset+p['location'], scale=p['scale'])

    def _inverse_draw(self, rng, lower):
        """Use survival probabilities in the right tail to avoid CDF cancellation."""
        import numpy as np
        from scipy import stats
        upper = self.upper
        u = rng.random(len(lower))
        # Keep inverse transforms off infinite support endpoints.
        u = np.clip(u, np.nextafter(0., 1.), np.nextafter(1., 0.))
        if self.family == 'normal':
            p = self.spec
            loc, scale = p['offset']+p['mean'], p['sd']
            return stats.truncnorm.ppf(u, (lower-loc)/scale, (upper-loc)/scale,
                                       loc=loc, scale=scale)
        if self.family == 'exponential':
            # Memorylessness also handles truncations far beyond CDF precision.
            scale = self.spec['scale']
            mass = -np.expm1(-(upper-lower)/scale)
            return lower - scale*np.log1p(-u*mass)
        if self.family == 'uniform':
            return lower + u*(upper-lower)
        cdf_lo, cdf_hi = self.base.cdf(lower), self.base.cdf(upper)
        sf_lo, sf_hi = self.base.sf(lower), self.base.sf(upper)
        right = cdf_lo > .5
        mass = np.where(right, sf_lo-sf_hi, cdf_hi-cdf_lo)
        if np.any(~np.isfinite(mass) | (mass <= 0)):
            raise ValueError(f'{self.family}: truncation probability is below numerical precision; rescale ages or relax bounds')
        result = np.empty(len(lower))
        result[right] = self.base.isf(sf_hi + (1-u[right])*mass[right])
        result[~right] = self.base.ppf(cdf_lo[~right] + u[~right]*mass[~right])
        return result

    def draw(self, rng, lower):
        """Draw conditional on per-row lower bounds; empty intervals yield NaN."""
        import numpy as np
        from scipy import stats
        lower = np.maximum(np.asarray(lower, dtype=float), self.lower)
        result = np.full(lower.shape, np.nan)
        remaining = np.flatnonzero(np.isfinite(lower) & (lower < self.upper))
        if self.family != 'skew-t' or self.spec['shape'] == 0:
            result[remaining] = self._inverse_draw(rng, lower[remaining])
        else:
            # Rejection from the truncated symmetric t gives the exact truncated
            # Azzalini density. Normalize the acceptance envelope to its maximum
            # on the interval, crucial when sampling the low-density skew tail.
            p = self.spec
            def log_weight(x):
                z = (x-p['offset']-p['location'])/p['scale']
                ratio = z/np.hypot(np.sqrt(p['df']), z)
                ratio = np.where(np.isposinf(z), 1., np.where(np.isneginf(z), -1., ratio))
                return stats.t.logcdf(p['shape']*np.sqrt(p['df']+1)*ratio, p['df']+1)
            endpoint = np.full(len(lower), self.upper) if p['shape'] > 0 else lower
            with np.errstate(invalid='ignore'):
                envelope = log_weight(endpoint)
            if np.any(~np.isfinite(envelope[remaining])):
                raise ValueError('skew-t: truncation probability is below numerical precision')
            for _ in range(10000):
                if not len(remaining):
                    break
                proposed = self._inverse_draw(rng, lower[remaining])
                accept = np.log(np.maximum(rng.random(len(remaining)), np.nextafter(0., 1.))) <= log_weight(proposed)-envelope[remaining]
                result[remaining[accept]] = proposed[accept]
                remaining = remaining[~accept]
            if len(remaining):
                raise ValueError('skew-t: conditional sampler exceeded 10000 proposals per age; revise parameters/bounds')
        active = np.isfinite(lower) & (lower < self.upper)
        if np.any(active & (~np.isfinite(result) | (result < lower) | (result > self.upper))):
            raise ValueError(f'{self.family}: inverse sampling exceeded numerical precision; rescale ages or relax bounds')
        return result
