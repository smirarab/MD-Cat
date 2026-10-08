"""Read treePL-style distributions and supported external calibration formats.

External files supply calibration densities only, not dating settings or tree
priors. Unsupported calibration semantics fail explicitly.
"""
import math
import re
import xml.etree.ElementTree as ET


def read_calibrations(text, input_format='auto'):
    from emd.calibration import convert
    stripped = text.lstrip('\ufeff \t\r\n')
    if input_format == 'auto':
        input_format = ('beast2' if stripped.startswith('<') else
                        'mcmctree' if stripped.startswith('(') or re.match(r'\d+\s+\d+\s*\n', stripped)
                        else 'treepl')
    if input_format == 'treepl':
        _, metadata = convert(stripped, allow_distributions=True)
    elif input_format == 'beast2':
        metadata = read_beast2(stripped)
    elif input_format == 'mcmctree':
        metadata = read_mcmctree(stripped)
    else:
        raise ValueError(f'unknown calibration format {input_format!r}')
    metadata['input_format'] = input_format
    return metadata


def validate_names(name, taxa):
    if len(taxa) < 2 or len(set(taxa)) != len(taxa):
        raise ValueError(f'{name}: MRCA requires at least two distinct taxa, without duplicates')
    if any(not token or any(c.isspace() or c in '+="\'' for c in token) for token in [name, *taxa]):
        raise ValueError(f'{name}: unsupported whitespace or MD-CAT delimiter in name/taxa')


def read_mcmctree(text):
    """Import quoted ST(location,scale,shape,df) and G(shape,rate) node labels."""
    import treeswift
    from emd.calibration_distributions import specification
    header = re.match(r'\s*(\d+)\s+(\d+)\s*\n', text)
    if header:
        if int(header[2]) != 1:
            raise ValueError('MCMCTree input must contain exactly one tree')
        text = text[header.end():]
    if text.count(';') != 1 or text.split(';', 1)[1].strip():
        raise ValueError('MCMCTree input must contain exactly one semicolon-terminated tree')
    try:
        tree = treeswift.read_tree_newick(text)
    except Exception as exc:
        raise ValueError(f'invalid MCMCTree calibration tree: {exc}') from exc
    tips = [n.label for n in tree.traverse_leaves()]
    if len(tips) != len(set(tips)) or (header and int(header[1]) != len(tips)):
        raise ValueError('MCMCTree tree has duplicate tips or a mismatched taxon count')
    entries = {}
    for node in tree.traverse_postorder():
        if node.is_leaf() or not node.label:
            continue
        label = node.label.strip("'\"")
        match = re.fullmatch(r'(ST|G)\s*\(([^()]*)\)', label)
        if not match:
            raise ValueError(f'unsupported MCMCTree calibration {label!r}; supported: ST(location,scale,shape,df), G(shape,rate). Soft-bound B/L/U calibrations are not hard truncations.')
        try:
            parameters = [float(x.strip()) for x in match[2].split(',')]
        except ValueError:
            raise ValueError(f'invalid MCMCTree parameters in {label!r}') from None
        if match[1] == 'ST' and len(parameters) == 4:
            spec = specification('skew-t', dict(zip(('location', 'scale', 'shape', 'df'), parameters)))
        elif match[1] == 'G' and len(parameters) == 2:
            shape, rate = parameters
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError('MCMCTree gamma rate must be finite and positive')
            spec = specification('gamma', dict(shape=shape, scale=1/rate))
        else:
            raise ValueError(f'wrong number of MCMCTree parameters in {label!r}')
        name = f'calibration_{len(entries)+1}'
        taxa = [n.label for n in node.traverse_leaves()]
        validate_names(name, taxa)
        entries[name] = dict(mrca=taxa, distribution=spec, monophyletic=True)
    if not entries:
        raise ValueError('no supported MCMCTree calibrations found')
    return dict(calibrations=entries)


def read_beast2(text):
    """Import fixed scalar parameters from BEAST 2 MRCAPrior elements.

    Deliberately does not execute XML plugins, templates, or expressions.
    """
    from emd.calibration_distributions import specification
    if '<!DOCTYPE' in text.upper() or '<!ENTITY' in text.upper():
        raise ValueError('BEAST XML DTDs/entities are unsupported')
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f'invalid BEAST XML: {exc}') from exc
    ids = {}
    for el in root.iter():
        if 'id' in el.attrib:
            if el.attrib['id'] in ids:
                raise ValueError(f'duplicate BEAST XML id {el.attrib["id"]!r}')
            ids[el.attrib['id']] = el
    maps = {el.attrib['name']: (el.text or '').strip() for el in root.iter('map') if 'name' in el.attrib}
    state_parameters = {el.get('id') for state in root.iter('state') for el in state.iter() if el.get('id')}

    def kind(el):
        value = el.get('spec', maps.get(el.tag, el.tag))
        return value.rsplit('.', 1)[-1]

    def resolve(el):
        seen = set()
        while 'idref' in el.attrib:
            ref = el.attrib['idref']
            if ref in seen or ref not in ids:
                raise ValueError(f'cyclic or unresolved BEAST reference {ref!r}')
            seen.add(ref)
            el = ids[ref]
        return el

    def get(el, key, default=None):
        matches = [c for c in el if c.get('name', c.tag) == key]
        if len(matches) > 1 or (matches and key in el.attrib):
            raise ValueError(f'duplicate BEAST input {key}')
        value = matches[0] if matches else el.get(key, default)
        if isinstance(value, str) and value.startswith('@'):
            if value[1:] not in ids:
                raise ValueError(f'unresolved BEAST reference {value!r}')
            value = ids[value[1:]]
        return resolve(value) if isinstance(value, ET.Element) else value

    def boolean(value):
        if str(value).lower() not in ('true', 'false', '1', '0'):
            raise ValueError(f'unsupported BEAST boolean {value!r}')
        return str(value).lower() in ('true', '1')

    def number(el, key, default):
        value = get(el, key, default)
        if isinstance(value, ET.Element):
            if kind(value) not in ('RealParameter', 'parameter') or len(value):
                raise ValueError(f'BEAST {key}: requires a scalar constant parameter')
            if boolean(value.get('estimate', 'true' if value.get('id') in state_parameters else 'false')):
                raise ValueError(f'BEAST {key}: estimated hyperparameters cannot be imported as fixed distributions')
            value = value.get('value', value.text or '')
        try:
            result = float(value)
        except (ValueError, TypeError):
            raise ValueError(f'BEAST {key}: requires a numeric scalar') from None
        if not math.isfinite(result):
            raise ValueError(f'BEAST {key}: requires a finite scalar')
        return result

    entries = {}
    tree_refs = set()
    for prior in root.iter():
        if kind(prior) != 'MRCAPrior' or 'idref' in prior.attrib:
            continue
        name = prior.get('id', f'calibration_{len(entries)+1}')
        if name in entries:
            raise ValueError(f'duplicate BEAST calibration name {name!r}')
        if boolean(get(prior, 'tipsonly', 'false')) or boolean(get(prior, 'useOriginate', 'false')):
            raise ValueError(f'{name}: tipsonly/useOriginate priors are unsupported')
        distribution = get(prior, 'distr')
        # A prior may use <LogNormal name="distr"> or <distr spec="...">.
        if not isinstance(distribution, ET.Element):
            raise ValueError(f'{name}: MRCA prior requires a supported distribution')
        taxonset = get(prior, 'taxonset')
        if not isinstance(taxonset, ET.Element):
            raise ValueError(f'{name}: requires an explicit taxonset')
        taxa = []
        for child in taxonset:
            if child.tag != 'taxon' or kind(resolve(child)) not in ('Taxon', 'taxon'):
                raise ValueError(f'{name}: requires an explicit list of taxa (no alignment or template taxonsets)')
            taxa.append(child.get('idref', child.get('id', '')))
        validate_names(name, taxa)
        tree = get(prior, 'tree')
        if isinstance(tree, ET.Element):
            tree_refs.add(tree.get('id', str(id(tree))))
            if get(tree, 'trait') is not None or any(kind(el) == 'TraitSet' or el.tag == 'trait' for el in tree.iter()):
                raise ValueError(f'{name}: dated-tip trees are unsupported; all tips must be present-day')
        dist = kind(distribution)
        offset = number(distribution, 'offset', 0)
        if dist == 'Exponential':
            family, params = 'exponential', dict(scale=number(distribution, 'mean', 1))
        elif dist == 'Uniform':
            lo, hi = number(distribution, 'lower', 0), number(distribution, 'upper', 1)
            family, params = 'uniform', dict(scale=hi-lo)
            offset += lo
        elif dist == 'Normal':
            tau = get(distribution, 'tau')
            if tau is not None and get(distribution, 'sigma') is not None:
                raise ValueError(f'{name}: Normal cannot specify both sigma and tau')
            precision = number(distribution, 'tau', 1)
            if precision <= 0:
                raise ValueError(f'{name}: Normal precision must be positive')
            family, params = 'normal', dict(mean=number(distribution, 'mean', 0),
                                           sd=1/math.sqrt(precision) if tau is not None else number(distribution, 'sigma', 1))
        elif dist == 'LogNormalDistributionModel' or dist == 'LogNormal':
            m, s = number(distribution, 'M', 0), number(distribution, 'S', 1)
            if boolean(get(distribution, 'meanInRealSpace', 'false')):
                if m <= 0:
                    raise ValueError(f'{name}: real-space lognormal mean must be positive')
                m = math.log(m)-s*s/2
            family, params = 'lognormal', dict(meanlog=m, sdlog=s)
        elif dist == 'Gamma':
            alpha, beta = number(distribution, 'alpha', 2), number(distribution, 'beta', 2)
            mode = get(distribution, 'mode', 'ShapeScale')
            if alpha <= 0 or beta <= 0:
                raise ValueError(f'{name}: Gamma alpha/beta must be positive')
            if mode == 'OneParameter':
                scale = 1/alpha
            elif mode == 'ShapeScale':
                scale = beta
            elif mode in ('ShapeRate', 'ShapeMean'):
                # BEAST initializes scale to 2 when beta is omitted, in
                # either mode; conversion applies only to a supplied beta.
                scale = 2 if get(distribution, 'beta') is None else (1/beta if mode == 'ShapeRate' else beta/alpha)
            else:
                raise ValueError(f'{name}: unsupported BEAST Gamma mode {mode!r}')
            family, params = 'gamma', dict(shape=alpha, scale=scale)
        else:
            raise ValueError(f'{name}: unsupported BEAST distribution {dist!r}')
        # Reject unknown distribution inputs instead of silently dropping
        # custom truncation or other plugin semantics.
        allowed = {'Exponential': {'mean'}, 'Uniform': {'lower', 'upper'},
                   'Normal': {'mean', 'sigma', 'tau'},
                   'LogNormalDistributionModel': {'M', 'S', 'meanInRealSpace'},
                   'LogNormal': {'M', 'S', 'meanInRealSpace'},
                   'Gamma': {'alpha', 'beta', 'mode'}}[dist] | {'offset', 'id', 'spec', 'name'}
        unknown = (set(distribution.attrib) | {c.get('name', c.tag) for c in distribution})-allowed
        if unknown:
            raise ValueError(f'{name}: unsupported BEAST distribution inputs: {", ".join(sorted(unknown))}')
        entries[name] = dict(mrca=taxa, distribution=specification(family, dict(params, offset=offset)),
                             monophyletic=boolean(get(prior, 'monophyletic', 'false')))
    if len(tree_refs) > 1:
        raise ValueError('BEAST calibrations refer to multiple trees; supply priors for one tree')
    if not entries:
        raise ValueError('no BEAST 2 MRCAPrior calibration distributions found')
    return dict(calibrations=entries)
