r"""Input data and explicit storage replacing the Fortran COMMON blocks.

Array indices follow the original source: index zero is padding except for
radial grids, factorial tables and other arrays explicitly starting at zero.
Matrices packed into a vector retain Fortran's active-block leading dimension.
"""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from numbers import Integral
import math
import numpy as np


ELEMENTS = ('n H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn '
            'Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd '
            'In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu '
            'Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu '
            'Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og').split()


@dataclass(frozen=True)
class NucleusConfig:
    r"""球形有限核计算的输入配置与合法性检查。

    The neutron and proton counts obey $N=A-Z$ and
    $Z=\mathrm{ELEMENTS.index(symbol)}$. Radii use fm, energies/gaps use
    MeV, and the basis cutoffs are oscillator-shell counts.
    """
    symbol: str = 'Ca'
    mass_number: int = 48
    n0f: int = 20
    n0b: int = 30
    inin: int = 1
    inink: int = 1
    gap_neutron: float = 1.0
    gap_proton: float = 1.0
    force: str = 'FSU-dlt67'
    max_iterations: int = 1000
    tolerance: float = 1e-6
    constraint: float = 0.0
    center_of_mass: int | None = None
    coulomb: int | None = None

    def __post_init__(self):
        """Validate integer, physical, and solver-control constraints."""
        for name in ('mass_number','n0f','n0b','inin','inink','max_iterations'):
            value = getattr(self,name)
            if not isinstance(value,Integral) or isinstance(value,bool):
                raise ValueError(f'{name} must be an integer.')
        for name in ('gap_neutron','gap_proton','tolerance','constraint'):
            if not math.isfinite(getattr(self,name)):
                raise ValueError(f'{name} must be finite.')
        if self.gap_neutron < 0 or self.gap_proton < 0:
            raise ValueError('Initial pairing gaps must be nonnegative.')
        if self.symbol not in ELEMENTS:
            raise ValueError(f'Unknown element symbol: {self.symbol}')
        if self.mass_number <= 0 or self.mass_number < self.proton_number:
            raise ValueError('Mass number must be positive and at least the proton number.')
        if not 1 <= self.n0f <= 20 or not 0 <= self.n0b <= 36:
            raise ValueError('Source basis limits: 1 <= n0f <= 20, 0 <= n0b <= 36.')
        if self.inin not in (0, 1) or self.inink not in (0, 1):
            raise ValueError('Initialization flags must be 0 (restart) or 1 (new).')
        if self.max_iterations < 1 or self.tolerance <= 0:
            raise ValueError('Iteration limit and tolerance must be positive.')
        if self.center_of_mass not in (None, 0, 1, 2) or self.coulomb not in (None, 0, 1, 2):
            raise ValueError('Invalid center-of-mass or Coulomb option.')

    @property
    def proton_number(self):
        # Formula: Z = index(symbol) in the one-based ELEMENTS table.
        r"""Return :math:`Z`, the one-based position of ``symbol`` in ``ELEMENTS``."""
        return ELEMENTS.index(self.symbol)

    @classmethod
    def from_file(cls, path):
        r"""Read a labelled ``dirhb.dat`` file and validate its configuration.

        The parsed nucleus obeys :math:`N=A-Z`, while the two shell and gap
        values are taken from the labelled input records. Comments beginning
        with ``!`` are ignored, so spacing and unrelated comment lines do not
        affect the parser.

        Args:
            path (path-like): Input file location.

        Returns:
            NucleusConfig: Parsed and validated configuration.
        """
        input_path = Path(path)
        lines = input_path.read_text(encoding='utf-8-sig').splitlines()

        def uncommented(line):
            """Remove the Fortran ``!`` comment suffix from one input line."""
            return line.split('!', 1)[0].strip()

        def field_tokens(label):
            """Return whitespace-separated values following a labelled field."""
            for line in lines:
                content = uncommented(line)
                if content.lower().startswith(label.lower()):
                    return content.split('=', 1)[-1].replace(',', ' ').split()
            raise ValueError(f'missing {label} record')

        try:
            shells = field_tokens('n0f')
            initialization = field_tokens('inin')
            gaps = field_tokens('init.gap')
            force_tokens = field_tokens('force')
            nucleus = None
            for line in lines:
                candidate = uncommented(line).split()
                if len(candidate) >= 2 and candidate[0] in ELEMENTS:
                    try:
                        int(candidate[1])
                    except ValueError:
                        continue
                    nucleus = candidate
                    break
            if nucleus is None:
                raise ValueError('missing nucleus record')
            return cls(symbol=nucleus[0], mass_number=int(nucleus[1]),
                       n0f=int(shells[0]), n0b=int(shells[1]),
                       inin=int(initialization[0]), inink=int(initialization[1]),
                       gap_neutron=float(gaps[0]), gap_proton=float(gaps[1]),
                       force=force_tokens[0])
        except (IndexError, ValueError) as exc:
            raise ValueError(f'Invalid dirhb.dat input: {input_path}: {exc}') from exc


def new_state(config=None):
    r"""分配一次计算所需的默认状态、数组和工作空间。

    Arrays with source-style one-based indexing reserve index zero as padding;
    radial integration arrays retain physical index zero where defined.

    Args:
        config (NucleusConfig, optional): Configuration copied into the state.

    Returns:
        types.SimpleNamespace: Mutable solver state and allocated work arrays.
    """
    config = config or NucleusConfig()
    state = SimpleNamespace(config=config, log=[], converged=False, progress=None)
    dimensions = dict(igfv=100, ngh=60, n0fx=20, n0bx=36, jdim=3311)
    dimensions.update(ngh2=120, nnmax=20, nbx=42, ntx=482, nkx=251,
                      nrx=11, nlx=21, ndx=11, ngx=11, nfx=11, nox=19,
                      mvx=946, mvxg=1078, nnn1=253, nnn2=3500, nnn3=40,
                      mvtx=3793, mv2x=946**2, nb2x=84, nhx=22, nhbx=44,
                      nhbqx=44**2, nfgx=121, nhhx=484, noox=361, lx=40)
    for name, value in dimensions.items():
        setattr(state, name, value)
    scalar_names = ('nb nt no mv nrm nlm nrbm hom hb0 b0 si siold epsi xmix xmix0 '
                    'xmax maxi ii inxt iaut itx icm icou ipc inl idd hbc hqc alphi r0 '
                    'l6 lin lou lwin lwou lplo laka lvpp ibroyd amu amsig amome amdel '
                    'amrho gal ggsig ggome ggdel ggrho ddsig ddome dddel ddrho '
                    'gsig gome gdel grho z_bs z_cs z_cw z_lv z_ds dsat '
                    'ea rms etot rc rn binding_energy radius_charge radius_neutron '
                    'radius_proton radius_mass els_2d_p els_3p_n els_2f_n').split()
    for name in scalar_names:
        setattr(state, name, 0.0 if name[0] not in 'ijklmn' else 0)
    arrays = {
        'kb mb ijb ipb ikb ipos': ('nbx',),
        'nrbl ka kd kacan kdcan': ('nbx', 4),
        'ia id ilb': ('nbx', 2),
        'nr nl nj kk np': ('ntx',),
        'nib': ('mvx',), 'nni': (2, 'mvx'),
        'de hh rosh': ('nhhx', 'nb2x'),
        'ala tz del_ spk spk0 gl slp': (2,),
        'eecan decan vvcan mulcan equ': ('nkx', 4),
        'fgcan fg': ('nhx', 'nkx', 4),
        'fguv': ('nhbx', 'nkx', 4),
        'nkcan nk': (4,), 'ibk': ('nkx', 4),
        'rnl rnl1': ('nrx', 'nlx', 'ngh'),
        'xh wh ph rb wdcor cou drvp': ('ngh',),
        'wave dwave': ('ngh2', 'nkx', 2),
        'ames ggmes ddmes lmes a_m b_m c_m d_m': (4,),
        'ecmd ecmn ecm': (3,), 'ggc': ('ngh', 'ngh'),
        'ff': ('ngh', 4, 2), 'aka': ('mvx', 2), 'vn': ('mvx', 'n0fx'),
        'ro dro phi': ('ngh', 4), 'rs rv vps vms': ('ngh', 2),
        'sp': ('nfgx', 'nbx'), 'gg': ('ngh', 'ngh', 4),
        'rnb': ('nox', 'ngh'), 'wig': ('jdim',), 'locs': ('lx', 'lx', 'lx'),
        'iv sq sqi sqh shi fak fad fi fdi wf wfi wfd gm2 gmi wg wgi': ('igfv',)
    }
    integer_arrays = set('kb mb ijb ipb ikb ipos nrbl ka kd kacan kdcan ia id ilb nr nl nj kk np nib nni nkcan nk ibk lmes locs iv mulcan'.split())
    for names, dims in arrays.items():
        if dims is None:
            continue
        shape = tuple(
            (dimensions[dimension] if isinstance(dimension, str) else dimension) + 1
            for dimension in dims
        )
        for name in names.split():
            setattr(state, name, np.zeros(shape, dtype=np.int64 if name in integer_arrays else np.float64))
    state.tb = np.full(state.nbx + 1, '', dtype='U8')
    state.tt = np.full(state.ntx + 1, '', dtype='U12')
    state.tk = np.full((state.nkx + 1, 5), '', dtype='U12')
    state.tp, state.tis, state.tit = ['', '+', '-'], ['', 'n', 'p'], ['', 'Neutron:', 'Proton:']
    state.tl = list('spdfghijklmnoPqrStuvwxyz00000000')
    state.zero, state.one, state.two, state.half, state.third, state.pi = 0., 1., 2., .5, 1./3., np.pi
    apply_config(state, config)
    return state


def apply_config(state, config):
    # Formula: N = A - Z, with Z = proton_number and A = mass_number.
    r"""将经过检查的核素输入复制到可变求解状态中。

    Args:
        state: State returned by :func:`new_state`.
        config (NucleusConfig): Nucleus, basis, restart, and iteration inputs.

    Returns:
        None: Updates ``state`` in place.
    """
    state.n0f, state.n0b = config.n0f, config.n0b
    state.nucnam = config.symbol
    state.nama = state.nmas = config.mass_number
    state.amas = float(config.mass_number)
    state.npro = config.proton_number
    state.nneu = state.nama - state.npro
    state.npr = np.array([0, state.nneu, state.npro], dtype=np.int64)
    state.tz[1:] = state.npr[1:]
    state.inin, state.inink = config.inin, config.inink
    state.del_[1:] = [config.gap_neutron, config.gap_proton]
    state.parname = config.force
    state.maxi, state.epsi = config.max_iterations, config.tolerance
    state.xlambda = config.constraint
    if config.center_of_mass is not None:
        state.icm = config.center_of_mass
    if config.coulomb is not None:
        state.icou = config.coulomb
