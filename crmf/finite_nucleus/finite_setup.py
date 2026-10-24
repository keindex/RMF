r"""Spherical oscillator basis and interactions for the Python CRMF solver.

This module translates the setup routines of ``dirhbs_CRMF.f``. Shared
arrays retain their Fortran indices, with an unused index zero wherever
the source lower bound is one. Masses stored on the state are in fm^-1;
potentials are in MeV. No Fortran source or executable is used at runtime.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import linalg, special

from .numerics import (
    component_orbital,
    large_component_orbital,
    small_component_orbital,
)


def single_precision_real(value):
    r"""Preserve fixed-form single precision for source literals.

    Numeric values are rounded as $x\mapsto\operatorname{float}_{32}(x)$,
    while Fortran ``d`` exponents are normalized to ``e`` before parsing.
    """
    if isinstance(value, str) and 'd' in value.lower():
        return float(value.lower().replace('d', 'e'))
    return float(np.float32(value))


# Columns: sigma/omega/delta masses (MeV), scalar/vector/delta/rho
# couplings, b_sigma, c_sigma, c_omega, Lambda_v, c_delta-sigma.
# Unsuffixed literals in the source were default REAL, not REAL*8.
_NONLINEAR_FORCES = {
    'FSU': ('491.5d0','782.5d0','0d0','10.59243d0','14.3020','0d0','5.88367d0',
            '.0007563','.00396','.01','.24','0'),
    'new1': ('491.5d0','782.5d0','980d0','10.2143d0','13.4353','5.5d0','7.33728d0',
             '.00159524','.000540269','.00528340','.0892780','.053'),
    'new2': ('491.5d0','782.5d0','980d0','10.2143d0','13.4353','6.2d0','7.60162d0',
             '.00159524','.000540269','.00528340','.0669420','.044'),
    'new3': ('491.5d0','782.5d0','980d0','10.2143d0','13.4353','6.4d0','7.65801d0',
             '.00159524','.000540269','.00528340','.0571612','.047'),
    'new4': ('491.5d0','782.5d0','980d0','10.2143d0','13.4353','6.2d0','7.57249d0',
             '.00159524','.000540269','.00528340','.0620518','.050'),
    'new5': ('491.5d0','782.5d0','980d0','10.2143d0','13.4353','6.2d0','7.56755d0',
             '.00159524','.000540269','.00528340','.0612239','.051'),
    'FSU-0': ('491.5d0','782.5d0','0d0','10.59243d0','14.3020','0d0','4.4794475',
              '.0007563','.00396','.01','0','0'),
    'FSU-2': ('491.5d0','782.5d0','0d0','10.59243d0','14.3020','0d0','5.2799763',
              '.0007563','.00396','.01','.16','0'),
    'FSU-J0': ('491.5d0','782.5d0','980d0','10.2143d0','13.4353d0','0d0','5.31345d0',
               '.00159524','.000540269','.0052834','.171042','0'),
    'NLrho': ('508.194','782.501','0d0','8.277d0','9.232','0d0','3.769',
              '.00693','-.0048','0','0','0'),
    'NLrho-d': ('508.194','782.501','763.','8.277d0','9.232','6.114','6.863d0',
                '.00693','-.0048','0','0','0'),
    'Lc56dlt': ('491.5','782.500','980.','8.20277d0','9.39816','7.8253','7.91316',
               '.00739801','-.00619976','0','.0744192','0'),
    'NL3': ('508.194','782.501','0d0','10.217d0','12.868','0d0','4.474d0',
            '.002055','-.00265','0','0','0'),
    'NL3-1': ('508.194','782.501','0d0','10.217d0','12.868','0d0','4.767d0',
              '.002055','-.00265','0','.08','0'),
    'FSU-dlt67': ('491.5','782.50','980d0','10.2143d0','13.4353','6.7','7.26866',
                  '.00159524','.000540269','.0052830','.0214','.0385'),
    'FSU-dlt62': ('491.5','782.50','980d0','10.2143d0','13.4353','6.2','6.98518d0',
                  '.00159524','.000540269','.0052830','.0200112','.054'),
}
FORCE_NAMES = tuple(_NONLINEAR_FORCES) + ('DD-ME2', 'DD-PC1', 'EOS')


def initialize_numeric_tables(state):
    r"""Initialize signs, roots, factorials, and half-integer gamma tables.

    The recurrence tables satisfy $n!=n(n-1)!$ and
    $\Gamma(n+1/2)=(n-1/2)\Gamma(n-1/2)$, with inverse tables stored too.
    """
    state.zero, state.one, state.two, state.half, state.third, state.pi = 0., 1., 2., .5, 1./3., math.pi
    table_names = ('sq','sqi','sqh','shi','fak','fad','fi','fdi','wf','wfi',
             'wfd','gm2','gmi','wg','wgi')
    for table_name in table_names:
        setattr(state, table_name, np.ones(state.igfv+1, dtype=np.float64))
    state.iv = np.ones(state.igfv+1, dtype=np.int64)
    state.sq[0], state.sqi[0] = 0., 1.e30
    state.sqh[0] = math.sqrt(.5)
    state.shi[0] = 1./state.sqh[0]
    state.gm2[0] = math.sqrt(math.pi)
    state.gmi[0] = 1./state.gm2[0]
    state.wg[0] = math.sqrt(state.gm2[0])
    state.wgi[0] = 1./state.wg[0]
    for index in range(1, state.igfv+1):
        state.iv[index] = -state.iv[index-1]
        state.sq[index] = math.sqrt(index)
        state.sqi[index] = 1./state.sq[index]
        state.sqh[index] = math.sqrt(index+.5)
        state.shi[index] = 1./state.sqh[index]
        state.fak[index] = index*state.fak[index-1]
        state.fad[index] = (2*index+1)*state.fad[index-1]
        state.fi[index] = 1./state.fak[index]
        state.fdi[index] = 1./state.fad[index]
        state.wf[index] = state.sq[index]*state.wf[index-1]
        state.wfi[index] = 1./state.wf[index]
        state.wfd[index] = math.sqrt(state.fad[index])
        state.gm2[index] = (index-.5)*state.gm2[index-1]
        state.gmi[index] = 1./state.gm2[index]
        state.wg[index] = state.sqh[index-1]*state.wg[index-1]
        state.wgi[index] = 1./state.wg[index]


def initialize_default_state(state):
    """Populate source-compatible defaults after allocating the state arrays."""
    initialize_numeric_tables(state)
    state.tp, state.tis, state.tit = ['', '+', '-'], ['', 'n', 'p'], ['', 'Neutron:', 'Proton: ']
    state.tl = list('spdfghijklmnoPqrStuvwxyz0000000')
    state.hbc, state.r0, state.alphi = 197.328284, 1.2, single_precision_real(137.03602)
    state.itx, state.icm, state.icou = 2, 0, 1
    state.spk0.fill(0.)
    state.spk.fill(0.)
    state.del_.fill(0.)
    state.ala[1:3] = -7.
    state.maxi, state.si, state.siold, state.epsi = 1000, 1., 1., 1.e-6
    state.iaut, state.inxt, state.xmix, state.xmax = 1, 3, .5, .7
    state.xmix0, state.ibroyd, state.ii = state.xmix, 1, 0
    state.inin, state.inink, state.b0 = 1, 1, -2.320
    state.lmes.fill(0)
    state.ggmes.fill(0.)
    state.ff.fill(0.)
    state.ff[:,1:5,1] = 1.


def _sync_force_arrays(state):
    """Copy scalar interaction parameters into the packed channel arrays."""
    state.ames[1:5] = state.amsig, state.amome, state.amdel, state.amrho
    state.ggmes[1:5] = state.ggsig, state.ggome, state.ggdel, state.ggrho
    state.ddmes[1:5] = state.ddsig, state.ddome, state.dddel, state.ddrho
    for coupling_prefix in ('a', 'b', 'c', 'd'):
        getattr(state, coupling_prefix+'_m')[1:5] = [getattr(state, coupling_prefix+'_'+channel_suffix)
                                         for channel_suffix in ('s','v','ts','tv')]


def configure_interaction(state, name=None, eos_parameters=None):
    r"""设置有限核相互作用参数并计算对应的场耦合。

    Meson-exchange couplings are mapped to field strengths proportional to
    $\pm(g_a/m_a)^2$; the point-coupling force instead uses its density
    functions and derivative terms.

    Args:
        state: Initialized mutable solver state.
        name (str, optional): Interaction key; defaults to ``state.parname``.
        eos_parameters (mapping or object, optional): Uniform-matter
            parameters for the ``EOS`` interaction; masses are GeV.

    Returns:
        None: Populates the interaction and unit-converted arrays on ``state``.

    EOS accepts the nuclear-matter parameter mapping (or namespace) with
    ``xms,xmw,xmd,xmr`` in GeV and ``g_s,g_w,g_d,g_r,b_s,c_s,c_w,xl_v,c_ds``.
    """
    name = (name if name is not None else state.parname).strip()
    if name not in FORCE_NAMES:
        raise ValueError(f'Unknown interaction {name!r}; choose from {FORCE_NAMES}')
    state.parname = name
    for key in ('amsig','amome','amdel','amrho','gsig','gome','gdel','grho',
                'ggsig','ggome','ggdel','ggrho','ddsig','ddome','dddel','ddrho',
                'z_bs','z_cs','z_cw','z_lv','z_ds'):
        setattr(state, key, 0.)
    for coupling_prefix in ('a','b','c','d'):
        for channel_suffix in ('s','v','ts','tv'):
            setattr(state, coupling_prefix+'_'+channel_suffix, 0.)
    state.lmes.fill(0)
    state.amu, state.ipc, state.icm, state.idd, state.icou = 939., 0, 2, 2, 1
    state.a_s, state.a_v, state.dsat = 1., 1., 1.
    if name in _NONLINEAR_FORCES:
        force_values = map(single_precision_real, _NONLINEAR_FORCES[name])
        (state.amsig,state.amome,state.amdel,state.gsig,state.gome,state.gdel,state.grho,
         state.z_bs,state.z_cs,state.z_cw,state.z_lv,state.z_ds) = force_values
        state.amrho = 763.
        state.z_bs *= state.amu/state.hbc
        if name in ('FSU', 'new1','new2','new3','new4','new5'):
            state.icm = 1
        if name in ('NL3','NL3-1'):
            state.icm = 0
        if name in ('new1','FSU-J0','FSU-dlt67','FSU-dlt62'):
            state.icou = 2
    elif name == 'DD-ME2':
        state.amsig, state.amome, state.amrho = 550.12380, 783., 763.
        state.gsig, state.gome, state.grho = 10.5396, 13.0189, 3.6836
        state.b_s, state.c_s, state.c_v, state.a_tv, state.dsat = 1.0943, 1.7057, 1.4620, .5647, .152
        state.d_s = 1./math.sqrt(3*state.c_s)
        state.a_s = (1+state.c_s*(1+state.d_s)**2)/(1+state.b_s*(1+state.d_s)**2)
        state.d_v = 1./math.sqrt(3*state.c_v)
        scalar_factor = 2*state.a_s*(state.b_s-state.c_s)*(1-3*state.c_s*(1+state.d_s)**2)/(1+state.c_s*(1+state.d_s)**2)**3
        vector_factor = (1-3*state.c_v*(1+state.d_v)**2)/(1+state.c_v*(1+state.d_v)**2)**3
        ratio = scalar_factor/(2*vector_factor)
        state.b_v = (ratio+state.c_v*(1+state.c_v*(1+state.d_v)**2))/(1+state.c_v*(1+state.d_v)**2-ratio*(1+state.d_v)**2)
        state.a_v = (1+state.c_v*(1+state.d_v)**2)/(1+state.b_v*(1+state.d_v)**2)
    elif name == 'DD-PC1':
        state.dsat = .152
        state.a_s,state.b_s,state.c_s,state.d_s = -10.0462,-9.1504,-6.4273,1.3724
        state.a_v,state.b_v,state.c_v,state.d_v = 5.9195,8.8637,0.,.6584
        state.a_tv,state.b_tv,state.c_tv,state.d_tv = 0.,1.8360,0.,.6403
        state.ddsig = -.8149
        state.ggsig = state.ggome = state.ggdel = state.ggrho = 1.
        state.ipc = 1
    else:
        if eos_parameters is None:
            raise ValueError('The EOS interaction requires eos_parameters')
        eos_mapping = eos_parameters if isinstance(eos_parameters, dict) else vars(eos_parameters)
        required = ('xms','xmw','xmd','xmr','g_s','g_w','g_d','g_r',
                    'b_s','c_s','c_w','xl_v','c_ds')
        missing = [key for key in required if key not in eos_mapping]
        if missing:
            raise ValueError('Missing EOS parameters: '+', '.join(missing))
        state.amsig,state.amome,state.amdel,state.amrho = (1000.*eos_mapping[key] for key in required[:4])
        state.gsig,state.gome,state.gdel,state.grho = (eos_mapping[key] for key in required[4:8])
        state.z_bs = eos_mapping['b_s']*state.amu/state.hbc
        state.z_cs,state.z_cw,state.z_lv,state.z_ds = (eos_mapping[key] for key in ('c_s','c_w','xl_v','c_ds'))
        state.icou = 2
    for key in ('amu','amsig','amome','amdel','amrho'):
        setattr(state, key, getattr(state, key)/state.hbc)
    if state.ipc == 0:
        state.ggsig = -(state.gsig/(state.amsig+1.e-10))**2
        state.ggome = (state.gome/(state.amome+1.e-10))**2
        state.ggdel = -(state.gdel/(state.amdel+1.e-10))**2
        state.ggrho = (state.grho/(state.amrho+1.e-10))**2
        state.lmes[1:5] = np.abs([state.gsig,state.gome,state.gdel,state.grho]) > 1.e-5
        state.gdel = max(abs(state.gdel), 1.e-6)
    state.gl[1:3], state.gal = -728., .415
    _sync_force_arrays(state)


def build_gaussian_quadrature(state, log_progress=False):
    r"""建立正半轴 Gauss-Hermite 径向积分网格。

    The nodes and weights satisfy
    $\int_{-\infty}^{\infty}e^{-x^2}f(x)dx\approx\sum_iw_if(x_i)$.

    Args:
        state: Initialized state with mesh storage.
        log_progress (bool): Reserved source-compatible verbosity flag.

    Returns:
        None: Stores nodes, weights, and the transformed radial weights.
    """
    mesh_points, mesh_weights = special.roots_hermite(2*state.ngh)
    state.xh[1:] = mesh_points[state.ngh:]
    state.ph[1:] = mesh_weights[state.ngh:]
    state.wh[1:] = state.ph[1:]*np.exp(state.xh[1:]**2)
    state.xh[0] = state.ph[0] = state.wh[0] = 1.e-10


def prepare_oscillator_mesh(state, log_progress=False):
    r"""计算谐振子长度、径向坐标和球坐标积分权重。

    The source uses $\hbar\omega=41A^{-1/3}$ MeV and
    $b_0=\sqrt{\hbar^2/(m\hbar\omega)}$ with its optional center-of-mass
    correction.
    """
    state.hb0 = state.hbc/(2*state.amu)
    state.hom = 41.*state.amas**(-state.third)
    if state.icm == 1:
        state.hb0 *= 1.-1./state.amas
    state.b0 = math.sqrt(2*state.hb0/state.hom)
    state.rb[:] = state.xh*state.b0
    state.wdcor[:] = state.b0**3*4*math.pi*state.xh**2*state.wh
    if state.itx == 1:
        state.icou = 0


def initialize_starting_fields(state, log_progress=False):
    r"""Initialize Woods-Saxon scalar/vector fields and Coulomb potential.

    The radial profile is $f(r)=[1+\exp((r-R)/a)]^{-1}$; inside a
    uniformly charged sphere, $V_C(r)\propto(3/R-r^2/R^3)$, and outside
    $V_C(r)\propto1/r$.
    """
    if state.inin == 0:
        return
    potential_depth, asymmetry_factor = single_precision_real(-71.28), single_precision_real(.4616)
    vector_radius, vector_diffuseness, spin_orbit_strength, scalar_radius, scalar_diffuseness = [
        np.asarray(parameter_values, dtype=np.float32).astype(float)
        for parameter_values in ((1.2334, 1.2496), (.615, .6124),
                                 (11.1175, 8.9698), (1.1443, 1.1401),
                                 (.6476, .6469))
    ]
    radius_scale = state.amas**state.third
    for species_index in range(1, state.itx+1):
        parameter_index = species_index-1
        vector_potential = potential_depth*(1.-asymmetry_factor*(state.npr[species_index]-state.npr[3-species_index])/state.amas)
        vector_argument = (state.rb-vector_radius[parameter_index]*radius_scale)/vector_diffuseness[parameter_index]
        scalar_argument = (state.rb-scalar_radius[parameter_index]*radius_scale)/scalar_diffuseness[parameter_index]
        state.vps[:,species_index] = np.where(vector_argument <= 65., vector_potential*special.expit(-vector_argument), 0.)
        state.vms[:,species_index] = np.where(scalar_argument <= 65., -vector_potential*spin_orbit_strength[parameter_index]*special.expit(-scalar_argument), 0.)
    if state.itx == 1:
        state.vps[:,2] = state.vps[:,1]
        state.vms[:,2] = state.vms[:,1]
    state.cou.fill(0.)
    if state.icou:
        charge_radius = vector_radius[1]*radius_scale
        state.cou[:] = np.where(state.rb < charge_radius, .5*(3./charge_radius-state.rb**2/charge_radius**3), 1./state.rb)*state.npr[2]/state.alphi
        state.vps[:,2] += state.cou*state.hbc
        state.vms[:,2] += state.cou*state.hbc


def build_oscillator_basis(state, log_progress=False):
    r"""按费米子壳层截断 ``n0f`` 枚举球形 Dirac 基底块。

    The large/small radial oscillator spaces obey the shell condition
    $2n+l\leq N_F$; blocks are labeled by $\kappa$ and carry
    degeneracy $2|\kappa|$.

    Args:
        state: Initialized state with basis dimensions and storage.
        log_progress (bool): Reserved source-compatible verbosity flag.

    Returns:
        None: Populates block, orbital, and pairing-index tables.
    """
    if not 0 <= state.n0f <= state.n0fx:
        raise ValueError(f'n0f must lie between 0 and {state.n0fx}')
    state.nrm, state.nlm, state.no = 1+(state.n0f+1)//2, state.n0f+1, state.n0b//2+1
    if state.nrm > min(state.ndx,state.nrx) or state.nlm > state.nlx or state.no > state.nox:
        raise ValueError('Oscillator basis exceeds allocated dimensions')
    for total_angular_momentum in range(1, state.n0f+2):
        for parity in (1,2):
            block_index = 2*(total_angular_momentum-1)+parity
            if total_angular_momentum%2 == parity-1:
                kappa, large_orbital, small_orbital = total_angular_momentum, total_angular_momentum, total_angular_momentum-1
            else:
                kappa, large_orbital, small_orbital = -total_angular_momentum, total_angular_momentum-1, total_angular_momentum
            state.ijb[block_index],state.ilb[block_index,1],state.ilb[block_index,2],state.ipb[block_index],state.ikb[block_index] = total_angular_momentum,large_orbital,small_orbital,parity,kappa
    block_index = level_index = level_count = 0
    for total_angular_momentum in range(1, state.n0f+2):
        for parity in (1,-1):
            large_orbital = total_angular_momentum-(1-parity*state.iv[total_angular_momentum])//2
            if large_orbital > state.n0f:
                continue
            kappa = total_angular_momentum*(2*(large_orbital-total_angular_momentum)+1)
            small_orbital = 2*total_angular_momentum-large_orbital-1
            large_dimension, small_dimension = (state.n0f-large_orbital)//2+1, (state.n0f+1-small_orbital)//2+1
            block_index += 1
            state.kb[block_index], state.mb[block_index] = kappa, 2*abs(kappa)
            state.tb[block_index] = f'{state.tl[large_orbital]}{2*total_angular_momentum-1:2d}/2'
            state.id[block_index,1],state.id[block_index,2] = large_dimension,small_dimension
            state.ia[block_index,1],state.ia[block_index,2] = level_index,level_index+large_dimension
            for component in (1,2):
                orbital = component_orbital(kappa,component)
                for radial_index in range(1,int(state.id[block_index,component])+1):
                    level_index += 1
                    state.nr[level_index],state.nl[level_index],state.nj[level_index],state.kk[level_index],state.np[level_index] = radial_index,orbital,total_angular_momentum,kappa,component
                    state.tt[level_index] = f'{radial_index:2d}{state.tl[orbital]}{2*total_angular_momentum-1:3d}/2'
            level_count += max(large_dimension,small_dimension)
    state.nb,state.nt = block_index,level_index
    # BASE's local NK is a dimension check; the shared NK(4) belongs to DIRHB.
    if level_count > state.nkx or level_index > state.ntx or block_index > state.nbx:
        raise ValueError('Number of basis states exceeds allocated dimensions')
    pair_count = 0
    for block_index in range(1,state.nb+1):
        state.ipos[block_index] = pair_count
        large_dimension = int(state.id[block_index,1])
        for second_index in range(1,large_dimension+1):
            for first_index in range(second_index,large_dimension+1):
                pair_count += 1
                state.nib[pair_count] = block_index
                state.nni[1,pair_count],state.nni[2,pair_count] = first_index,second_index
    state.mv = pair_count
    if pair_count > state.mvx:
        raise ValueError('Number of pairing matrix elements exceeds allocated dimensions')


def build_oscillator_wavefunctions(state, log_progress=False):
    r"""在径向网格上构造谐振子径向波函数及其导数。

    The radial functions use the Laguerre form
    $R_{nl}(r)\propto r^l e^{-r^2/(2b^2)}L_n^{l+1/2}(r^2/b^2)$.

    Fermion functions include sqrt(wh*xh**2), whereas boson functions do
    not. Derivatives refer to the radial oscillator before that measure.
    """
    radial_coordinate, radial_square = state.xh, state.xh**2
    gaussian_factor = 2*state.wgi[0]*np.exp(-.5*radial_square)
    first_radial_function = gaussian_factor*np.sqrt(state.wh*radial_square)
    for orbital in range(state.nlm+1):
        state.rnl[1,orbital,:] = first_radial_function
        state.rnl1[1,orbital,:] = (orbital-radial_square)*state.rnl[1,orbital,:]/radial_coordinate
        if state.nrm >= 2:
            state.rnl[2,orbital,:] = first_radial_function*(orbital+1.5-radial_square)*state.shi[orbital+1]
            state.rnl1[2,orbital,:] = ((2+orbital-radial_square)*state.rnl[2,orbital,:]-2*state.sqh[orbital+1]*state.rnl[1,orbital,:])/radial_coordinate
        first_radial_function = first_radial_function*radial_coordinate*state.shi[orbital+1]
        for radial_index in range(3,state.nrm+1):
            state.rnl[radial_index,orbital,:] = ((2*radial_index+orbital-2.5-radial_square)*state.rnl[radial_index-1,orbital,:]-
                            state.sq[radial_index-2]*state.sqh[radial_index-2+orbital]*state.rnl[radial_index-2,orbital,:])*state.sqi[radial_index-1]*state.shi[radial_index-1+orbital]
            state.rnl1[radial_index,orbital,:] = ((2*radial_index+orbital-2-radial_square)*state.rnl[radial_index,orbital,:]-
                             2*state.sq[radial_index-1]*state.sqh[radial_index-1+orbital]*state.rnl[radial_index-1,orbital,:])/radial_coordinate
    state.rnb[1,:] = gaussian_factor
    if state.no >= 2:
        state.rnb[2,:] = gaussian_factor*(1.5-radial_square)*state.shi[1]
    for radial_index in range(3,state.no+1):
        state.rnb[radial_index,:] = ((2*radial_index-2.5-radial_square)*state.rnb[radial_index-1,:]-
                      state.sq[radial_index-2]*state.sqh[radial_index-2]*state.rnb[radial_index-2,:])*state.sqi[radial_index-1]*state.shi[radial_index-1]


def build_spin_orbit_matrix(state, large_dimension, small_dimension, block_index, log_progress=False):
    r"""Project the radial Dirac derivative coupling into a basis block.

    The matrix elements are $T_{ba}=\langle R_b^{(S)}|
    -d/dr-(\kappa+1)/r|R_a^{(L)}\rangle$.
    """
    kappa = int(state.kb[block_index])
    large_orbital = large_component_orbital(kappa)
    small_orbital = small_component_orbital(kappa)
    derivative_basis = (
        -state.rnl1[1:large_dimension + 1, large_orbital, 1:]
        + (-kappa - 1)
        * state.rnl[1:large_dimension + 1, large_orbital, 1:]
        / state.xh[1:]
    )
    return state.rnl[1:small_dimension + 1, small_orbital, 1:] @ derivative_basis.T


def initialize_spin_orbit_fields(state, log_progress=False):
    """Fill packed spin-orbit matrices for all basis blocks."""
    for block_index in range(1,state.nb+1):
        large_dimension,small_dimension = (int(value) for value in state.id[block_index,1:3])
        state.sp[1:large_dimension*small_dimension+1,block_index] = build_spin_orbit_matrix(state,large_dimension,small_dimension,block_index).ravel(order='F')


def build_coulomb_green_function(state, log_progress=False):
    r"""Build the radial Coulomb kernel.

    For $r_>=\max(r,r')$ and $r_<=\min(r,r')$,
    $G_C(r,r')=[3r_>+r_<^2/r_>]/(6\alpha)$ times the quadrature weight.
    """
    if state.icou == 0:
        return
    greater_radius = np.maximum(state.xh[:,None],state.xh[None,:])
    lesser_radius = np.minimum(state.xh[:,None],state.xh[None,:])
    state.ggc[:,:] = (3*greater_radius+lesser_radius**2/greater_radius)*state.wdcor[None,:]/(6*state.alphi)


def build_meson_green_functions(state, log_progress=False):
    r"""构造四种 Klein-Gordon 介子的径向 Green 矩阵。

    Each matrix represents the inverse of $-\nabla^2+m_a^2$ in the
    truncated boson oscillator basis, projected back to the radial mesh.

    Args:
        state: Initialized state with basis and meson masses.
        log_progress (bool): Reserved source-compatible verbosity flag.

    Returns:
        None: Stores the Green matrices in ``state.gg``; point coupling has
        no finite-range meson propagator and returns early.
    """
    if state.ipc == 1:
        return
    radial_basis = state.rnb[1:state.no+1,1:]
    radial_indices = np.arange(1,state.no+1)
    for meson_index in range(1,5):
        scale_factor = (1./((state.ames[meson_index]+1.e-10)*state.b0))**2
        matrix = np.diag(1.+scale_factor*(2*radial_indices-.5))
        off_diagonal = scale_factor*state.sq[1:state.no]*state.sqh[1:state.no]
        matrix += np.diag(off_diagonal,1)+np.diag(off_diagonal,-1)
        solved_basis = linalg.solve(matrix,radial_basis,assume_a='pos',check_finite=False)
        state.gg[1:,1:,meson_index] = (radial_basis.T @ solved_basis)*(state.wdcor[None,1:]/(4*math.pi*state.b0**3))


# Compatibility aliases for the original Fortran routine names.
gfv = initialize_numeric_tables
default = initialize_default_state
forces = configure_interaction
gaush = build_gaussian_quadrature
prep = prepare_oscillator_mesh
start = initialize_starting_fields
lfkap = large_component_orbital
lgkap = small_component_orbital
lfgkap = component_orbital
base = build_oscillator_basis
gaupol = build_oscillator_wavefunctions
sigp = build_spin_orbit_matrix
singf = initialize_spin_orbit_fields
greecou = build_coulomb_green_function
greemes = build_meson_green_functions
