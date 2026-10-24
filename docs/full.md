# Efects of isoscalar- and isovector-scalar meson mixing on neutron star structure

Fan Li<sub>,</sub><sup>1</sup> Bao-Jun Cai<sub>,</sub><sup>2</sup> Ying Zhou<sub>,</sub><sup>2</sup> Wei-Zhou Jiang <sub>,</sub><sup>3</sup> and Lie-Wen Chen <sup>1</sup>

<sup>1</sup>School of Physics and Astronomy, Shanghai Key Laboratory for Particle Physics and Cosmology, and Key Laboratory for Particle

Astrophysics and Cosmology (MOE), Shanghai Jiao Tong University, Shanghai 200240, China

<sup>2</sup>Quantum Machine Learning Laboratory, Shadow Creator Inc., Shanghai 201208, China

<sup>3</sup>School of Physics, Southeast University, Nanjing 211189, China

## ABSTRACT

Based on the accurately calibrated interaction FSUGold, we show that including isovector scalar δ meson and its coupling to isoscalar scalar σ meson in the relativistic mean field (RMF) model can soften the symmetry energy $E _ { \mathrm { s y m } } ( n )$ at intermediate densities while stifen the $E _ { \mathrm { s y m } } ( n )$ at high densities. We find this new RMF model can be simultaneously compatible with (1) the constraints on the equation of state of symmetric nuclear matter at suprasaturation densities from flow data in heavy-ion collisions, (2) the neutron skin thickness of <sup>208</sup>Pb from the PREX-II experiment, (3) the largest mass of neutron star (NS) reported so far from PSR J0740+6620, (4) the limit of $\Lambda _ { 1 . 4 } \leq 5 8 0$ for the dimensionless tidal deformability of the canonical $1 . 4 M _ { \odot }$ NS from the gravitational wave signal GW170817, (5) the mass-radius relation of PSR J0030+0451 and PSR J0740+6620 measured by NICER, and thus remove the tension between PREX-II and GW170817 observed in the conventional RMF model.

## 1. INTRODUCTION

A model-independent extraction of the neutron skin thickness of <sup>208</sup>Pb is reported recently by the updated Lead Radius EXperiment (PREX-II) (Adhikari et al. 2021):

$$
\Delta r _ {\mathrm{np}} \equiv r _ {n} - r _ {p} = 0. 2 8 3 \pm 0. 0 7 1 \mathrm{fm} ^ {- 3},\tag{1}
$$

where $r _ { n } \ \left( r _ { p } \right)$ is the rms radii of the neutron (proton) distribution in the nucleus. This value means a remarkably thick $\Delta r _ { \mathrm { n p } } \mathrm { f o r ^ { 2 0 8 } P b }$ compared to other results, and suggests a very stif nuclear symmetry energy $E _ { \mathrm { s y m } } ( n )$ around nuclear saturation density $n _ { 0 } \sim 0 . 1 6 ~ \mathrm { f m ^ { - 3 } }$ . The $E _ { \mathrm { s y m } } ( n )$ is an essential but poorly-known quantity that encodes the isospin dependence of nuclear matter equation of state (EOS) and plays a key role in nuclear physics and astrophysics (Baran et al. 2005; Steiner et al. 2005; Li et al. 2008; Gandolfi et al. 2015; Ozel & Freire<sup>¨</sup> 2016; Baldo & Burgio 2016; Zhang & Li 2019). Indeed, it has been established that the symmetry energy density slope $L ( n )$ around $n _ { 0 }$ exhibits a strong positive linear correlation with the $\Delta r _ { \mathrm { n p } }$ of finite nuclei (Brown 2000; Furnstahl 2002; Chen et al. 2005; Centelles et al. 2009; Chen et al. 2010; Roca-Maza et al. 2011). In particular, there is an even stronger linear correlation between the $\Delta r _ { \mathrm { n p } }$ and the $L ( n _ { c } )$ at a subsaturation cross density $n _ { c } = 0 . 1 1 / 0 . 1 6 n _ { 0 } \approx 2 / 3 n _ { 0 }$ (roughly corresponding to the average density of finite nuclei), and the $\Delta r _ { \mathrm { n p } }$ of heavy nuclei can be uniquely determined by $L ( n _ { c } )$ (Zhang & Chen 2013). The large $\Delta r _ { \mathrm { n p } }$ of ${ } ^ { 2 0 8 } \mathrm { { P b } }$ from PREX-II requires $L ( n _ { c } ) ~ \mathrm { ~ \stackrel { ~ > ~ } { ~ } ~ 5 2 ~ ~ M e V }$ (Yue et al. 2021) or $L ( 2 / 3 n _ { 0 } ) \gtrsim 4 9$ MeV (Reed et al. 2021), suggesting a quite stif $E _ { \mathrm { s y m } } ( n )$ at subsaturation density.

The stif $E _ { \mathrm { s y m } } ( n )$ usually results in a large value of neutron star (NS) radius and tidal deformability. Using the $\Delta r _ { \mathrm { n p } } = 0 . 2 8 3 \pm 0 . 0 7 1 ~ \mathrm { f m ^ { - 3 } ~ o f ~ ^ { 2 0 8 } } .$ Pb from PREX-II, Reed et al. (Reed et al. 2021) have recently obtained a constraint of $6 4 2 \ \leqslant \ \Lambda _ { 1 . 4 } \ \leqslant \ 9 5 5$ for the dimensionless tidal deformability of $1 . 4 M _ { \odot }$ NS within the nonlinear relativistic mean field (RMF) model. However, a limit of $\Lambda _ { 1 . 4 } \leq 5 8 0$ has been extracted from the gravitational wave signal GW170817 (Abbott et al. 2018), indicating a serious tension between the PREX-II experiment and GW170817 in the RMF model, although this tension has not been observed in the non-relativistic energy density functional with the extended Skyrme interactions including the momentum dependence of effective many-body forces (Yue et al. 2021). Also, we note that the tension between PREX-II and GW170817 can be ameliorated in Bayesian analysis with parameterized EOSs of nuclear matter and NS matter (Biswas 2021). On the other hand, in the terrestrial lab, the EOS of symmetric nuclear matter (SNM) at suprasaturation densities has been constrained by collective flow data in heavy-ion collisions (HIC) (Danielewicz et al. 2002), which strongly restricts the NS maximum mass $M _ { \mathrm { m a x } }$ with $M _ { \mathrm { m a x } } \sim 2 . 3 M _ { \odot }$ (Zhou et al. 2019; Zhou & Chen 2019; Cai & Chen 2017). In addition, the heaviest NS with mass $2 . 0 8 _ { - 0 . 0 7 } ^ { + 0 . 0 7 } M _ { \odot }$ observed so far for PSR J0740+6620 (Cromartie et al. 2020; Fonseca et al. 2021) can put strong constraints on the dense matter EOS and rule out the supersoft high density symmetry energy (Zhou & Chen 2019). Furthermore, the simultaneous mass-radius (M-R) determinations from Neutron Star Interior Composition Explorer (NICER) for PSR J0030+0451 (Riley et al. 2019; Miller et al. 2019) with mass around $1 . 4 M _ { \odot }$ as well as for PSR J0740+6620 (Riley et al. 2021; Miller et al. 2021) with mass around $2 . 0 M _ { \odot }$ have been obtained, which further put stringent constraints on the symmetry energy. With so many constraints/data in the multimessenger era, it is of extreme importance to test the applicability of nuclear energy density functionals, including relativistic and nonrelativistic models.

The RMF models are covariant efective theory and this relativistic covariant formulation has made great success during the last decades in understanding many nuclear phenomena (see, e.g., Refs. (Serot & Walecka 1986, 1997; Meng et al. 2006)). Based on the Walecka model (Walecka 1974) and its extensions (Boguta & Bodmer 1977; Serot 1979; M¨uller & Serot 1996), the RMF models usually involve the Dirac nucleons, the isoscalar scalar meson σ, isovector scalar meson ω and isovector vector meson $\rho .$ In addition, the σ meson selfinteractions were used (Boguta & Bodmer 1977) to reduce the nuclear incompressibility coeficient of symmetric nuclear matter while the $\omega$ meson self-interactions were introduced to eficiently tune the NS maximum mass without sacrificing the agreement with other wellreproduced observables (M¨uller & Serot 1996). Furthermore, the crossing term $\rho { - } \omega$ is highly sensitive to the density dependence of the symmetry energy and is important for the properties of neutron-rich nuclei as well as the structure and cooling dynamics of NSs (Horowitz & Piekarewicz 2001a,b; Carriere et al. 2003).

According to relativistic covariance in the efective theory, the isovector scalar meson δ should also be included in the Lagrangian of the RMF model, and it may play an important role in understanding the properties of isospin asymmetric matter at high densities (Kubis & Kutschera 1997; Hofmann et al. 2001; Liu et al. 2002; Chen et al. 2007). Very recently, the coupling of δ meson with σ meson has been studied by Zabari et al. (Zabari et al. 2019a,b), and it is found that the $\delta { - } \sigma$ mixing can soften the $E _ { \mathrm { s y m } } ( n )$ at intermediate densities but stifen the $E _ { \mathrm { s y m } } ( n )$ at high densities. This interesting feature provides the possibility to decrease the $\Lambda _ { 1 . 4 }$ to be compatible with the constraint of $\Lambda _ { 1 . 4 } \leq 5 8 0$ (Abbott et al. 2018) since the $\Lambda _ { 1 . 4 }$ is essentially controlled by the density behaviors of $E _ { \mathrm { s y m } } ( n )$ at intermediate densities around $2 n _ { 0 }$ . Nevertheless, we note that the parameter sets used in Ref. (Zabari et al. 2019a) violate the constraints on the EOS of SNM at suprasaturation densities from flow data in heavy-ion collisions (Danielewicz et al. 2002), predict a $\Lambda _ { 1 . 4 }$ larger than 580. Also in Ref. (Zabari et al. 2019a), the ω meson self-interactions and the crossing term $\rho { - } \omega$ are not considered. A similar study has been performed very recently in Ref. (Miyatsu et al. 2022).

In this work, based on the accurately calibrated interaction FSUGold (Todd-Rutel & Piekarewicz 2005), we include the δ meson and the crossing term $\delta { - } \sigma$ in the RMF model, and find that the $\delta { - } \sigma$ mixing can soften the $E _ { \mathrm { s y m } } ( n )$ at intermediate densities while stifen the $E _ { \mathrm { s y m } } ( n )$ at high densities. Our results indicate that the new RMF model can remove the tension between the PREX-II and GW170817 in the conventional RMF model, and at the same time it is compatible with the constraints on the EOS of SNM at suprasaturation densities from flow data in HIC, the largest NS mass reported so far from PSR J0740+6620, and the M-R relation of PSR J0030+0451 and PSR J0740+6620 measured by NICER.

## 2. FORMALISM

The EOS of isospin asymmetric matter, defined by its binding energy per nucleon, can be expressed as

$$
E (n, \alpha) = E _ {0} (n) + E _ {\mathrm{sym}} (n) \alpha^ {2} + O (\alpha^ {4})\tag{2}
$$

where $n = n _ { n } + n _ { p }$ is the baryon density with $n _ { n } \left( n _ { p } \right)$ denoting the neutron (proton) density, $\alpha = ( n _ { n } - n _ { p } ) / ( n _ { n } +$ $n _ { p } )$ is the isospin asymmetry, $E _ { 0 } ( n ) = E ( n , \alpha = 0 )$ is the EOS of SNM, and the symmetry energy $E _ { \mathrm { s y m } } ( n )$ is defined as

$$
E _ {\mathrm{sym}} (n) = \frac {1}{2} \frac {\partial^ {2} E (n , \alpha)}{\partial \alpha^ {2}} | _ {\alpha = 0}.\tag{3}
$$

At nuclear matter saturation density $n _ { 0 }$ , the $E _ { 0 } ( n )$ can be expanded in $\chi \equiv ( n - n _ { 0 } ) / 3 n _ { 0 }$ as

$$
E _ {0} (n) = E _ {0} (n _ {0}) + \frac {K _ {0}}{2 !} \chi^ {2} + \frac {J _ {0}}{3 !} \chi^ {3} + O (\chi^ {4})\tag{4}
$$

where $E _ { 0 } ( n _ { 0 } )$ is the binding energy per nucleon in SNM at $n _ { 0 }$ $\begin{array} { l } { { \displaystyle K _ { 0 } = 9 n _ { 0 } ^ { 2 } \frac { \partial ^ { 2 } E _ { 0 } ( n ) } { \partial n ^ { 2 } } \ \left| _ { n = n _ { 0 } } \right. } } \end{array}$ is the incompressibility coeficient, and $J _ { 0 } = 2 7 n _ { 0 } ^ { 3 } \frac { \partial ^ { 3 } E _ { 0 } ( n ) } { \partial n ^ { 3 } } \ | _ { n = n _ { 0 } }$ is the skewness coeficient. Similarly, the $E _ { \mathrm { s y m } } ( n )$ can be expanded at a

reference density $n _ { r }$ as

$$
E _ {\mathrm{sym}} (n) = E _ {\mathrm{sym}} (n _ {r}) + L (n _ {r}) \chi_ {r} + O (\chi_ {r} ^ {2})\tag{5}
$$

with $\chi _ { r } ~ = ~ ( n - n _ { r } ) / ( 3 n _ { r } )$ , and the symmetry energy density slope $L ( n _ { r } )$ at $n _ { r }$ can be obtained as

$$
L (n _ {r}) = 3 n _ {r} \frac {\partial E _ {\mathrm{sym}} (n)}{\partial n} | _ {n = n _ {r}}.\tag{6}
$$

The conventional $L \equiv L ( n _ { 0 } )$ is recovered by setting $n _ { r } ~ = ~ n _ { 0 }$ . In particular, the $L _ { c } \equiv L ( n _ { c } )$ is obtained at a subsaturation cross density $n _ { c \mathrm { ~ } } ~ = ~ 0 . 1 1 / 0 . 1 6 n _ { 0 }$ 8 which essentially determines the neutron skin thickness of <sup>208</sup>Pb (Zhang & Chen 2013).

Based on the accurately calibrated interaction FSUGold (Todd-Rutel & Piekarewicz 2005), we consider here the isovector scalar δ meson and its coupling to the isoscalar scalar σ meson in the RMF model. The Lagrangian density L of the nucleon system is given by

$$
\begin{array}{r l} & {\mathcal {L} = \overline {{\psi}} (i \partial_ {\mu} \gamma^ {\mu} - m) \psi} \\ & {\quad + g _ {\sigma} \sigma \overline {{\psi}} \psi - g _ {\omega} \omega_ {\mu} \overline {{\psi}} \gamma^ {\mu} \psi - g _ {\rho} \vec {\rho} _ {\mu} \overline {{\psi}} \gamma^ {\mu} \vec {\tau} \psi + g _ {\delta} \overrightarrow {\delta \psi} \tau \psi} \\ & {\quad + \frac {1}{2} (\partial_ {\mu} \sigma \partial^ {\mu} \sigma - m _ {\sigma} ^ {2} \sigma^ {2}) - \frac {1}{3} b _ {\sigma} m (g _ {\sigma} \sigma) ^ {3} - \frac {1}{4} c _ {\sigma} (g _ {\sigma} \sigma) ^ {4}} \\ & {\quad - \frac {1}{4} \omega_ {\mu \nu} \omega^ {\mu \nu} + \frac {1}{2} m _ {\omega} ^ {2} \omega_ {\mu} \omega^ {\mu} + \frac {1}{4} c _ {\omega} (g _ {\omega} ^ {2} \omega_ {\mu} \omega^ {\mu}) ^ {2}} \\ & {\quad - \frac {1}{4} \rho_ {\mu \nu} \rho^ {\mu \nu} + \frac {1}{2} m _ {\rho} ^ {2} \vec {\rho} _ {\mu} \vec {\rho} ^ {\mu} + \frac {1}{2} \Lambda_ {V} (g _ {\rho} ^ {2} \vec {\rho} _ {\mu} \vec {\rho} ^ {\mu}) (g _ {\omega} ^ {2} \omega_ {\mu} \omega^ {\mu})} \\ & {\quad + \frac {1}{2} (\partial_ {\mu} \vec {\delta} \partial^ {\mu} \vec {\delta} - m _ {\delta} ^ {2} \vec {\delta} ^ {2}) + \frac {1}{2} C _ {\delta \sigma} (g _ {\sigma} ^ {2} \sigma^ {2}) (g _ {\delta} ^ {2} \vec {\delta} ^ {2}) (7)} \end{array}
$$

where m is the nucleon mass, $m _ { \sigma } , m _ { \omega } , m _ { \rho }$ and $m _ { \delta }$ are the meson masses, and $g _ { \sigma } , g _ { \omega } , g _ { \rho }$ and $g _ { \delta }$ represent coupling constants for nucleons with the corresponding mesons. In addition, $b _ { \sigma }$ and $c _ { \sigma }$ are dimensionless for the selfinteraction of the σ meson, $c _ { \omega }$ is for the self-interaction of the ω meson, $\Lambda _ { V }$ is the coupling coeficient between ρ meson and ω meson, and $C _ { \delta \sigma }$ is the coupling coeficient between σ meson and $\delta$ meson. We note that the Lagrangian density of the FSUGold interaction (without including the electromagnetic interactions) can be obtained by removing the terms associated with δ meson in Eq. (7).

With the standard Euler-Lagrange equation

$$
\partial_ {\mu} (\frac {\partial \mathcal {L}}{\partial (\partial_ {\mu} \phi)}) - \frac {\partial \mathcal {L}}{\partial \phi} = 0,\tag{8}
$$

one can obtain equations of motion for the nucleon and meson fields from the Lagrangian density. The resulting Dirac equation for the nucleon field is

$$
[ \gamma^ {\mu} (i \partial_ {\mu} + \Sigma_ {\mu} ^ {J}) - (m + \Sigma_ {s} ^ {J}) ] \Psi = 0\tag{9}
$$

with the nucleon scalar self-energy and vector self-energy given, respectively, by

$$
\Sigma_ {s} ^ {J} = - g _ {\sigma} \sigma - g _ {\delta} \vec {\delta} \cdot \vec {\tau},\tag{10a}
$$

$$
\Sigma_ {\mu} ^ {J} = g _ {\omega} \omega_ {\mu} + g _ {\rho} \vec {\rho} _ {\mu} \cdot \vec {\tau}.\tag{10b}
$$

The equations of motion for the meson fields can be obtained as

$$
(\partial_ {\mu} \partial^ {\mu} + m _ {\sigma} ^ {2}) \sigma = g _ {\sigma} [ \overline {{\Psi}} \Psi - b _ {\sigma} M (g _ {\sigma} \sigma) ^ {2} - c _ {\sigma} (g _ {\sigma} \sigma) ^ {3}
$$

$$
+ C _ {\delta \sigma} (g _ {\sigma} \sigma) (g _ {\delta} \vec {\delta}) ^ {2} ],
$$

$$
\partial_ {\mu} \omega^ {\mu \nu} + m _ {\omega} ^ {2} \omega^ {\nu} = g _ {\omega} [ \overline {{\Psi}} \gamma^ {\nu} \Psi - c _ {\omega} g _ {\omega} ^ {3} \omega_ {\mu} \omega^ {\mu} \omega^ {\nu}\tag{11}
$$

$$
\left. - \Lambda_ {V} (g _ {\omega} \omega^ {\nu}) g _ {\rho} ^ {2} \vec {\rho} _ {\mu} \vec {\rho} ^ {\mu} \right],\tag{12}
$$

$$
\partial_ {\mu} \vec {\rho} ^ {\mu \nu} + m _ {\rho} ^ {2} \vec {\rho} ^ {\nu} = g _ {\rho} [ \overline {{\Psi}} \gamma^ {\nu} \vec {\tau} \Psi - \Lambda_ {V} (g _ {\rho} \vec {\rho} ^ {\nu}) g _ {\omega} ^ {2} \omega_ {\mu} \omega^ {\mu} ], (1 3)
$$

$$
(\partial_ {\mu} \partial^ {\mu} + m _ {\delta} ^ {2}) \vec {\delta} = g _ {\delta} [ \overline {{\Psi}} \vec {\tau} \Psi + C _ {\delta \sigma} (g _ {\delta} \vec {\delta}) (g _ {\sigma} \sigma) ^ {2} ].\tag{14}
$$

For a static and homogenous infinite nuclear matter, all derivative terms drop out and the expectation values of spacelike components of vector fields vanish (only zero components $\vec { \rho } _ { 0 }$ and $\omega _ { 0 }$ survive) due to translational invariance and rotational symmetry of the nuclear matter. In addition, only the third component of isovector fields $( \delta ^ { ( 3 ) }$ and $\rho ^ { ( 3 ) } )$ needs to be considered because of the rotational invariance around the third axis in the isospin space. In the mean-field approximation, meson fields are replaced by their expectation values, i.e., $\sigma  \overline { { \sigma } } .$ 2 $\omega _ { \mu }  \overline { { \omega } } _ { 0 } , \rho _ { \mu }  \overline { { \rho } } _ { 0 } ^ { ( 3 ) }$ , and $\delta  \overline { { \delta } } ^ { ( 3 ) }$ , and the equations of motion for the meson fields are reduced to

$$
\begin{array}{r} m _ {\sigma} ^ {2} \overline {{\sigma}} = g _ {\sigma} [ n ^ {s} - b _ {\sigma} M (g _ {\sigma} \overline {{\sigma}}) ^ {2} - c _ {\sigma} (g _ {\sigma} \overline {{\sigma}}) ^ {3} \\ + C _ {\delta \sigma} (g _ {\sigma} \overline {{\sigma}}) (g _ {\delta} \overline {{\delta}} ^ {(3)}) ^ {2} ], \end{array}\tag{15}
$$

$$
m _ {\omega} ^ {2} \overline {{\omega}} _ {0} = g _ {\omega} [ n - c _ {\omega} (g _ {\omega} \overline {{\omega}} _ {0}) ^ {3} - \Lambda_ {V} (g _ {\omega} \overline {{\omega}} _ {0}) (g _ {\rho} \overline {{\rho}} _ {0} ^ {(3)}) ^ {2} ] (1 6)
$$

$$
m _ {\rho} ^ {2} \overline {{\rho}} _ {0} ^ {(3)} = g _ {\rho} [ (n _ {p} - n _ {n}) - \Lambda_ {V} (g _ {\rho} \overline {{\rho}} _ {0} ^ {(3)}) (g _ {\omega} \overline {{\omega}} _ {0}) ^ {2} ],\tag{17}
$$

$$
m _ {\delta} ^ {2} \overline {{\delta}} ^ {(3)} = g _ {\delta} [ (n _ {p} ^ {s} - n _ {n} ^ {s}) + C _ {\delta \sigma} (g _ {\delta} \overline {{\delta}} ^ {(3)}) (g _ {\sigma} \overline {{\sigma}}) ^ {2} ],\tag{18}
$$

where the nucleon scalar density $n ^ { s }$ and vector density are defined as

$$
n ^ {s} = \langle \overline {{\Psi}} \Psi \rangle = n _ {n} ^ {s} + n _ {p} ^ {s},\tag{19a}
$$

$$
n = \langle \overline {{\Psi}} \gamma^ {0} \Psi \rangle = n _ {n} + n _ {p}\tag{19b}
$$

with neutron scalar density $n _ { n } ^ { s }$ and proton scalar density $n _ { p } ^ { s }$ given by

$$
\begin{array}{l} n _ {J} ^ {s} = \frac {2}{(2 \pi) ^ {3}} \int_ {0} ^ {k _ {F} ^ {J}} \frac {m _ {J} ^ {*}}{\sqrt {(\vec {k}) ^ {2} + (m _ {J} ^ {*}) ^ {2}}} d \vec {k} \\ = \frac {m _ {J} ^ {*}}{2 \pi^ {2}} [ k _ {F} ^ {J} \epsilon_ {F} ^ {*, J} - (m _ {J} ^ {*}) ^ {2} \ln \frac {k _ {F} ^ {J} + \epsilon_ {F} ^ {* , J}}{m _ {J} ^ {*}} ], J = n, p, (2 0) \end{array}
$$

where $\epsilon _ { F } ^ { * , J }$ and $k _ { F } ^ { J }$ are expressed as

$$
\begin{array}{c} \epsilon_ {F} ^ {*, J} = \sqrt {(k _ {F} ^ {J}) ^ {2} + (m _ {J} ^ {*}) ^ {2}}, \\ k _ {F} ^ {J} = k _ {F} (1 + \tau_ {3} ^ {J} \alpha) ^ {\frac {1}{3}}, k _ {F} = (\frac {3}{2} \pi^ {2} n) ^ {\frac {1}{3}} \end{array}\tag{21}
$$

with $k _ { F }$ being the Fermi momentum of SNM. The efective (Dirac) masses of the neutron $\boldsymbol { m } _ { n } ^ { * }$ and proton $m _ { p } ^ { * }$ can be expressed as

$$
m _ {J} ^ {*} = m - g _ {\sigma} \overline {{\sigma}} - g _ {\delta} \overline {{\delta}} ^ {(3)} \tau_ {3} ^ {J}, J = n, p.\tag{22}
$$

In addition, the nucleon scalar self-energy and vector self-energy are given by

$$
\Sigma_ {s} ^ {J} = - g _ {\sigma} \overline {{\sigma}} - g _ {\delta} \overline {{\delta}} ^ {(3)} \tau_ {3} ^ {J},\tag{23a}
$$

$$
\Sigma_ {\mu} ^ {J} = g _ {\omega} \overline {{\omega}} _ {0} + g _ {\rho} \overline {{\rho}} _ {0} ^ {(3)} \tau_ {3} ^ {J}\tag{23b}
$$

with $\tau _ { 3 } = 1 ( - 1 )$ for protons (neutrons).

The set of coupled equations for the nucleon and meson fields can be solved self-consistently using the iteration method, and the properties of the nuclear matter can then be obtained from these fields. According to the energy-momentum tensor

$$
\mathcal {T} ^ {\mu \nu} = \sum_ {\phi_ {a}} \frac {\partial \mathcal {T}}{\partial \partial_ {\mu} \phi_ {a}} \partial^ {\nu} \phi_ {a} - \mathcal {T} g ^ {\mu \nu},\tag{24}
$$

the energy density  and the pressure $P$ can be obtained as

$$
\begin{array}{l} \epsilon = \langle \mathcal {T} ^ {0 0} \rangle = \epsilon_ {p} ^ {\mathrm{kin}} + \epsilon_ {n} ^ {\mathrm{kin}} \\ \quad + \frac {1}{2} m _ {\sigma} ^ {2} (\overline {{\sigma}}) ^ {2} + \frac {1}{2} m _ {\omega} ^ {2} (\overline {{\omega}} _ {0}) ^ {2} + \frac {1}{2} m _ {\rho} ^ {2} (\overline {{\rho}} _ {0} ^ {(3)}) ^ {2} + \frac {1}{2} m _ {\delta} ^ {2} (\overline {{\delta}} ^ {(3)}) ^ {2} \\ \quad + \frac {1}{3} m b _ {\sigma} (g _ {\sigma} \overline {{\sigma}}) ^ {3} + \frac {1}{4} c _ {\sigma} (g _ {\sigma} \overline {{\sigma}}) ^ {4} + \frac {3}{4} c _ {\omega} (g _ {\omega} \overline {{\omega}} _ {0}) ^ {4} \\ \quad + \frac {3}{2} \Lambda_ {V} (g _ {\rho} \overline {{\rho}} _ {0} ^ {(3)}) ^ {2} (g _ {\omega} \overline {{\omega}} _ {0}) ^ {2} - \frac {1}{2} C _ {\delta \sigma} (g _ {\delta} \overline {{\delta}} ^ {(3)}) ^ {2} (g _ {\sigma} \overline {{\sigma}}) ^ {2} \end{array} \tag {25}
$$

and

$$
\begin{array}{l} P = \frac {1}{3} \sum_ {j = 1} ^ {3} \langle \mathcal {T} ^ {j j} \rangle = P _ {p} ^ {\mathrm{kin}} + P _ {n} ^ {\mathrm{kin}} \\ - \frac {1}{2} m _ {\sigma} ^ {2} (\overline {{\sigma}}) ^ {2} + \frac {1}{2} m _ {\omega} ^ {2} (\overline {{\omega}} _ {0}) ^ {2} + \frac {1}{2} m _ {\rho} ^ {2} (\overline {{\rho}} _ {0} ^ {(3)}) ^ {2} - \frac {1}{2} m _ {\delta} ^ {2} (\overline {{\delta}} ^ {(3)}) ^ {2} \\ - \frac {1}{3} m b _ {\sigma} (g _ {\sigma} \overline {{\sigma}}) ^ {3} - \frac {1}{4} c _ {\sigma} (g _ {\sigma} \overline {{\sigma}}) ^ {4} + \frac {1}{4} c _ {\omega} (g _ {\omega} \overline {{\omega}} _ {0}) ^ {4} \\ + \frac {1}{2} \Lambda_ {V} (g _ {\rho} \overline {{\rho}} _ {0} ^ {(3)}) ^ {2} (g _ {\omega} \overline {{\omega}} _ {0}) ^ {2} + \frac {1}{2} C _ {\delta \sigma} (g _ {\delta} \overline {{\delta}} ^ {(3)}) ^ {2} (g _ {\sigma} \overline {{\sigma}}) ^ {2}, \end{array} \tag {26}
$$

where $\epsilon _ { J } ^ { \mathrm { k i n } }$ and $P _ { J } ^ { \mathrm { k i n } }$ are the kinetic contributions to the energy densities and pressure of the nuclear matter, respectively, and they are given as follow

$$
\begin{array}{r l} & {\epsilon_ {J} ^ {\mathrm{kin}} = \frac {2}{(2 \pi) ^ {3}} \int_ {0} ^ {k _ {F} ^ {J}} \sqrt {(\vec {k}) ^ {2} + (m _ {J} ^ {*}) ^ {2}} d \vec {k}} \\ & {\quad = \frac {1}{(\pi) ^ {2}} \int_ {0} ^ {k _ {F} ^ {J}} k ^ {2} \sqrt {k ^ {2} + (m _ {J} ^ {*}) ^ {2}} d k} \\ & {\quad = \frac {1}{4} [ 3 \epsilon_ {F} ^ {*, J} n _ {J} + m _ {J} ^ {*} n _ {J} ^ {s} ], J = n, p} \end{array}\tag{27}
$$

and

$$
\begin{array}{r} P _ {J} ^ {\mathrm{kin}} = \frac {2}{3 (2 \pi) ^ {3}} \int_ {0} ^ {k _ {F} ^ {J}} \frac {(\vec {k}) ^ {2}}{\sqrt {(\vec {k}) ^ {2} + (m _ {J} ^ {*}) ^ {2}}} d ^ {3} k \\ = \frac {1}{4} [ \epsilon_ {F} ^ {*, J} n _ {J} - m _ {J} ^ {*} n _ {J} ^ {s} ], J = n, p. \end{array}\tag{28}
$$

The binding energy per nucleon can be obtained from the energy density via

$$
E = E (n, \alpha = 0) = \frac {\epsilon}{n} - m,\tag{29}
$$

and the analytical expressions of $E _ { \mathrm { s y m } } ( n )$ and $L ( n )$ can be obtained from their definitions Eqs. (3) and (6) as

$$
\begin{array}{r l} & E _ {\mathrm{sym}} (n) = \frac {k _ {F} ^ {2}}{6 \epsilon_ {F} ^ {*}} + \frac {1}{2} (\frac {g _ {\rho}}{m _ {\rho} ^ {*}}) ^ {2} n \\ & \qquad - \frac {1}{2} (\frac {g _ {\delta}}{m _ {\delta} ^ {*}}) ^ {2} \frac {(m _ {0} ^ {*}) ^ {2} n}{(\epsilon_ {F} ^ {*}) ^ {2} [ 1 + (\frac {g _ {\delta}}{m _ {\delta} ^ {*}}) ^ {2} A (k _ {F} , m _ {0} ^ {*}) ]} \end{array}\tag{30}
$$

and

$$
\begin{array}{r l} & L (n) = \frac {k _ {F} ^ {2} [ (\epsilon_ {F} ^ {*}) ^ {2} + (m _ {0} ^ {*}) ^ {2} ]}{6 (\epsilon_ {F} ^ {*}) ^ {3}} + \frac {g _ {\sigma} ^ {2} (m _ {0} ^ {*}) ^ {2} k _ {F} ^ {2} n}{2 Q _ {\sigma} (\epsilon_ {F} ^ {*}) ^ {4}} \\ & \qquad + \frac {3 g _ {\rho} ^ {2} n}{2 Q _ {\rho}} - \frac {3 g _ {\omega} ^ {3} g _ {\rho} ^ {4} \Lambda_ {V} \overline {{\omega}} _ {0} n ^ {2}}{Q _ {\omega} Q _ {\rho} ^ {2}} \\ & \qquad - \frac {3 g _ {\sigma} g _ {\delta} ^ {2} (m _ {0} ^ {*}) ^ {2} n ^ {2}}{2 (\epsilon_ {F} ^ {*}) ^ {3} Q _ {\sigma} Q _ {\delta}} [ \frac {2 g _ {\sigma} (m _ {0} ^ {*}) ^ {2}}{(\epsilon_ {F} ^ {*}) ^ {2}} + \frac {m _ {0} ^ {*} \Gamma}{Q _ {\delta}} - 2 g _ {\sigma} ] \\ & \qquad - \frac {3 g _ {\delta} ^ {2} (m _ {0} ^ {*}) ^ {2} n}{2 (\epsilon_ {F} ^ {*}) ^ {2} Q _ {\delta}} [ 1 - \frac {2 k _ {F} ^ {2}}{3 (\epsilon_ {F} ^ {*}) ^ {2}} - \frac {n \Phi}{2 Q _ {\delta}} ] \end{array}\tag{31}
$$

where the Γ and Φ are defined as

$$
\begin{array}{l} \Gamma = 3 g _ {\sigma} g _ {\delta} ^ {2} [ \frac {2 n _ {0} ^ {s}}{(m _ {0} ^ {*}) ^ {2}} - \frac {3 n}{m _ {0} ^ {*} \epsilon_ {F} ^ {*}} + \frac {m _ {0} ^ {*} n}{(\epsilon_ {F} ^ {*}) ^ {3}} ] + 2 C _ {\delta \sigma} g _ {\delta} ^ {2} g _ {\sigma} ^ {2} \overline {{\sigma}}, \\ \Phi = \frac {2 g _ {\delta} ^ {2} k _ {F} ^ {2}}{(\epsilon_ {F} ^ {*}) ^ {3}} \end{array}\tag{32}
$$

and

$$
Q _ {\sigma} = m _ {\sigma} ^ {2} + g _ {\sigma} ^ {2} A (k _ {F}, m _ {0} ^ {*}) + 2 m b _ {\sigma} g _ {\sigma} ^ {3} \overline {{\sigma}} + 3 c _ {\sigma} g _ {\sigma} ^ {4} \overline {{\sigma}} ^ {2}\tag{33a}
$$

$$
Q _ {\omega} = m _ {\omega} ^ {2} + 3 c _ {\omega} g _ {\omega} ^ {4} \overline {{\omega}} _ {0} ^ {2}\tag{33b}
$$

$$
Q _ {\rho} = (m _ {\rho} ^ {*}) ^ {2} = m _ {\rho} ^ {2} + \Lambda_ {V} g _ {\rho} ^ {2} g _ {\omega} ^ {2} \overline {{\omega}} _ {0} ^ {2}\tag{33c}
$$

$$
Q _ {\delta} = (m _ {\delta} ^ {*}) ^ {2} + g _ {\delta} ^ {2} A (k _ {F}, m _ {0} ^ {*})\tag{33d}
$$

In addition, the $( m _ { \delta } ^ { * } ) ^ { 2 }$ and $A ( k _ { F } , m _ { 0 } ^ { * } )$ are given by

$$
(m _ {\delta} ^ {*}) ^ {2} = m _ {\delta} ^ {2} - C _ {\delta \sigma} g _ {\delta} ^ {2} g _ {\sigma} ^ {2} \overline {{\sigma}} ^ {2}\tag{34}
$$

and

$$
\begin{array}{c} A (k _ {F}, m _ {0} ^ {*}) = \frac {4}{(2 \pi) ^ {3}} \int_ {0} ^ {k _ {F}} \frac {(\vec {k}) ^ {2}}{\sqrt {[ (\vec {k}) ^ {2} + (m _ {0} ^ {*}) ^ {2} ] ^ {\frac {3}{2}}} d ^ {3} k} \\ = 3 (\frac {n ^ {s}}{m _ {0} ^ {*}} - \frac {n}{\epsilon_ {F} ^ {*}}) \end{array}\tag{35}
$$

with $\epsilon _ { F } ^ { * } = \sqrt { k _ { F } ^ { 2 } + ( m _ { 0 } ^ { * } ) ^ { 2 } }$ and $m _ { 0 } ^ { * } = m - g _ { \sigma } \overline { { { \sigma } } }$ is the nucleon Dirac mass in symmetric nuclear matter.

The analytical expressions of $E _ { \mathrm { s y m } } ( n ) ~ ( \mathrm { E q . ~ ( 3 0 ) } )$ and $L ( n ) ~ ( { \mathrm { E q . ~ } } ( 3 1 ) )$ are very useful and convenient for exploring the properties of asymmetric nuclear matter. We note that the expression Eq. (31) is an extension of the analytical expression for $L ( n )$ in the nonlinear RMF model (e.g., the FSUGold model) in Refs. (Cai & Chen 2012a) by considering the contribution of $\delta { - } \sigma$ mixing.

For NSs, we adopt here the conventional NS model, i.e., the NS contains core, inner crust and outer crust, and the core is assumed to be composed of only neutrons, protons, electrons and possible muons (npeµ) and its EOS can be calculated within the RMF models. For the details, one can refer to Refs. (Zhou et al. 2019; Zhou & Chen 2019; Xu et al. 2009; Zhang & Chen 2016). We would like to point out that in this work, the core-crust transition density $n _ { t }$ , which separates the liquid core from the solid inner crust of neutron stars, is obtained self-consistently by the thermodynamical method (Xu et al. 2009; Kubis 2007; Cai & Chen 2012b). In addition, the causality condition is satisfied for all the EOSs used in the following NS calculations.

## 3. RESULTS AND DISCUSSIONS

The FSUGold interaction (Todd-Rutel & Piekarewicz 2005) is an accurately calibrated relativistic parametrization in the non-linear RMF model and it can very successfully describe the ground state binding energy and charge radius of finite nuclei as well as their linear response. In the FSUGold model (Todd-Rutel $\&$ Piekarewicz 2005; Fattoyev et al. 2010), the nucleon mass is fixed at $m = 9 3 9 . 0 \ \mathrm { M e V }$ , and the three meson masses are also fixed, namely, $m _ { \sigma } = 4 9 1 . 5 ~ \mathrm { M e V }$ $m _ { \omega } = 7 8 2 . 5$ MeV and $m _ { \rho } = 7 6 3 . 0 \mathrm { M e V }$ . In this case, the FSUGold has thus seven model parameters, i.e., $g _ { \sigma } , g _ { \omega } ,$ $g _ { \rho } , b _ { \sigma } , c _ { \sigma } , c _ { \omega }$ and $\Lambda _ { V }$ , which are listed in TABLE 1. The bulk properties of infinite nuclear matter of FSUGold can be obtained as: $n _ { 0 } = 0 . 1 4 8 \mathrm { f m } ^ { - 3 }$ , the binding energy per nucleon of SNM at $n _ { 0 }$ is $e _ { 0 } \equiv E _ { 0 } ( n _ { 0 } ) = - 1 6 . 3 1 \mathrm { { M e V } }$ $K _ { 0 } = 2 2 9 . 2 \ \mathrm { M e V }$ , the nucleon Dirac mass in SNM at $n _ { 0 }$ is $m _ { \mathrm { D i r a c } } ^ { * } ~ = ~ m _ { 0 } ^ { * } / m ~ = ~ 0 . 6 1$ , the neutron (proton)

Dirac mass in asymmetric nuclear matter at $n _ { 0 }$ and $\alpha = 0 . 5 \mathrm { ~ i s ~ } m _ { \mathrm { D i r a c . n . 0 . 5 } } ^ { * } = m _ { n } ^ { * } ( n = n _ { 0 } , \alpha = 0 . 5 ) / m = 0 . 6 1$ $( m _ { \mathrm { D i r a c . p . 0 . 5 } } ^ { * } ~ = ~ m _ { p } ^ { * } ( n ~ = ~ n _ { 0 } , \alpha ~ = ~ 0 . 5 ) / m ~ = ~ 0 . 6 1 )$ $J _ { 0 } = - 5 2 1 . 6 ~ \mathrm { M e V } , ~ E _ { \mathrm { s y m } , c } \equiv E _ { \mathrm { s y m } } ( n _ { c } ) = 2 5 . 8 8 ~ \mathrm { M e V }$ and $L _ { c } ~ = ~ 4 8 . 0 2$ MeV. In addition, the FSUGold predicts $E _ { \mathrm { s y m } } ( n _ { 0 } ) ~ = ~ 3 2 . 5 9$ MeV, $L ~ = ~ 6 0 . 5 ~ \mathrm { { \ M e V } } ,$ $n _ { t } ~ = ~ 0 . 0 8 2 ~ \mathrm { ~ f m ^ { - 3 } }$ , the central density of the maximum mass NS configuration $\begin{array} { l c l } { n _ { \mathrm { c e n } } } & { = } & { 1 . 1 5 ~ \ \mathrm { \ f m ^ { - 3 } } } \end{array}$ $R _ { 1 , 4 } ~ = ~ 1 2 . 6 6$ km, $\Lambda _ { 1 . 4 } ~ = ~ 4 0 1$ , and $M _ { \mathrm { m a x } } = 1 . 7 2 M _ { \odot }$ TABLE 2 summarizes these results. Also included in TABLE 2 are the FSUGold RMF predictions of some ground state properties of doubly magic nuclei ${ } ^ { 2 0 8 } \mathrm { { P b } }$ and $^ { 4 8 } \mathrm { C a }$ , i.e., the binding energy per nucleon $E _ { B } ^ { A }$ , the charge rms radius $R _ { \mathrm { c h } } ^ { A }$ , the neutron skin thickness $\Delta r _ { \mathrm { n o } } ^ { A }$ with $A = 2 0 8 \ ( 4 8 )$ $\mathrm { f o r ^ { 2 0 8 } P b \ ( ^ { 4 8 } C a ) }$ , as well as $\mathrm { t h e ^ { 2 0 8 } \vec { P b } }$ spin-orbit splittings of proton 2d state $\epsilon _ { l s , 2 d \pi } ^ { 2 0 8 }$ , neutron 3p state $\epsilon _ { l s , 3 p \nu } ^ { 2 0 8 }$ and neutron $2 f$ state $\epsilon _ { l s , 2 f \nu } ^ { 2 0 8 }$

One sees from TABLE 2 that while the FSUGold nicely reproduces the ground state properties of fi nite nuclei, it predicts a too small value of $M _ { \mathrm { m a x } } ~ =$ $1 . 7 2 M _ { \odot }$ , which is significantly smaller than the observed $2 . 0 8 _ { - 0 . 0 7 } ^ { + 0 . 0 7 } M _ { \odot }$ for PSR J0740+6620 (Cromartie et al. 2020; Fonseca et al. 2021). In addition, we note the $\Delta r _ { \mathrm { n p } }$ of ${ } ^ { 2 0 8 } \mathrm { { P b } }$ is 0.207 fm with the FSUGold, which is smaller than the PREX-II measurement. Noting the variation of the higher-order bulk parameters $\left( \mathrm { e . g . , ~ } J _ { 0 } \right)$ essentially does not change the properties of finite nuclei (Zhou et al. 2019), we thus increase $J _ { 0 }$ while keep n<sub>0</sub>, e<sub>0</sub>, $K _ { 0 } , \ m _ { \mathrm { D i r a c } } ^ { * }$ and $E _ { \mathrm { s y m , c } }$ unchanged to enhance the $M _ { \mathrm { m a x } }$ At the same time, we increase $L _ { c }$ from original 48.02 MeV to 55 MeV to fit the constraint of $L _ { c } \gtrsim 5 2$ MeV (Yue et al. 2021) from PREX-II. The $L _ { c } ~ = ~ 5 5$ MeV corresponds to $L ( 2 / 3 n _ { 0 } ) ~ = ~ 5 4$ MeV, again consistent with the constraint of $L ( 2 / 3 n _ { 0 } ) ~ \stackrel { > } { \sim }$ 49 MeV (Reed et al. 2021) from PREX-II. Indeed, the $L _ { c } = 5 5$ MeV leads to $\Delta r _ { \mathrm { n p } } \approx 0 . 2 2$ fm for ${ } ^ { 2 0 8 } \mathrm { { P b } }$ according to the relation between $\Delta r _ { \mathrm { n p } } ~ \mathrm { o f ~ ^ { 2 0 8 } P b }$ and $L _ { c } ,$ i.e., $\Delta r _ { \mathrm { n p } } = ( 0 . 0 6 1 5 \pm 0 . 0 0 2 2 ) + ( 0 . 0 0 2 8 2 \pm 0 . 0 0 0 0 3 4 4 ) L _ { c }$ (Yue et al. 2021). In addition, we note that the $\begin{array} { r l } { L _ { c } } & { { } = } \end{array}$ 55 MeV produces $\Lambda _ { 1 . 4 } ~ = ~ 5 7 6$ , and $L _ { c } ~ > ~ 5 6$ MeV will violate the constraint $\Lambda _ { 1 . 4 } ~ \leq ~ 5 8 0$ Increasing $J _ { 0 }$ can lead to larger $M _ { \mathrm { m a x } } .$ , but the $J _ { 0 }$ cannot be larger than −322.0 MeV to fit the constraint on the SNM EOS from flow data (Danielewicz et al. 2002). One then obtains a value of $M _ { \mathrm { { m a x } } } ~ = ~ 1 . 9 3 M _ { \odot }$ with $J _ { 0 } \ =$ −322.0 MeV, suggesting the FSUGold Lagrangian cannot simultaneously describe the observed $2 . 0 8 _ { - 0 . 0 7 } ^ { + 0 . 0 7 } M _ { \odot }$ for PSR J0740+6620 (Cromartie et al. 2020; Fonseca et al. 2021) and the constraint on the SNM EOS from flow data (Danielewicz et al. 2002). From the seven bulk parameters $n _ { 0 } = 0 . 1 4 8 ~ \mathrm { f m ^ { - 3 } } , ~ e _ { 0 } = - 1 6 . 3 1 ~ \mathrm { M e V }$ $K _ { 0 } = 2 2 9 . 2 ~ \mathrm { M e V } , \ : m _ { \mathrm { D i r a c } } ^ { * } = 0 . 6 1 , \ : J _ { 0 } = - 3 2 2 . 0 ~ \mathrm { M e V }$

Table 1. Model parameters for the four models FSUGold, FSU-J0, FSU-δ6.7 and FSU-δ6.2.

<table><tr><td>Model</td><td> $g_{\sigma}$ </td><td> $g_{\omega}$ </td><td> $g_{\rho}$ </td><td> $g_{\delta}$ </td><td> $b_{\sigma}(10^{-3})$ </td><td> $c_{\sigma}(10^{-3})$ </td><td> $c_{\omega}(10^{-2})$ </td><td> $\Lambda_V$ </td><td> $C_{\delta\sigma}(10^{-2})$ </td></tr><tr><td>FSUGold</td><td>10.5924</td><td>14.3020</td><td>5.88367</td><td>-</td><td>0.756283</td><td>3.96033</td><td>1.00000</td><td>0.240000</td><td>-</td></tr><tr><td>FSU-J0</td><td>10.2143</td><td>13.4353</td><td>5.31345</td><td>-</td><td>1.59524</td><td>0.540269</td><td>0.528340</td><td>0.171042</td><td>-</td></tr><tr><td>FSU- $\delta 6.7$ </td><td>10.2143</td><td>13.4353</td><td>7.26866</td><td>6.70000</td><td>1.59524</td><td>0.540269</td><td>0.528340</td><td>0.0214000</td><td>3.85000</td></tr><tr><td>FSU- $\delta 6.2$ </td><td>10.2143</td><td>13.4353</td><td>6.98518</td><td>6.20000</td><td>1.59524</td><td>0.540269</td><td>0.528340</td><td>0.0200112</td><td>5.40000</td></tr></table>

Table 2. Nuclear matter bulk parameters, NS properties and ground state properties of $^ { 2 0 8 } \mathrm { { P b } }$ and $^ { 4 8 } \mathrm { { C a } }$ in FSUGold, FSU-J0, FSU-δ6.7 and FSU-δ6.2. For comparison, the nuclear ground state data are listed as following: the binding energy $E _ { B } ^ { 2 0 8 } = - 7 . 8 7$ MeV and $E _ { B } ^ { 4 8 } = - 8 . 6 ^ { \prime }$ 7 MeV (Wang et al. 2021), the charge rms radius $R _ { \mathrm { c h } } ^ { 2 0 8 } ~ = ~ 5 . 5 0$ fm and $R _ { \mathrm { c h } } ^ { 4 8 } ~ = ~ 3 . 4 8$ fm (Angeli & Marinova 2013), and the spinorbit splitting $\epsilon _ { l s , 2 d \pi } ^ { 2 0 8 } = 1 . 3 2$ MeV, $\epsilon _ { l s , 3 p \nu } ^ { 2 0 8 } = 0 . 8 9$ MeV and $\epsilon _ { l s , 2 f \nu } ^ { 2 0 8 } = 1 . 7 7$ MeV (Vautherin & Brink 1972).

<table><tr><td>Quantity</td><td>FSUGold</td><td>FSU-J0</td><td>FSU-δ6.7</td><td>FSU-δ6.2</td></tr><tr><td> $n_0$ (fm $^{-3}$ )</td><td>0.148</td><td>0.148</td><td>0.148</td><td>0.148</td></tr><tr><td> $e_0$ (MeV)</td><td>-16.31</td><td>-16.31</td><td>-16.31</td><td>-16.31</td></tr><tr><td> $K_0$ (MeV)</td><td>229.2</td><td>229.2</td><td>229.2</td><td>229.2</td></tr><tr><td> $J_0$ ( MeV)</td><td>-521.6</td><td>-322.0</td><td>-322.0</td><td>-322.0</td></tr><tr><td> $m^{*}_{Dirac}$ </td><td>0.610</td><td>0.610</td><td>0.610</td><td>0.610</td></tr><tr><td> $m^{*}_{Dirac,n,0.5}$ </td><td>0.610</td><td>0.610</td><td>0.576</td><td>0.578</td></tr><tr><td> $m^{*}_{Dirac,p,0.5}$ </td><td>0.610</td><td>0.610</td><td>0.643</td><td>0.640</td></tr><tr><td> $E_{sym,c}$ (MeV)</td><td>25.88</td><td>25.88</td><td>25.88</td><td>25.88</td></tr><tr><td> $L_c$ (MeV)</td><td>48.02</td><td>55.00</td><td>55.00</td><td>55.00</td></tr><tr><td> $E_{sym}$ (MeV)</td><td>32.59</td><td>33.45</td><td>32.75</td><td>32.53</td></tr><tr><td> $L$ (MeV)</td><td>60.50</td><td>68.14</td><td>53.50</td><td>48.21</td></tr><tr><td> $n_t$ (fm $^{-3}$ )</td><td>0.082</td><td>0.081</td><td>0.088</td><td>0.089</td></tr><tr><td> $n_{cen}$ (fm $^{-3}$ )</td><td>1.15</td><td>1.00</td><td>0.91</td><td>0.92</td></tr><tr><td> $R_{1.4}$ (km)</td><td>12.66</td><td>13.65</td><td>13.67</td><td>12.76</td></tr><tr><td> $\Lambda_{1.4}$ </td><td>401.0</td><td>576.0</td><td>578.5</td><td>420.0</td></tr><tr><td> $M_{\text{max}}$ ( $M_{\odot}$ )</td><td>1.72</td><td>1.93</td><td>2.05</td><td>2.10</td></tr><tr><td> $E_B^{208}$ (MeV)</td><td>-7.89</td><td>-7.94</td><td>-7.69</td><td>-7.71</td></tr><tr><td> $R_{\text{ch}}^{208}$ (fm)</td><td>5.54</td><td>5.55</td><td>5.55</td><td>5.54</td></tr><tr><td> $\Delta r_{\text{np}}^{208}$ (fm)</td><td>0.207</td><td>0.227</td><td>0.235</td><td>0.242</td></tr><tr><td> $\epsilon_{ls,2d\pi}^{208}$ (MeV)</td><td>1.60</td><td>1.57</td><td>1.55</td><td>1.55</td></tr><tr><td> $\epsilon_{ls,3p\nu}^{208}$ (MeV)</td><td>0.77</td><td>0.75</td><td>0.71</td><td>0.71</td></tr><tr><td> $\epsilon_{ls,2f\nu}^{208}$ (MeV)</td><td>2.06</td><td>1.98</td><td>1.96</td><td>1.95</td></tr><tr><td> $E_B^{48}$ (MeV)</td><td>-8.56</td><td>-8.61</td><td>-8.46</td><td>-8.47</td></tr><tr><td> $R_{\text{ch}}^{48}$ (fm)</td><td>3.46</td><td>3.49</td><td>3.49</td><td>3.49</td></tr><tr><td> $\Delta r_{\text{np}}^{48}$ (fm)</td><td>0.197</td><td>0.208</td><td>0.213</td><td>0.217</td></tr></table>

$E _ { \mathrm { s y m , c } } = 2 5$ .88 MeV and $L _ { c } = 5 5 ~ \mathrm { { M e V } } ,$ one can obtain the seven model parameters $g _ { \sigma } , g _ { \omega } , g _ { \rho } , b _ { \sigma } , c _ { \sigma } , c _ { \omega }$ and $\Lambda _ { V }$ , and the corresponding parameter set is denoted as ‘FSU-J0’ and shown in TABLE 1. Moreover, the nuclear matter bulk parameters and NS properties as well as the ground state properties of ${ } ^ { 2 0 8 } \mathrm { { P b } }$ and $^ { 4 8 } \mathrm { C a }$ with FSU-J0 are summarized in TABLE 2. As expected, one sees that compared to FSUGold, the FSU-J0 gives very similar ground state properties except a larger neutron skin thickness of $\Delta r _ { \mathrm { n p } } ^ { \mathrm { 2 0 8 } } = 0 . 2 2 7$ fm due to the larger $L _ { c }$ and thus it is compatible with the PREX-II measurement.

Now we turn to the Lagrangian density $\mathcal { L }$ given by Eq. (7), which additionally includes the $\delta$ meson and the $\delta { - } \sigma$ mixing compared with the FSUGold model and has nine model parameters, namely $g _ { \sigma } , g _ { \omega } , g _ { \rho } , g _ { \delta } , b _ { \sigma } , c _ { \sigma } , c _ { \omega } ,$ $\Lambda _ { V }$ and $C _ { \delta \sigma }$ with the $\delta$ meson mass fixed at 980.0 MeV. To explore the efects of the $\delta$ meson and the $\delta { - } \sigma$ mixing, our strategy is to vary the $g _ { \delta }$ and $C _ { \delta \sigma }$ while fix the values of $n _ { 0 } , e _ { 0 } , K _ { 0 } , J _ { 0 } , m _ { \mathrm { D i r a c } } ^ { * } , E _ { \mathrm { s y m , c } }$ and $L _ { c }$ at their corresponding values as in FSU-J0, and this guarantees that the constraints from PREX-II and SNM EOS from flow data can be satisfied as well as the basic properties of finite nuclei can be reasonably described. In such a way, we find that increasing the value of $g _ { \delta }$ can stifen the symmetry energy at suprasaturation densities and enhance the $M _ { \mathrm { m a x } }$ . However, the $g _ { \delta }$ cannot be too large since the $\Lambda _ { 1 . 4 }$ will also increase with $g _ { \delta }$ but it is limited by $\Lambda _ { 1 . 4 } \leq 5 8 0$ . On the other hand, it is interesting to note that the $C _ { \delta \sigma }$ can soften the symmetry energy at intermediate densities around $2 \sim 3 n _ { 0 }$ but stifen the symmetry energy at high densities, and this feature can decrease the $\Lambda _ { 1 . 4 }$ and increase the $M _ { \mathrm { m a x } }$ . As an example, we find the parameters $g _ { \delta } = 6 . 7$ and $C _ { \delta \sigma } = 0 . 0 3 8 5$ lead to $\Lambda _ { 1 . 4 } = 5 7 8 . 5$ and $M _ { \mathrm { m a x } } = 2 . 0 5 M _ { \odot }$ , and we denote this parameter set as $\mathrm { \cdot F S U } \mathrm { - } \delta 6 . 7 ^ { \mathrm { \cdot } }$ and the model parameters are listed in TABLE 1. The nuclear matter bulk parameters and NS properties as well as the ground state properties $\mathrm { o f ~ ^ { 2 0 8 } P b }$ and $^ { 4 8 } \mathrm { C a }$ with FSU-δ6.7 are summarized in TABLE 2. One sees that the FSU-δ6.7 predicts an isospin splitting of the nucleon Dirac mass with $m _ { \mathrm { D i r a c , p , 0 . 5 } } ^ { * } > m _ { \mathrm { D i r a c , n , 0 . 5 } } ^ { * }$ due to the inclusion of the δ meson, which is not observed in the FSUGold and FSU-J0. As for the ground state properties, the FSUδ6.7 reproduces the experimental data on binding energies at the $2 \%$ level and charge rms radii better than 1%, which is already satisfactory in the present work without the best-fit procedure. In addition, it is interesting to see that the inclusion of the δ meson seems to slightly improve the description of the spin-orbit splittings and obviously enhance the neutron skin thickness.

![](images/eff4b6f6d9092a191f886f712afc1d20170b88a6fc82273b274dc7b299749fe4.jpg)  
Figure 1. Pressure vs density (a) and density dependence of the symmetry energy (b) predicted by FSUGold, FSU-J0, FSU-δ6.7 and FSU-δ6.2. The shaded area in panel (a) represents the constraints from flow data in heavy-ion collisions (Danielewicz et al. 2002). The central density of maxi mum mass NS configuration is indicated by $^ \ast { } _ { \ast } ^ { \ast }$ in panel (b).

As a more interesting example, we find the parameters $g _ { \delta } = 6 . 2$ and $C _ { \delta \sigma } = 0 . 0 5 4$ lead to $\Lambda _ { 1 . 4 } = 4 2 0$ and $M _ { \mathrm { m a x } } = 2 . 1 M _ { \odot }$ . This parameter set is denoted as ‘FSU-$\delta 6 . 2 ^ { \cdot }$ and its model parameters are listed in TABLE 1. The nuclear matter bulk parameters and NS properties as well as the ground state properties of ${ } ^ { 2 0 8 } \mathrm { { P b } }$ and $^ { 4 8 } \mathrm { C a }$ with FSU-δ6.2 are also included in TABLE 2 for comparison. It is seen that the FSU-δ6.2 predicts almost the same isospin splitting of nucleon Dirac mass and nuclear ground state properties as those with the FSU-δ6.7 except a larger neutron skin thickness. The important feature of FSU-δ6.2 is that it gives a very soft symmetry energy around $2 \sim 3 n _ { 0 }$ due to the smaller $g _ { \delta }$ and larger $C _ { \delta \sigma }$ compared with FSU-δ6.7. It should be mentioned that a larger $C _ { \delta \sigma }$ value may lead to extremely soft symmetry energy around intermediate densities and cause the spinodal instability for the NS matter, i.e., the pressure decreases with the increment of density, and this case needs special treatment with the construction of two-phase system in NSs (Zabari et al. 2019a; Kubis et al. 2020).

To illustrate the above discussions and analyses, we show in Fig. 1 (a) the pressure vs density for SNM and (b) the $E _ { \mathrm { s y m } }$ with FSUGold, FSU-J0, FSU-δ6.7 and

![](images/beb6e873bd7b9c84df1c383be6b04d57ebd2d72a1768106e2bc38b02571434ab.jpg)  
Figure 2. M-R relation for static NSs from FSUGold, FSU-J0, FSU-δ6.7 and FSU-δ6.2. The NICER (XMM-Newton) constraints with 68% C.L for PSR J0030+0451 (Riley et al. 2019; Miller et al. 2019) and PSR J0740+6620 (Riley et al. 2021; Miller et al. 2021) are also included for comparison.

FSU-δ6.2. The constraint on pressure vs density from collective flow data in heavy-ion collisions (Danielewicz et al. 2002) is also included in Fig. 1 (a) for comparison. Indeed, one sees from Fig. 1 (a) that the results from all the four parameter sets FSUGold, FSU-J0, FSU-δ6.7 and FSU-δ6.2 are consistent with the constraint on the SNM EOS from collective flow data. In addition, it is seen from Fig. 1 (b) that compared with FSU-J0, the FSU-δ6.7 predicts a softer symmetry energy around $2 n _ { 0 }$ but much stifer symmetry energy at high densities due to the efects of δ meson and the $\delta { - } \sigma$ mixing, which leads to a larger $M _ { \mathrm { m a x } }$ for FSU-δ6.7 although both FSU-J0 and the FSU-δ6.7 have the same EOS of SNM. Furthermore, the FSU-δ6.2 predicts a very soft symmetry energy around $2 \sim 3 n _ { 0 }$ but a very stif symmetry energy at high densities where the symmetry energy increases rapidly with density (resulting in a larger sound speed in the NS matter), and this produces an even larger $M _ { \mathrm { m a x } } ~ ( \mathrm { i . e . , 2 . 1 } M _ { \odot } )$ compared with $\mathrm { F S U }  – \delta 6 . 7 .$ . The very soft symmetry energy around $2 \sim 3 n _ { 0 }$ with FSU-δ6.2 produces a much smaller $\Lambda _ { 1 . 4 } ~ ( \mathrm { i . e . , ~ 4 2 0 } )$ . These results clearly indicate that the $\delta$ meson and the $\delta { - } \sigma$ mixing can have very interesting and significant influence on the symmetry energy and the NS properties.

Finally, it is interesting to check the model predictions on the NS M-R relation since the simultaneous M-R determinations from NICER for PSR J0030+0451 (Riley et al. 2019; Miller et al. 2019) with mass around $1 . 4 M _ { \odot }$ and for PSR J0740+6620 (Riley et al. 2021; Miller et al. 2021) with mass around $2 . 0 M _ { \odot }$ have been obtained. Shown in Fig. 2 is the M-R relation for static NSs predicted from the four models FSUGold, FSU-J0, FSUδ6.7 and FSU-δ6.2. Also included in Fig. 2 are the two independent simultaneous M-R measurements from NICER by analyzing the X-ray data for the isolated 205.53 Hz millisecond pulsar PSR J0030+0451 (Riley et al. 2019; Miller et al. 2019) as well as the recent radius determination of PSR J0740+6620 from NICER and X-ray Multi-Mirror (XMM-Newton) X-ray observations (Riley et al. 2021; Miller et al. 2021). It is seen that the FSU-δ6.7 and FSU-δ6.2, which consider the δ meson and the δ-σ mixing, are in very good agreement with the astrophysical observations and measurements for both PSR J0030+0451 and PSR J0740+6620. On the other hand, the FSUGold and FSU-J0 within the original FSUGold framework fail to describe the constraints for PSR J0740+6620 due to the too small value of $M _ { \mathrm { m a x } }$ . At last, we would like to point out that the present work is based on the conventional NS model within a single unified framework without considering the possible appearance of new degrees of freedom such as hyperons, meson condensates, quark matter in NSs. It will be interesting to consider these new degrees of freedom in NSs within the new RMF model proposed in the present work.

## 4. SUMMARY

By including the isovector scalar δ meson and its coupling to isoscalar scalar σ meson in the accurately calibrated interaction FSUGold, we have demonstrated that the new RMF model can be simultaneously compatible with the constraints on the EOS of SNM at suprasaturation densities from flow data in heavy-ion collisions, the neutron skin thickness of <sup>208</sup>Pb from the PREX-II experiment, the measured largest NS mass reported so far from PSR J0740+6620, the limit of $\Lambda _ { 1 . 4 } \leq 5 8 0$ from GW170817, and the M-R relation of PSR J0030+0451 and PSR J0740+6620 measured by NICER, and thus remove the tension between PREX-II and GW170817 observed in the conventional RMF model.

The great success of the new RMF model is mainly due to the fact that the $\delta { - } \sigma$ mixing can soften the symmetry energy at intermediate densities but stifen the symmetry energy at high densities, which further leads to a larger $M _ { \mathrm { m a x } }$ but a smaller $\Lambda _ { 1 . 4 }$ . Further investigation of the $\delta { - } \sigma$ mixing efects on nuclear structure, heavy-ion collisions and neutron stars is an extremely interesting topic.

## ACKNOWLEDGEMENTS

## REFERENCES

Abbott, B. P., Abbott, R., Abbott, T. D., et al. 2018, PhRvL, 121, 161101, doi: 10.1103/PhysRevLett.121.161101

Adhikari, D., Albataineh, H., Androic, D., et al. 2021, PhRvL, 126, 172502, doi: 10.1103/PhysRevLett.126.172502

Angeli, I., & Marinova, K. P. 2013, Atom. Data Nucl. Data Tabl., 99, 69, doi: 10.1016/j.adt.2011.12.006

This work was supported by National SKA Program of China No. 2020SKA0120300 and the National Natural Science Foundation of China under Grant No. 11625521 and No. 11775049.

Baldo, M., & Burgio, G. F. 2016, Progress in Particle and Nuclear Physics, 91, 203, doi: 10.1016/j.ppnp.2016.06.006

Baran, V., Colonna, M., Greco, V., & Di Toro, M. 2005, PhR, 410, 335, doi: 10.1016/j.physrep.2004.12.004

Biswas, B. 2021, ApJ, 921, 63, doi: 10.3847/1538-4357/ac1c72

Brown, B. A. 2000, PhRvL, 85, 5296, doi: 10.1103/PhysRevLett.85.5296

Boguta, J., & Bodmer, A. R. 1977, Nucl. Phys. A, 292, 413, doi: 10.1016/0375-9474(77)90626-1

Cai, B.-J., & Chen, L.-W. 2012a, Physics Letters B, 711, 104, doi: 10.1016/j.physletb.2012.03.058

—. 2012b, PhRvC, 85, 024302, doi: 10.1103/PhysRevC.85.024302

—. 2017, Nucl. Sci. Tech., 28, 185, doi: 10.1007/s41365-017-0329-1

Carriere, J., Horowitz, C. J., & Piekarewicz, J. 2003, ApJ, 593, 463, doi: 10.1086/376515

Centelles, M., Roca-Maza, X., Vi˜nas, X., & Warda, M. 2009, PhRvL, 102, 122502, doi: 10.1103/PhysRevLett.102.122502

Chen, L.-W., Ko, C. M., & Li, B.-A. 2005, PhRvC, 72, 064309, doi: 10.1103/PhysRevC.72.064309 —. 2007, PhRvC, 76, 054316, doi: 10.1103/PhysRevC.76.054316

Chen, L.-W., Ko, C. M., Li, B.-A., & Xu, J. 2010, PhRvC, 82, 024321, doi: 10.1103/PhysRevC.82.024321

Cromartie, H. T., Fonseca, E., Ransom, S. M., et al. 2020, Nature Astronomy, 4, 72, doi: 10.1038/s41550-019-0880-2

Danielewicz, P., Lacey, R., & Lynch, W. G. 2002, Science, 298, 1592, doi: 10.1126/science.1078070

Fattoyev, F. J., Horowitz, C. J., Piekarewicz, J., & Shen, G. 2010, PhRvC, 82, 055803, doi: 10.1103/PhysRevC.82.055803

Fonseca, E., Cromartie, H. T., Pennucci, T. T., et al. 2021, ApJL, 915, L12, doi: 10.3847/2041-8213/ac03b8

Furnstahl, R. J. 2002, NuPhA, 706, 85, doi: 10.1016/S0375-9474(02)00867-9

Gandolfi, S., Gezerlis, A., & Carlson, J. 2015, Annual Review of Nuclear and Particle Science, 65, 303, doi: 10.1146/annurev-nucl-102014-021957

Hofmann, F., Keil, C. M., & Lenske, H. 2001, PhRvC, 64, 034314, doi: 10.1103/PhysRevC.64.034314

Horowitz, C. J., & Piekarewicz, J. 2001a, PhRvL, 86, 5647, doi: 10.1103/PhysRevLett.86.5647

Kubis, S. 2007, PhRvC, 76, 025801, doi: 10.1103/PhysRevC.76.025801

Kubis, S., & Kutschera, M. 1997, Physics Letters B, 399, 191, doi: 10.1016/S0370-2693(97)00306-7

Kubis, S., W´ojcik, W., & Zabari, N. 2020, PhRvC, 102, 065803, doi: 10.1103/PhysRevC.102.065803

Li, B.-A., Chen, L.-W., & Ko, C. M. 2008, PhR, 464, 113, doi: 10.1016/j.physrep.2008.04.005

Liu, B., Greco, V., Baran, V., Colonna, M., & di Toro, M. 2002, PhRvC, 65, 045201, doi: 10.1103/PhysRevC.65.045201

Meng, J., Toki, H., Zhou, S. G., et al. 2006, Progress in Particle and Nuclear Physics, 57, 470, doi: 10.1016/j.ppnp.2005.06.001

Miller, M. C., Lamb, F. K., Dittmann, A. J., et al. 2019, ApJL, 887, L24, doi: 10.3847/2041-8213/ab50c5

—. 2021, ApJL, 918, L28, doi: 10.3847/2041-8213/ac089b

Miyatsu, T., Cheoun, M.-K., & Saito, K. 2022, arXiv e-prints, arXiv:2202.06468. https://arxiv.org/abs/2202.06468

M¨uller, H., & Serot, B. D. 1996, NuPhA, 606, 508, doi: 10.1016/0375-9474(96)00187-X

Ozel, F., & Freire, P. 2016, ARA&A, 54, 401,<sup>¨</sup> doi: 10.1146/annurev-astro-081915-023322

Reed, B. T., Fattoyev, F. J., Horowitz, C. J., & Piekarewicz, J. 2021, PhRvL, 126, 172503, doi: 10.1103/PhysRevLett.126.172503

Riley, T. E., Watts, A. L., Bogdanov, S., et al. 2019, ApJL, 887, L21, doi: 10.3847/2041-8213/ab481c

Riley, T. E., Watts, A. L., Ray, P. S., et al. 2021, ApJL, 918, L27, doi: 10.3847/2041-8213/ac0a81

Roca-Maza, X., Centelles, M., Vi˜nas, X., & Warda, M. 2011, PhRvL, 106, 252501, doi: 10.1103/PhysRevLett.106.252501

Serot, B. D. 1979, Physics Letters B, 86, 146, doi: 10.1016/0370-2693(79)90804-9

Serot, B. D., & Walecka, J. D. 1986, Adv. Nucl. Phys., 16, 1 —. 1997, International Journal of Modern Physics E, 6, 515, doi: 10.1142/S0218301397000299

Steiner, A. W., Prakash, M., Lattimer, J. M., & Ellis, P. J. 2005, PhR, 411, 325, doi: 10.1016/j.physrep.2005.02.004

Todd-Rutel, B. G., & Piekarewicz, J. 2005, PhRvL, 95, 122501, doi: 10.1103/PhysRevLett.95.122501

Vautherin, D., & Brink, D. M. 1972, PhRvC, 5, 626, doi: 10.1103/PhysRevC.5.626

Walecka, J. D. 1974, Annals Phys., 83, 491, doi: 10.1016/0003-4916(74)90208-5

Wang, M., Huang, W. J., Kondev, F. G., Audi, G., & Naimi, S. 2021, Chin. Phys. C, 45, 030003, doi: 10.1088/1674-1137/abddaf

Xu, J., Chen, L.-W., Li, B.-A., & Ma, H.-R. 2009, ApJ, 697, 1549, doi: 10.1088/0004-637X/697/2/1549

Yue, T.-G., Chen, L.-W., Zhang, Z., & Zhou, Y. 2021, arXiv e-prints, arXiv:2102.05267. https://arxiv.org/abs/2102.05267

Zabari, N., Kubis, S., & W´ojcik, W. 2019a, PhRvC, 99, 035209, doi: 10.1103/PhysRevC.99.035209

—. 2019b, PhRvC, 100, 015808, doi: 10.1103/PhysRevC.100.015808

Zhang, N.-B., & Li, B.-A. 2019, European Physical Journal A, 55, 39, doi: 10.1140/epja/i2019-12700-0

Zhang, Z., & Chen, L.-W. 2013, Physics Letters B, 726, 234, doi: 10.1016/j.physletb.2013.08.002 —. 2016, PhRvC, 94, 064326, doi: 10.1103/PhysRevC.94.064326

Zhou, Y., & Chen, L.-W. 2019, ApJ, 886, 52,doi: 10.3847/1538-4357/ab4adf

Zhou, Y., Chen, L.-W., & Zhang, Z. 2019, PhRvD, 99, 121301, doi: 10.1103/PhysRevD.99.121301