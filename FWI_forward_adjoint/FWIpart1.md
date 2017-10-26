---
title: Full-Waveform Inversion - Part 1``:`` forward and adjoint modeling
runninghead : Part 1, modeling for inversion
author: |
	Mathias Louboutin^1^\*, Philipp Witte^1^, Michael Lange^2^, Navjot Kurjeka^2^, Fabio Luporini^2^, Gerard Gorman^2^, and Felix J. Herrmann^1,3^\
	^1^ Seismic Laboratory for Imaging and Modeling (SLIM), The University of British Columbia \
	^2^ Imperial College London, London, UK\
	^3^ now at Georgia Institute of Technology, USA \
bibliography:
	- bib_tuto.bib
---


## Introduction

Since its re-introduction by @Pratt Full-waveform inversion (FWI) has gained a lot of attention in geophysical exploration because of its ability to build velocity models more or less automatically in areas of complex geology. While there is an extensive and growing literature on this topic, the publications focus mostly on technical aspect making this topic inaccessible since it lacks simple introductory resources for geophysical newcomers. This is part one of two tutorials that attempt to provide an introduction and software to help people getting started. We hope to accomplish this by providing a hands-on walkthrough of FWI with Devito, which is a compiler for a domain-specific language (DSL) that automatically generates code for time-domain finite differences. In this capacity, Devito provides a concise and straightforward computational framework for discretizing wave equations. We will show that it generates verifiable executable code at run time for wave propagators associated with the forward and adjoint wave equation. Devito [@lange2016dtg] releases the user from recurrent and time-consuming coding of performant time-stepping codes and allows the user to concentrate on the geophysics of the problem rather than on low-level implementation details of wave-equation simulators. This tutorial covers the conventional adjoint-state formulation of full-waveform tomography [@Tomo] that underlies most of the current methods referred to as full-waveform inversion. While other formulations have been developed to improve the convergence properties of FWI, we will concentrate on the standard formulation that relies on the combination of a forward/adjoint pair of propagators and a correlation-based gradient.

As part of this tutorial for FWI, we first introduce propagators that simulate forward modeled synthetic data, which will be compared with field recorded data for inversion in the following tutorials. This tutorial is accompanied by a jupyter notebook:
- **`forward_modeling.ipynb`** --- in this notebook, we describe how to simulate synthetic data and how to save the corresponding wavefields and shot records for a given source and receiver geometry

## Wave simulations for inversion

The acoustic wave equation for the squared slowness ``m``, defined as ``m(x,y)=c^{-2}(x,y)`` with ``c(x,y)`` being the unknown spatially varying wavespeed, and ``q(x,y,t;x_s)`` a source located at ``(x_s,y_s)``, is given by:

```math {#WE}
 m \frac{d^2 u(x,y,t)}{dt^2} - \nabla^2 u(x,y,t) + \eta(x,y) \frac{d u(x,y,t)}{dt}=q(x,y,t;x_s, y_s).
```

Here, ``\eta(x,y)`` is a space-dependent dampening parameter for the absorbing boundary layer [@Cerjan]. As shown in Figure #model, the physical model is extended in every direction by `nbpml` grid points to mimic an infinite domain. The dampening term ``\eta \frac{d u(x,t)}{dt}`` attenuates the waves in the dampening layer [@Cerjan]. In Devito, the physical parameters ``m`` and ``\eta`` are contained in the `model` object that contains all relevant information such as the origin, grid spacing and size of the model---i.e., in `Python` we have

```python
	# Define a Devito model with physical size, velocity vp 
	# and absorbing layer width in number of grid points (nbpml)
	model = Model(vp=vp, origin=origin, shape=shape, spacing=spacing, nbpml=40)
```

In the `Model` instantiation, `vp` is the velocity in ``\text{km}/\text{s}``, origin is the origin of the physical model in meters, spacing is the discrete grid spacing in meters and shape is the number of grid points in each directions. Is is really important to not that `shape` is the size of the physical domain, the total size including the absorbing boundary layer will be automatically derived from `shape` and `nbpml`.

####Figure: {#model}
![Model](Figures/setup.png){width=50%}
: Representation of the computational domain and its extension, which contains the absorbing boundaries layer.


### Symbolic definition of the wave propagator

The primary design objective of Devito is to allow users to define
complex matrix-free finite difference operators from high-level
symbolic definitions, while employing automated code generation to
create highly optimized low-level C code. For this purpose Devito uses
the symbolic algebra package SymPy @Meurer17 to facilitate the
automatic creation of derivative expressions, allowing the quick and
efficient generation of high-order wave propagators with variable
stencil orders.

At the core of Devito's symbolic API are two symbolic types that
behave like a `sympy.Function` objects, while also managing
user data:

* `DenseData` objects represent a spatially varying function
  discretized on a regular cartesian grid. For example, a function
  symbol `f = DenseData(name='f', shape=(nx, ny), space_order=2)`
  is denoted symbolically as `f(x, y)`. Auto-generated symbolic
  expressions for finite difference derivatives are provided by these
  objects via shorthand expressions, enabling the syntax `f.dx` to
  denote $\frac{\partial f}{\partial x}$ and `f.dx2` for
  $\frac{\partial^2 f}{\partial x^2}$.

* `TimeData` objects represent a time-dependent function
  that includes leading dimension $t$, for example `g(t, x, y)`. In
  addition to spatial derivatives `TimeData` symbols also provide
  time derivatives `g.dt` and `g.dt2`, as well as options to save
  the entire data along the time axis.

To demonstrate Devito's symbolic capabilities, let us consider a
time-dependent function $\vd{u}(t, x, y)$ representing the forward
wavefield. We can define this as a `TimeData` object in Devito as

```python
    u = TimeData(name="u", shape=model.shape_domain, time_order=2,
                 space_order=2, save=True, time_dim=nt)
```

where the parameter `shape` defines the size of the allocated memory
region, `time_order` and `space_order` define the default
discretization order of the derivative expressions and the parameters
`save=True` and `time_dim` force the entire wavefield to be stored in
memory.

We can now use this wavefield to generate simple discretized stencil
expressions for finite difference derivatives as:

```python
  In []: u
  Out[]: u(t, x, y)

  In []: u.dt
  Out[]: -u(t, x, y)/s + u(t+s, x, y)/s

  In []: u.dt2
  Out[]: -2*u(t, x, y)/s**2 + u(t-s, x, y)/s**2 + u(t+s, x, y)/s**2
```

Using the automatic derivation of derivative expressions we can now
implement a discretized expression for Equation #WE\ without the
source term $q(x,y,t;x_s, y_s)$, and using `DenseData` objects for
$\vd{m}(x, y)$ and $\vd{\eta}(x, y)$ provided by the `Model` utility simply as

```python
    # Set up discretized wave equation
    pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
```

The shorthand expression `u.laplace` hereby denotes the Laplacian
$\Delta \vd{u}$, where the order of the resulting derivative stencil is
defined by the `space_order` parameter used to create the symbol
`u(t, x, y, z)`.

The resulting expression, however, needs to be rearranged
to update the forward stencil point $\vd{u}(t+s, x, y)$, represented by
the shorthand expression `u.forward`. For this we can use the SymPy
utility function `solve` to create a stencil expression that defines
the update of the wavefield $\vd{u}$ during a single timestep.

```python
    # Generation of the stencil
    stencil = Eq(u.forward, solve(pde, u.forward)[0])
	print(latex(stencil))
```

that produces the sympy expression:

```math
u{\left (\text{time} + s,x,y \right )} = \frac{1}{6 h_{x}^{2} h_{y}^{2} \left(s \text{damp}{\left (x,y \right )} + 2 m{\left (x,y \right )}\right)} \left(6 h_{x}^{2} h_{y}^{2} s \text{damp}{\left (x,y \right )} u{\left (\text{time} - s,x,y \right )} + 24 h_{x}^{2} h_{y}^{2} m{\left (x,y \right )} u{\left (\text{time},x,y \right )} - 12 h_{x}^{2} h_{y}^{2} m{\left (x,y \right )} u{\left (\text{time} - s,x,y \right )} - 30 h_{x}^{2} s^{2} u{\left (\text{time},x,y \right )} - h_{x}^{2} s^{2} u{\left (\text{time},x,y - 2 h_{y} \right )} + 16 h_{x}^{2} s^{2} u{\left (\text{time},x,y - h_{y} \right )} + 16 h_{x}^{2} s^{2} u{\left (\text{time},x,y + h_{y} \right )} - h_{x}^{2} s^{2} u{\left (\text{time},x,y + 2 h_{y} \right )} - 30 h_{y}^{2} s^{2} u{\left (\text{time},x,y \right )} - h_{y}^{2} s^{2} u{\left (\text{time},x - 2 h_{x},y \right )} + 16 h_{y}^{2} s^{2} u{\left (\text{time},x - h_{x},y \right )} + 16 h_{y}^{2} s^{2} u{\left (\text{time},x + h_{x},y \right )} - h_{y}^{2} s^{2} u{\left (\text{time},x + 2 h_{x},y \right )}\right)
```

Mathematically, as we detail step by step in the notebook, it is equivalent to

```math {#WEdis}
 \vd{u}[\text{time}+s] = 2\vd{u}[\text{time}] - \vd{u}[\text{time}-s] + \frac{s^2}{\vd{m}} \Big(\Delta \vd{u}[\text{time}]\Big), \quad \text{time}=1 \cdots  n_t-1_,
```

ignoring the boundary (``\eta =  0`` inside the physical domain) for simplicity.

### Setting up the acquisition geometry

In Devito, we model monopole sources/receivers with the object `PointData`, which includes methods that interpolate between the computational grid on which the wave equation is discretized and possibly off-the-grid source/receiver locations.
We showed previously the stencil obtained without a source. In presence of a source the stencil is

```math {#WEdisa}
 \vd{u}[\text{time}+s] = 2\vd{u}[\text{time}] - \vd{u}[\text{time}-s] + \frac{s^2}{\vd{m}} \Big(\Delta \vd{u}[\text{time}] + \vd{q}[\text{time}]\Big), \quad \text{time}=1 \cdots  n_t-1_.
```


and we see that at time step `i` we need to add the source term corresponding to this time-step into the updated wavefield `u[i+1]` (`u.forward` in Devito) with the discretization weight  ``\frac{\Delta t^2}{\vd{m}}`` as the source is inside the physical domain (``\eta = 0``).

The code that implements the definition of the receiver and sources, with locations collected in the arrays `rec_coords` and `src_coords`, reads in `Python` as 

```	
	# define source injection array for given a source wavelet, coordinates and frequency
	src = RickerSource(name='src', ndim=2, f0=f0, time=time, coordinates=src_coords)
	src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m, offset=model.nbpml)
```

The parameter `offset` is the size of the absorbing layer as shown in Figure #model (source position shifted by `offset`).

On the other side, the receiver is only a read of the wavefield at a specific position at time `i` and does not require any weight.

```python
	# create receiver array from receiver coordinates
	rec = Receiver(name='rec', npoint=101, ntime=nt, ndim=2, coordinates=rec_coords)
	rec_term = rec.interpolate(u, offset=model.nbpml)
```

and the `offset` parameter also correct for the origin shift from the model extension.

### Forward simulation 

With the source/receiver geometry set and the wave-equation stencil generated, we can now define our forward propagator by symbolically adding the source and receiver terms into our previously defined `stencil` object---i.e., in `Python` we have

```python
	# Create forward propagator
	op_fwd = Operator([stencil] + src_term + rec_term,
	              subs={t.spacing: dt, x.spacing: spacing[0],
	                    y.spacing: spacing[1]})
```

The Devito operator creation is minimalist thanks to the symbolic representation of the stencil. All information such as dimension sizes (to generate loops), are contained in the stencil and more specifically in its arguments `u, m, dam, src, rec` that still carry the metadata provided at object instantiation (`TimeData, DenseData, ...` creation). The only extra argument is `subs` that provides substitution method for the constants. At generation time, all instances of `x.spacing=h_x` will be replaced by its actual value. Once the operator created, we can access the generated C code with `op_fw.ccode`.

####Figure:{#Cgen}
![Generated C code](Figures/ccode-crop.png)

We can finally execute the forward modeling propagator with the simple command:

```python	
	# Generate wavefield snapshots and a shot record
	op_fwd.apply()
	# Access the wavefield and shot record at the end of the propagation.
	wavefield = u.data
	shotrecord = rec.data
```

In Figure #Forward, we show the resulting shot record. A movie of snapshots of the forward wavefield can be generated by executing the last cell of **`forward_modeling.ipynb`**.

####Figure: {#Forward}
![Two layer shot record](Figures/shotrecord.pdf){width=45%}
![Marmousi shot record](Figures/shotrecord_marmou.pdf){width=45%}
: Shot record on a two layer and marmousimodel for a single source and split-spread receiver geometry from **`modeling.ipynb`**.

####Figure: {#Snaps}
![T=..33s](Figures/snap1.pdf){width=30%}
![T=.5s](Figures/snap2.pdf){width=30%}
![T=..67](Figures/snap3.pdf){width=30%}
: Snapshots of the wavefield in a two layer model for a source in the middle of the x axis **`modeling.ipynb`**.

## Conclusions

In this first part of the tutorial, we demonstrated how to set up discretized forward  wave equations, their associated propagators with at runtime code generation. In the follwoing part, we will show how to calculate a valid gradient of the FWI objective using the adjoint state method. In part three, we will demonstrate how to set up a complete matrix-free and scalable optimization framework for acoustic FWI.

### Installation

This tutorial and the coming second part are based on Devito version 3.0.3. It also require to install the full software with examples, not only the code generation API. To install devito

```
	git clone -b v3.0.3 https://github.com/opesci/devito
	cd devito
	conda env create -f environment.yml
	source activate devito
	pip install -e .
```
 
### Usefull links

- [Devito documentation](http://www.opesci.org/)
- [Devito source code and examples](https://github.com/opesci/Devito)
- [Tutorial notebooks with latest Devito/master](https://github.com/opesci/Devito/examples/seismic/tutorials)


### Need to add acknow
##references

``` math_def
\def\argmin{\mathop{\rm arg\,min}}
\def\vec{\mbox{``\mathrm{vec}``}}
\def\ivec{\mbox{``\mathrm{vec}^{-1}``}}
\newcommand{\m}{{\mathsf{m}}}
\newcommand{\PsDO}{\mbox{PsDO\,}}
\newcommand{\Id}{\mbox{``\tensor{I}\,``}}
\newcommand{\R}{\mbox{``\mathbb{R}``}}
\newcommand{\Z}{\mbox{``\mathbb{Z}``}}
\newcommand{\DE}{:=}
\newcommand{\Order}{\mbox{``{\cal O}``}} \def\bindex#1{{\mathcal{#1}}}
\def\pector#1{\mathrm{\mathbf{#1}}} 
\def\cector#1{#1} 
\def\censor#1{#1} 
\def\vd#1{\mathbf{#1}}
\def\fvector#1{{\widehat{\vd{#1}}}}
\def\evector#1{{\widetilde{\vd{#1}}}}
\def\pvector#1{{\breve{\vd{#1}}}}
\def\pector#1{\mathrm{#1}}
\def\ctensor#1{\bm{\mathcal{#1}}}
\def\tensorm#1{\bm{#1}}
\def\tensor#1{\vd{#1}}
\def\hensor#1{\tensor{#1}}
\def\uensor#1{\underline{\bm{#1}}}
\def\hector#1{\vd{#1}}
\def\ftensor#1{{\widehat{\tensor{#1}}}}
\def\calsor#1{{\boldsymbol{\mathcal{#1}}}}
\def\optensor#1{{\boldsymbol{\mathcal{#1}}}}
\def\hvector#1{\hat{\boldsymbol{\mathbf{#1}}}}
\def\minim{\mathop{\hbox{minimize}}}
\newcommand{\norm}[1]{\left\lVert#1\right\rVert_2^2}
\newcommand{\overbar}[1]{\mkern 1.5mu\overline{\mkern-1.5mu#1\mkern-1.5mu}\mkern 1.5mu}
```