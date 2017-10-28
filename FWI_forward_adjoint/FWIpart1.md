---
title: Full-Waveform Inversion - Part 1``:`` forward modeling
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

Since its re-introduction by @Pratt, Full-waveform inversion (FWI) has gained a lot of attention in geophysical exploration because of its ability to build high resolution velocity models more or less automatically in areas of complex geology. While there is an extensive and growing literature on this topic, publications focus mostly on technical aspects, making this topic inaccessible for a broader audience due to the lack of simple introductory resources for geophysical newcomers. This is part one of two tutorials that attempt to provide an introduction and software to help people getting started. We hope to accomplish this by providing a hands-on walkthrough of FWI using Devito [@lange2016dtg], a system based on domain specific languages that automatically generates code for time-domain finite differences. In this capacity, Devito provides a concise and straightforward computational framework for discretizing wave equations, which underlie all FWI frameworks. We will show that it generates verifiable executable code at run time for wave propagators associated with forward and adjoint wave equations. Devito releases the user from recurrent and time-consuming coding of performant time-stepping codes and allows to concentrate on the geophysics of the problem rather than on low-level implementation details of wave-equation simulators. This tutorial covers the conventional adjoint-state formulation of full-waveform tomography [@Tomo] that underlies most of the current methods referred to as full-waveform inversion. While other formulations have been developed to improve the convergence properties of FWI, we will concentrate on the standard formulation that relies on the combination of a forward/adjoint pair of propagators and a correlation-based gradient.

Full-waveform inversion tries to iteratively minimize the difference between data that was acquired in a seismic survey and synthetic data that is generated from a wave simulator with an estimated model of the subsurface. As such, each FWI framework essentially consists of a wave simulator for forward modeling the predicted data and an adjoint simulator for calculating a model update from the data misfit. The first part of this tutorial is dedicated to the forward modeling part and demonstrates how to discretize and implement the acoustic wave equation using Devito. This tutorial is accompanied by a Jupyter notebook - **`forward_modeling.ipynb`** -, in which we describe how to simulate synthetic data for a specified source and receiver setup and how to save the corresponding wavefields and shot records. In part two of this series, we will address how to calculate model updates, i.e. gradients of the FWI objective function, via adjoint modeling and part three will demonstrate how to use the gradient as part of an optimization framework for inverting an unknown velocity model. 

Installation instructions for Devito are detailed at the end of the paper and required to execute the notebook.

## Wave simulations for inversion

The acoustic wave equation with the squared slowness ``m``, defined as ``m(x,y)=c^{-2}(x,y)`` with ``c(x,y)`` being the unknown spatially varying wavespeed, is given by:

```math {#WE}
 m \frac{d^2 u(x,y,t)}{dt^2} - \nabla^2 u(x,y,t) + \eta(x,y) \frac{d u(x,y,t)}{dt}=q(x,y,t;x_s, y_s),
```
where ``q(x,y,t;x_s)`` is the seismic source, located at ``(x_s,y_s)`` and ``\eta(x,y)`` is a space-dependent dampening parameter for the absorbing boundary layer [@Cerjan]. As shown in Figure #model, the physical model is extended in every direction by `nbpml` grid points to mimic an infinite domain. The dampening term ``\eta \frac{d u(x,t)}{dt}`` attenuates the waves in the dampening layer [@Cerjan] and prevents waves to reflect at the model boundaries. In Devito, the discrete representations of  ``m`` and ``\eta`` are contained in a `model` object that contains all relevant information such as the origin of the coordinate system, grid spacing and size of the model---i.e., in `Python` we have

```python
	# Define a Devito object with the velocity model and grid information
	model = Model(vp=vp, origin=origin, shape=shape, spacing=spacing, nbpml=40)
```

In the `Model` instantiation, `vp` is the velocity in ``\text{km}/\text{s}``, `origin` is the origin of the physical model in meters, `spacing` is the discrete grid spacing in meters, `shape` is the number of grid points in each dimension and `nbpml` is the number of grid points in the absorbing boundary layer. Is is important to note that `shape` is the size of the physical domain only, while the total number of grid points, including the absorbing boundary layer, will be automatically derived from `shape` and `nbpml`.

####Figure: {#model}
![Model](Figures/setup.png){width=50%}
: Representation of the computational domain and its extension, which contains the absorbing boundaries layer.


### Symbolic definition of the wave propagator

To model seismic data by solving the acoustic wave equation, the first necessary step is to discretize our PDE, which includes discrete representations of the velocity model and wavefields, as well as approximations of the spatial and  temporal derivatives using finite differences (FD). However, instead of writing out  long finite difference stencils by hand, we are going to employ the powerful symbolic representations of Devito.

The primary design objective of Devito is to allow users to define complex matrix-free finite difference operators from high-level symbolic definitions, while employing automated code generation to create highly optimized low-level C code. For this purpose Devito uses the symbolic algebra package SymPy [@Meurer17] to facilitate the automatic creation of derivative expressions, allowing the quick and efficient generation of high-order wave propagators with variable stencil orders.

At the core of Devito's symbolic API are two symbolic types that behave like SymPy function objects, while also managing user data:

* `DenseData` objects represent a spatially varying function discretized on a regular cartesian grid. For example, a function symbol `f = DenseData(name='f', shape=(nx, ny), space_order=2)` is denoted symbolically as `f(x, y)`. Auto-generated symbolic expressions for finite difference derivatives are provided by these objects via shorthand expressions, enabling the syntax `f.dx` to denote $\frac{\partial f}{\partial x}$ and `f.dx2` for $\frac{\partial^2 f}{\partial x^2}$.

* `TimeData` objects represent a time-dependent function that includes leading dimension $t$, for example `g(t, x, y)`. In addition to spatial derivatives `TimeData` symbols also provide time derivatives `g.dt` and `g.dt2`, as well as options to save the entire data along the time axis.

To demonstrate Devito's symbolic capabilities, let us consider a time-dependent function $\vd{u}(t, x, y)$ representing the discrete forward wavefield. We can define this as a `TimeData` object in Devito:

```python
    u = TimeData(name="u", shape=model.shape_domain, time_order=2,
                 space_order=2, save=True, time_dim=nt)
```

where the parameter `shape` defines the size of the allocated memory region, `time_order` and `space_order` define the default discretization order of the derivative expressions and the parameters `save=True` and `time_dim` force the entire wavefield to be stored in memory.

We can now use this wavefield to generate simple discretized stencil expressions for finite difference derivatives as:

```python
  In []: u
  Out[]: u(t, x, y)

  In []: u.dt
  Out[]: -u(t, x, y)/s + u(t+s, x, y)/s

  In []: u.dt2
  Out[]: -2*u(t, x, y)/s**2 + u(t-s, x, y)/s**2 + u(t+s, x, y)/s**2
```

Using the automatic derivation of derivative expressions we can now implement a discretized expression for Equation #WE\ without the source term $q(x,y,t;x_s, y_s)$. The `model` object which we created earlier, already contains the squared slowness $\vd{m}$ and damping term $\vd{\eta}$ as `DenseData` objects:

```python
    # Set up discretized wave equation
    pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
```

If we write out the (second order) second time derivative `u.dt2` as shown above and ignore the damping term for the moment, out `pde` expression translates to the following discrete the wave equation:

```math {#WEdis}
 \frac{\vd{m}}{s^2} \Big( \vd{u}[\text{time}-s] - 2\vd{u}[\text{time}] + \vd{u}[\text{time}+s]\Big) - \Delta \vd{u}[\text{time}] = 0, \quad \text{time}=1 \cdots n_{t-1}
```

with time being the current time step and $s$ being the time stepping interval. As we can see, the Laplacian $\Delta \vd{u}$ is simply expressed with Devito by the shorthand expression `u.laplace`, where the order of the derivative stencil is defined by the `space_order` parameter used to create the symbol `u(t, x, y)`.  However, for solving the wave equation, Equation #WEdis needs to be rearranged so that we obtain an expression for the wavefield $\vd{u}(\text{time}+s)$ at the next time step. Ignoring the damping term once again, this yields:

```math {#WEstencil}
 \vd{u}[\text{time}+s] = 2\vd{u}[\text{time}] - \vd{u}[\text{time}-s] + \frac{s^2}{\vd{m}} \Delta \vd{u}[\text{time}]
```

In Python, we can rearrange our `pde` expression automatically using the SymPy utility function `solve`, to create a stencil expression that defines the update of the wavefield for the new time step $\vd{u}(\text{time}+s)$, which is expressed as `u.forward` in Devito:

```python
    # Rearrange PDE to obtain new wavefield at next time step
    stencil = Eq(u.forward, solve(pde, u.forward)[0])
```

This `stencil` expression now represents the finite difference scheme from Equation #WEstencil, including the FD approximation of the Laplacian and the damping term. The `stencil` expression defines the update for a single time step only, but since the wavefield `u` is a `TimeData` object, Devito knows that we are solving a time-dependent problem over a number of time steps.


### Setting up the acquisition geometry

The expression we derived in the previous section does not contain any seismic source function yet, so the update for the wavefield at a new time step is solely defined by the two previous wavefields. However as indicated in Equation #WE, wavefields for seismic experiments are excited by an artificial source ``q(x,y,t;x_s)``, which is a function of space and time (just like the wavefield `u`). To include a source term in our modeling scheme, we simply add the the source field as an additional term to our stencil expression (Equation #WEstencil):

```math {#WEdisa}
 \vd{u}[\text{time}+s] = 2\vd{u}[\text{time}] - \vd{u}[\text{time}-s] + \frac{s^2}{\vd{m}} \Big(\Delta \vd{u}[\text{time}] + \vd{q}[\text{time}]\Big).
```

Since the source appears on the right-hand side in the original equation (Equation #WE), the term also needs to be multiplied with ``\frac{s^2}{\vd{m}}`` (this follows from rearranging expression #WEdis, with the source on the right-hand side). Unlike the discrete wavefield `u` however, the source `q` is typically localized in space and only a function of time, which means the time-dependent source wavelet is injected into the wavefield at a specified source location. The same applies for sampling the wavefield at the receiver locations for obtaining a shot record, i.e. the wavefield is sampled at specified receiver locations only and those locations do not necessarily coincide with the modeling grid.

Since both sources and receivers are sparse in the spatial domain and potentially need to be interpolated onto the model grid, Devito provides a separate symbolic type specifically for sparse objects called `PointData`. Just like we defined wavefields as `TimeData` objects and model and damping terms as `DenseData` objects, we can construct `PointData` objects for sources and receiver and add them to our `stencil` expression.

Devito provides a special function for setting up a Ricker wavelet called `RickerSource`, which acts as a wrapper around `PointData` objects and automatically creates an instance of a `PointData` object for a Ricker wavelet with a specified peak frequency `f0` and source coordinates `src_coords`:

```	
	# define source object with Ricker wavelet and inject
	src = RickerSource(name='src', ndim=2, f0=f0, time=time, coordinates=src_coords)
	src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m, offset=model.nbpml)
```

The `src.inject` now injects the current time sample of the Ricker wavelet (weighted with ``\frac{s^2}{\vd{m}}`` as shown in equation #WEdisa) into the updated wavefield `u.forward` at the specified coordinates. The parameter `offset` is the size of the absorbing layer as shown in Figure #model (i.e. the source position is shifted by `offset`).

There is an according wrapper function for receivers as well, which creates a `PointData` object for a given number `npoint` of receivers, number `nt` of time samples  and specified receiver coordinates `rec_coords` (with `ndim=2`, since we have a two-dimensional example). 

```python
	# create receiver array from receiver coordinates
	rec = Receiver(name='rec', npoint=101, ntime=nt, ndim=2, coordinates=rec_coords)
	rec_term = rec.interpolate(u, offset=model.nbpml)
```

Rather than injecting a function into the model as we did for the source, we now simply save the wavefield at the grid points that correspond to receiver positions and interpolate the data to their exact location (`rec.interpolate` in Devito).

### Forward simulation 

Having defined `PointData` objects for sources and receivers, we can now define our forward propagator by adding the source and receiver terms to our stencil object; that is, in `Python` we have:

```python
	# Create forward propagator
	op_fwd = Operator([stencil] + src_term + rec_term,
	              subs={t.spacing: dt, x.spacing: spacing[0],
	                    y.spacing: spacing[1]})
```

Up to this point, our full expression `[stencil] + src_term + rec_term` only contains placeholders for the modeling parameters, such as number of grid points and sampling intervals. Only now, when we want to generate the actual code, those placeholders need to be filled in with concrete values. The `subs` keyword argument takes those modeling parameters as an input and, at code generation time, substitutes the placeholders with the parameters. Once the modeling operator is created, we can access the generated C code with `op_fw.ccode`. There is no need to explicitly define time loops in Python, since Devito infers the modeling time from the `TimeData` objects directly and automatically generates those time loops in the C code.

####Figure:{#Cgen}
![Generated C code](Figures/ccode-crop.png)

We can finally execute the forward modeling propagator with the simple command:

```python	
	# Generate wavefield snapshots and a shot record
	op_fwd.apply()
```

Once the propagator executed, we obtain the modeled wavefield and shot record from the symbolic objects. Each Devito symbolic types has a `data` field that contains the result.

```
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

In this first part of the tutorial, we have demonstrated how to set up discretized forward wave equations, their associated propagators with at runtime code generation. In the following part, we will show how to calculate a valid gradient of the FWI objective using the adjoint state method. In part three, we will demonstrate how to set up a complete matrix-free and scalable optimization framework for acoustic FWI.

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