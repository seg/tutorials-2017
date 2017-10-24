---
title: Full-Waveform Inversion - Part 1``:`` forward and adjoint modelling
runninghead : Part 1, modelling for inversion
author: |
	Mathias Louboutin^1^\*, Philipp Witte^1^, Michael Lange^2^, Felix J. Herrmann^1^, Navjot Kurjeka^2^, Fabio Luporini^2^ and Gerard Gorman^2^ \
	^1^ Seismic Laboratory for Imaging and Modeling (SLIM), The University of British Columbia \
	^2^ Imperial College London, London, UK
bibliography:
	- bib_tuto.bib
---

## Introduction

Full-waveform inversion (FWI) gained tremendous attention in geophysical exploration since it was first introduced [@Pratt]. However, the literature mostly contains specific and technical papers about applications and advanced method and lacks simple introductory resources for geophysical newcomers. Mathematical and geophysical FWI papers, as @Virieux, give excellent theoretical overviews, but typically do not cover the implementation side of the problem. In this two part tutorial, we provide a hands-on walkthrough of FWI using Devito, a finite-difference domain-specific language that provides a concise and straightforward interface for discretizing wave equations and generating operators for forward and adjoint modeling. Devito allows the user to concentrate on the geophysical side of the problem, rather than the low-level implementation details of a wave-equation simulator. This tutorial covers the conventional adjoint-state FWI formulation. Other method exist to improve the convergence properties of the algorithm, but most still rely on forward/adjoint pairs an correlation-based gradient and should use the proposed framework.

FWI relies on two main components:

 - A modelling operator to simulate synthetic data that can be compared to field recorded data
 - An adjoint operator to back-propagate the data residual and compute the cross-correlation update direction.

We will illustrate the workflow on a very simplistic 2D model that can be run on a laptop or desktop PC. Larger and more realistic models come at a computational and memory price that are beyond of the scope of this tutorial, but the workflow we describe here translates to velocity models of any size (in 2D and 3D) and any type of wave equations with known adjoints. The workflow for full-waveform inversion, and the corresponding notebooks, is the following:

 - Simulate synthetic data and save the corresponding wavefield (**modelling.ipynb**).
 - Compute the data residual, difference between the synthetic and observed data (**adjoint_gradient.ipynb**).
 - Back-propagate the residual (adjoint solve with the residual as a source term, **adjoint_gradient.ipynb**).
 - Compute the cross-correlation FWI gradient over time (**adjoint_gradient.ipynb**).
 - Repeat for all sources, at every iteration of the optimization algorithm in (**adjoint_gradient.ipynb**).
 
We start with the description of a modelling operator and then move on to the adjoint operator and gradient for FWI. A complete tutorial on how to optimize the FWI objective function, once the computational framework is in place, will be covered in the second part. This tutorials is linked to three notebooks that detail each step of the implementation from modelling to FWI. For clarity purposes, some details will be left out of the article but are fully detailed in the corresponding notebooks.

## Modelling

The acoustic wave equation for the squared slowness ``m``, defined as ``m=\frac{1}{c^2}`` with ``c`` being the speed of sound, and a source ``q`` is given by:

$$
\label{WE}
 m \frac{d^2 u(x,t)}{dt^2} - \nabla^2 u(x,t) + \eta \frac{d u(x,t)}{dt}&=q  \ \text{in } \Omega \\
 u(.,0) &= 0 \\
 \frac{d u(x,t)}{dt}|_{t=0} &= 0
$$

with the zero initial conditions to guarantee unicity of the solution.
The boundary conditions are Dirichlet conditions :
$$
 u(x,t)|_\delta\Omega = 0
$$

where ``\delta\Omega`` is the surface of the boundary of the model ``\Omega``.

In the field, seismic wave propagate in all directions in an "infinite" medium. However, solving the wave equation in a mathematically/discrete infinite domain is not feasible. Therefore, Absorbing Boundary Conditions (ABC) or Perfectly Matched Layers (PML) are used in practice to mimic an infinite domain. The two methods allow to approximate an infinite medium by damping and absorbing the waves at the limit of the domain to avoid boundary reflections.

The simplest of these methods is the absorbing damping mask. The core idea is to extend the physical domain and to add a sponge mask in this extension that will absorb the incident waves. In our case, we use ABC where ``\eta`` is the damping mask equal to ``0`` inside the physical domain and increasing inside the sponge layer. Multiple choice of profile can be chosen for ``\eta`` from linear to exponential [@Cerjan].

We discretize the wave equation with Devito, a finite-difference DSL that solves the discretized wave-equation on a cartesian grid. The finite-difference approximation is derived from Taylor expansions of the continuous field after removing the error term [@lange2016dtg].

### Discretization

The first step is to define a symbolic representation of the discrete wavefield. In Devtio, this is represented by a ```TimeData``` object and contains all information for the discretization such a discretization orders in space and time:

```python
	u = TimeData(name="u", shape=model.shape_domain, time_order=2, space_order=2, save=True, time_dim=nt)
```

In this tutorial we use a second order time discretization, which is the most commonly used time discretization. From the Taylor expansion of the continuous wavefield ``u`` in time, the second order discrete approximation of the second order time derivative as a function of the discrete wavefield ``\mathbf{u}`` is:

$$\label{timedis}
 \frac{d^2 u(x,t)}{dt^2} = \frac{\mathbf{u}(x, t+\Delta t) - 2 \mathbf{u}(x, t) + \mathbf{u}(x, t-\Delta t)}{\Delta t^2} + O(\Delta t^2).
$$

and the finite-difference approximation is ``\frac{d^2\mathbf{u}(x,t)}{\Delta t^2} =   \frac{\mathbf{u}(x, t+\Delta t) - 2 \mathbf{u}(x, t) + \mathbf{u}(x, t-\Delta t)}{\Delta t^2}`` where ``\mathbf{u}`` is the discrete wavefield, ``\Delta t`` is the discrete time-step (distance between two consecutive discrete times) and ``O(\Delta t^2)`` is the discretization error term. The discretized approximation of the second order time derivative is then given by dropping the error term. This derivative is represented in Devito by ```u.dt2```.

Apart from the temporal derivative, the acoustic wave equation contains spatial (second) derivatives. We therefore define the discrete Laplacian ``\Delta \mathbf{u}(x,y,z,t)`` as the sum of the second order spatial derivatives in the three dimensions. Each second spatial derivative is discretized with a ``k^{th}`` order finite-difference scheme  (**space_order=k** in the ```TimeData``` object creation) also derived from Taylor expansion. The Laplacian is represented in Devito by ```u.laplace``` and follow the same theoretical derivation as in Equation #timedis applied to the space variables ``x,y,z``.

With the space and time discretization defined, we can fully discretize the wave-equation with the combination of the temporal and spatial discretizations and obtain the following second order in time and ``k^{th}`` order in space discrete stencil to update one grid point at position ``x,y,z`` at time ``t``:

$$
\label{WEdis}
\mathbf{u}(x,y,z,t+\Delta t) = &2\mathbf{u}(x,y,z,t) - \mathbf{u}(x,y,z,t-\Delta t) +\\
& \frac{\mathbf{\Delta t}^2}{\mathbf{m}(x,y,z)} \Big(\Delta \mathbf{u}(x,y,z,t) \Big). 
$$

Using Devito, we can directly translate the discretized wave equation into a symbolic expression and define a $$stencil$$ expression, which defines the update for the new wavefield at each time step:

```python
	# Set up acoustic wave equation
	pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
	stencil = Eq(u.forward, solve(pde, u.forward)[0])
```

This wave equation does not contain a source term and is solely defined through its initial conditions(#WE). To simulate a seismic experiment, we still need to define a seismic source, i.e. a seismic wavelet which is injected into the model at a defined source location, as well as the receiver sampling operator, i.e. locations in the model at which we record the modeled wavefield as a function of time.

The source injection and receiver sampling operators are typically localized in space and are therefore defined as ```PointData``` objects in Devito. Since we cannot generally assume that we only want to inject or sample data at grid points, but also in between, the ```PointData``` objects contains methods for interpolation between the computational grid and the source/receiver coordinates:

```python
	# create receiver sampling operator from receiver coordinates
	rec = Receiver(name='rec', npoint=101, ntime=nt, ndim=2, coordinates=rec_coords)
	rec_term = rec.interpolate(u, offset=model.nbpml)
	
	# define source injection operator for given wavelet, coordinates and frequency
	src = RickerSource(name='src', ndim=2, f0=f0, time=time, coordinates=src_coords)
	src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m, offset=model.nbpml)
```

With the source/receiver projection operators, we can define our full modeling operator by symbolically adding the source and receiver term to our previously defined stencil:

```python
	# Create operator
	op = Operator([stencil] + src_term + rec_term,
	              subs={t.spacing: dt, x.spacing: spacing[0],
	                    y.spacing: spacing[1]})
```

The forward wavefield and shot record are then modelled with a simple call to ```op.apply()```. We show the shot record in Figure #Forward and a movie of the forward wavefield can be found in the last cell of the noteboook **modelling.ipynb**.

####Figure: {#Forward}
![Shot record](Figures/shotrecord.pdf){width=45%}
: Shot record on a two layer model for a single source and split-spread receiver geometry from **modelling.ipynb**.


## Inversion operators

Full-waveform inversion aims at recovering an accurate model of the discrete wave velocity, ``\mathbf{c}``, or equivalently the squared slowness ``\mathbf{m} = \mathbf{c}^{-2}``, from a given set of measurements of the pressure wavefield ``\mathbf{u}``. This can be expressed as the following optimization problem [@LionsJL1971, @Virieux, @haber10TRemp]:

$$
\label{FWI}
	\mathop{\hbox{minimize}}_{\mathbf{m}} \Phi(\mathbf{m})&=\frac{1}{2}\left\lVert\mathbf{P}_r
	\mathbf{A}(\mathbf{m})^{-1} \mathbf{P}_s^T \mathbf{q} - \mathbf{d}\right\rVert_2^2 \\
$$

Using the chain rule, the gradient of the FWI objective function ``\Phi(\mathbf{m})`` with respect to the squared slowness ``\mathbf{m}``  is then given by:

$$\label{FWIgradLA}
 \nabla\Phi_s(\mathbf{m})= - \left(\frac{d \mathbf{A}(\mathbf{m}) \mathbf{u}}{d\mathbf{m}}\right)^T \mathbf{A}(\mathbf{m})^{-T} \mathbf{P}_r^T \delta \mathbf{d} =\mathbf{J}^T\delta\mathbf{d},
$$

which is equivalent to summing (over time) the zero-lag cross-correlation of the forward and adjoint wavefields:

$$\label{FWIgrad}
 \nabla\Phi_s(\mathbf{m})= - \sum_{\mathbf{t} =1}^{n_t}\mathbf{u}_{tt}[\mathbf{t}] \mathbf{v}[\mathbf{t}].
$$

The partial derivative of the modeling operator ``\frac{d \mathbf{A}(\mathbf{m}) \mathbf{u}}{d\mathbf{m}}`` in Equation #FWIgradLA is simply the second time derivative, since ``\mathbf{m}`` appears only in front of this term (equation #WEdis). The parameter ``n_t`` is the number of computational time steps, ``\delta\mathbf{d} = \left(\mathbf{P}_r \mathbf{u} - \mathbf{d} \right)`` is the data residual (difference between the measured data and the modelled data), ``\mathbf{J}`` is the Jacobian (i.e. the linearized modeling or demigration operator) and ``\mathbf{u}_{tt}`` is the second-order time derivative of the forward wavefield solving #linWE\.


### Implementation

As explained in the introduction and shown in Equation #FWIgradLA, FWI with the adjoint state method is based on back-propagation. As we can see in Equation #FWIgradLA, the gradient expression contains the term ``\mathbf{A}(\mathbf{m})^{-T}``, which is the adjoint (inverse) wave equation. To understand what the adjoint wave equation is, it is helpful to go back to the forward modelling part and rewrite our forward wave equation as a linear system:

$$\label{linWE}
    \mathbf{A}(\mathbf{m}) \mathbf{u} = \mathbf{P}_s^T \mathbf{q}, 
$$

The adjoint system is defined accordingly, by transposing the linear system and with the data residual as the adjoint source:

$$\label{adjWE}
    \mathbf{A}(\mathbf{m})^T \mathbf{v} = \mathbf{P}_r^T \delta \mathbf{d}
$$

Solving the wave-equation is equivalent to solving the linear system ``\mathbf{Au}=\mathbf{q}`` where the vector ``\mathbf{u}`` is the discrete wavefield solution of the discrete wave-equation, ``\mathbf{q}`` is the source term and ``\mathbf{A}`` is the matrix representation of the discrete wave-equation. From Equation #WEdis we can see that the matrix ``\mathbf{A}`` is a lower triangular matrix that reflects the time-marching structure of the stencil. Simulation of the wavefield is equivalent to a forward elimination on the lower triangular matrix ``\mathbf{A}``. The adjoint of ``\mathbf{A}``, denoted as ``\mathbf{A}^T``, is then an upper triangular matrix and the solution ``\mathbf{v}`` of the discrete adjoint wave-equation ``\mathbf{A}^\top\mathbf{v}=\mathbf{q}_a`` for an adjoint source ``\mathbf{q}_a`` is equivalent to a backward elimination on the upper triangular matrix ``\mathbf{A}\top`` and is simulated backward in time starting from the last time-step. These matrices are never explicitly formed, but are instead matrix free operators with implicit implementation of ``\mathbf{u}=\mathbf{A}^{-1}\mathbf{q}``.

The implementation of the adjoint modelling is straightforward in the self-adjoint acoustic case. The only detail to consider is to adjust the non self-adjoint boundary conditions, which corresponds simply to a change of sign. Using Devito, we can define the adjoint wave equation in the same fashion as the forward equation: 

```python
	# Receiver setup
	rec = Receiver(name='rec', npoint=101, ntime=nt, ndim=2, coordinates=rec_coords)
	rec_term = rec.inject(field=u.forward,  expr=rec * dt**2 / model.m, offset=model.nbpml)
	
	# Source setup
	srca = PointSource(name='src', ndim=2, coordinates=src_coords, npoint=1)
	src_term = srca.interpolate(u, offset=model.nbpml)
	
	# Define adjoint wave equation
	pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
	stencil = Eq(u.forward, solve(pde, u.forward)[0])
	
	# Create operator
	op = Operator([stencil] + src_term + rec_term,
	              subs={t.spacing: dt, x.spacing: spacing[0],
	                    y.spacing: spacing[1]},
						time_axis=Backward)
```

For calculating the gradient of the FWI objective function, we then need to simply sum the pointwise multiplications of the adjoint wavefields with the second time derivative of the foward wavefield. In Devito this is symbolically expressed as ```grad_update = Eq(grad, grad - u.dt2 * v)```. The full script for calculating the gradient is given in the notebook **adjoint_gradient.ipynb**. 

Before we take a look at what the gradient for our test model looks like, we want to ensure that our implementations of the forward and adjoint wave equations are in fact a correct forward-adjoint pair. Not having correct adjoints leads to wrong gradients, which in turn lead to convergence to wrong solutions, or the convergence rate is slown down. To ensure that the actions of the wave equation operators are implemented correctly, the matrix-free linear operators need to pass the **dot test**, which can be found in the Devito test **tests/test_adjointA.py**.

Having tested the forward/adjoint wave equations, we can now calculate the gradient of the FWI objective function for a simple 2D test model. The Camembert model consists of a constant medium with a circular high velocity zone in its centre and we perform a transmission experiment, with the source on one side of the model and receivers at the other side. The gradient for a constant starting model (without the circular perturbation) looks as follows:

####Figure: {#Gradient}
![Gradient for a transmission camembert model and a single source-receiver pair](Figures/banana.pdf){width=45%}
![Gradient for a transmission camembert model and a sfull shot record](Figures/simplegrad.pdf){width=45%}
:Gradients for a simple camembert transmission model **adjoint_gradient.ipynb**.

Finally, with the gradient implemented, we can easily setup the FWI objective function that can be used in an optimization toolbox as we will show in the next part of the tutorial. A example of an FWI objective function is given in cell 18 of **adjoint_gradient.ipynb**.


## Conclusion

The first part of the tutorial demonstrated how to set up forward and adjoint wave equations and calculate the gradient of the FWI objective function with the adjoint state method. In the second part we will demonstrate how to set up an easy optimization framework that allows to solve FWI with gradient-based optimization algorithms.

### Installation

This tutorial and the coming second part are based on Devito version 3.0.3. It also require to install the full software with examples, not only the code generation API. To install devito

```
	git clone https://github.com/opesci/devito/tree/v3.0.3
	cd devito
	conda env create -f environment.yml
	source activate devito
	pip install -e .
```

 
### Usefull links

- [Devito documentation](http://www.opesci.org/)
- [Devito source code and examples](https://github.com/opesci/Devito)
- [Tutorial notebooks with latest Devito/master](https://github.com/opesci/Devito/examples/seismic/tutorials)

##references

[1] Cerjan, C., Kosloff, D., Kosloff, R., and Reshef, M., 1985, A nonreflecting boundary condition for discrete acoustic and elastic wave equations: GEOPHYSICS, 50, 705–708. doi:10.1190/1.1441945

[2] Haber, E., Chung, M., and Herrmann, F. J., 2012, An effective method for parameter estimation with PDE constraints with multiple right hand sides: SIAM Journal on Optimization, 22. Retrieved from http://dx.doi.org/10.1137/11081126X

[3] Lange, M., Kukreja, N., Louboutin, M., Luporini, F., Zacarias, F. V., Pandolfo, V., … Gorman, G., 2016, Devito: Towards a generic finite difference DSL using symbolic python: 6th workshop on python for high-performance and scientific computing. doi:10.1109/PyHPC.2016.9

[4] Lions, J. L., 1971, Optimal control of systems governed by partial differential equations: (1st ed.). Springer-Verlag Berlin Heidelberg.

[5] Pratt, R. G., 1999, Seismic waveform inversion in the frequency domain, part 1: Theory and verification in a physical scale model: GEOPHYSICS, 64, 888–901. doi:10.1190/1.1444597

[6] Virieux, J., and Operto, S., 2009, An overview of full-waveform inversion in exploration geophysics: GEOPHYSICS, 74, WCC1–WCC26. doi:10.1190/1.3238367