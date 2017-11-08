---
title: Full-Waveform Inversion - Part 2``:`` adjoint modeling
author: |
	Mathias Louboutin^1^\*, Philipp Witte^1^, Michael Lange^2^, Navjot Kukreja^2^, Fabio Luporini^2^, Gerard Gorman^2^, and Felix J. Herrmann^1,3^\
	^1^ Seismic Laboratory for Imaging and Modeling (SLIM), The University of British Columbia \
	^2^ Imperial College London, London, UK\
	^3^ now at Georgia Institute of Technology, USA \
bibliography:
	- bib_tuto.bib
---


## Introduction

This tutorial is the second part of a three part tutorial series on full-waveform inversion (FWI), in which we provide a step by step walkthrough of setting up forward and adjoint wave equation solvers and an optimization framework for inversion. In part 1 [@louboutin2017fwi], we demonstrated how to discretize the acoustic wave equation and how to set up a basic forward modeling scheme using Devito, a domain-specific language (DSL) in Python for automated finite-difference (FD) computations [@lange2016dtg]. Devito allows to define wave equations as symbolic Python expressions [@Meurer17], from which optimized FD stencil code is automatically generated during run time. In part 1, we show how we can use Devito to set up and solve acoustic wave equations with (impulsive) seismic sources and sample wavefields at the receiver locations to model shot records.

In the second part of this tutorial series, we will discuss how to set up and solve adjoint wave equations with Devito and from that, how we can calculate gradients and function values of the FWI objective function. The gradient of FWI is most commonly computed via the adjoint state method, by cross-correlating forward and adjoint wavefields and summing the contributions over all time steps [refer to @Plessix2006 in the context of seismic inversion]. Calculating the gradient (for one source location) therefore consists of three steps:

* Solving a forward wave equation to model a shot record and save the forward wavefields.

* Computing the data residual between the predicted and observed data.

* Solving an adjoint wave equation with the data residual as the adjoint source. In the adjoint (reverse) time loop, cross correlate the second time derivative of the forward wavefield with the adjoint wavefield and sum over time.


## The adjoint wave equation

Conceptually, adjoint wave equations seem like a very theoretical construct and understanding why we need adjoint wave equations and what they are is not always intuitive. To get a better understanding of adjoint wave equations, we will revisit the forward simulation from the first part, but look at them from the linear algebra point of view. In fact, we can think of solving a forward wave equation as solving a linear system of equations

```math {#linWE}
    \mathbf{A}(\mathbf{m}) \mathbf{u} = \mathbf{q}, 
```

where the solution of this system is obtained by inverting the matrix $\mathbf{A}(\mathbf{m})$, i.e. $\mathbf{u} = \mathbf{A}(\mathbf{m})^{-1} \mathbf{q}$. The vector $\mathbf{q}$ is the seismic source, $\mathbf{u}$ are the vectorized wavefields for **all** time steps and $\mathbf{A}(\mathbf{m})$ represents the discretized acoustic wave equation and consists of finite difference operators for spatial and temporal derivates, as well as the model $\mathbf{m}$ and damping terms. From the linear algebra and optimization point of view, solving wave equations therefore simply corresponds to multiplying the matrix $\mathbf{F} := \mathbf{A}(\mathbf{m})^{-1}$ with a vector $\mathbf{q}$. For calculating the gradient of (non-linear) least squares problems such as FWI, we also need the action of the adjoint matrix $\mathbf{F}^\top$, which corresponds to solving the linear system 

```math {#adjWE}
    \mathbf{A}(\mathbf{m})^\top \mathbf{v} = \delta \mathbf{d},
```

with $\mathbf{v}$ being the adjoint wave field and $\delta \mathbf{d}$ being the adjoint source (the adjoint source could really be anything, but we will see that the adjoint source for FWI is typically the data residual $\delta \mathbf{d}$). If we explicitly formed the full matrix $\mathbf{A}(\mathbf{m})$, we would see that $\mathbf{A}(\mathbf{m})$ is lower triangular, which means the only non-zero entries lie along or below the diagonal. However, due to its size ($\mathbf{A}(\mathbf{m})$ represents the full time-stepping scheme), the matrix is never explicitly formed and instead, we solve wave equations by successively updating the wavefields at each time step using the (two) previous wavefields. This process implicitly corresponds to inverting  $\mathbf{A}(\mathbf{m})$ via forward substitution, i.e. we invert the matrix time step by time step, starting at the top row of $\mathbf{A}(\mathbf{m})$ (the first time step). For solving the adjoint wave equation, we need to invert the transposed matrix, which accordingly is upper triangular, with all non-zero entries lying along or above the diagonal. Linear systems with upper triangular matrices are solved by backward substitution, which means we invert the matrix that represents the adjoint wave equation by starting at the bottom row (the last time step) and by moving backwards in time (hence the expression reverse-time).

To implement the adjoint wave equation with Devito, we once again start from the PDE in continuous form. With the variables defined as in part 1 and the data residual $\delta d(x,y,t; x_r, y_r)$, located at $x_r, y_r$, as the adjoint source, the adjoint wave equation is given by:

```math {#WEa}
 m \frac{d^2 v(t,x,y)}{dt^2} - \nabla^2 v(t,x,y) - \eta(x,y) \frac{d v(t,x,y)}{dt}=\delta d(t,x,y;x_r, y_r)
```

Since the acoustic wave equation contains only second spatial and temporal derivatives, which are both self-adjoint, the adjoint acoustic wave equation is equivalent to the forward equation with the exception of the damping term $\eta(x,y) \frac{d v(t,x,y)}{dt}$, which contains a first derivative and therefore has a change of sign. (A second derivative matrix is the same as its transpose, whereas a first derivative matrix is equal to its negative transpose and vice versa.)

Following part 1, we first define the discrete adjoint wave field $\mathbf{v}$ as a Devito `TimeFunction` object and then symbolically set up the PDE and rearrange the expression:

```python
	# Discrete adjoint wavefield
	v = TimeFunction(name="v", grid=model.grid, time_order=2, space_order=2)
	
	# Define adjoint wave equation and rearrange expression
	pde = model.m * v.dt2 - v.laplace - model.damp * v.dt
	stencil_v = Eq(v.backward, solve(pde, v.backward)[0])
```

Just as for the forward wave equation, `stencil_v` defines the update for the adjoint wavefield of a single time step within the (reverse) time loop and corresponds to the expression:

```math {#WEdisadj}
\mathbf{v}[\text{time}-dt] = 2\mathbf{v}[\text{time}] - \mathbf{v}[\text{time}+dt] + \frac{dt^2}{\mathbf{m}}\Delta \mathbf{v}[\text{time}],
```

with dt being the time stepping interval. Once again, this expression does not contain any (adjoint) source terms so far, which will be defined as a separate `SparseFunction` object. Since the source term for the adjoint wave equation is the difference between an observed and modeled shot record, we first define an (empty) shot record `rec` with 101 receivers and coordinates defined in `rec_coords`. We then set the data field `rec.data` of our shot record to be the data residual between the observed data `d_obs` and the predicted data `d_pred`. The symbolic source expression `src_term` for our adjoint wave equation is then obtained by *injecting* the data residual into the modeling scheme (`rec.inject`). Since we solve the time-stepping loop backwards in time, the `src_term` is used to update the previous adjoint wavefield `v.backward`, rather than the next wavefield. As in the forward modeling example, the source is scaled by  $\frac{dt^2}{\mathbf{m}}$.

```python
	# Set up data residual as adjoint source
	rec = Receiver(name='rec', npoint=101, ntime=nt, grid=model.grid, coordinates=rec_coords)
	rec.data = d_pred - d_obs
	src_term = rec.inject(field=v.backward,  expr=rec * dt**2 / model.m, offset=model.nbpml)
```	

Finally, we create the full propagator by adding the source expression to our previously defined stencil and set the flag `time_axis=Backward`, to specify that the propagator runs in backwards in time:

```python
	# Create propagator
	op_adj = Operator([stencil_v] + src_term, time_axis=Backward)
```

In contrast to forward modeling, we do not need to define a (receiver) sampling operator, since we are only intersted in the adjoint wavefield itself. The full script for setting up the adjoint wave equation, including an animation of the adjoint wavefield is available in **`adjoint_modeling.ipynb`**.

## Computing the FWI gradient

With propagators to solve forward and adjoint wave equations, we now have all parts in place to compute function values and gradients of the FWI objective function. The goal of FWI is to estimate a discrete parametrization of the subsurface by minimizing the misfit between the observed shot records of a seismic survey and numerically modeled shot records. The predicted shot records are obtained by solving an individual wave equation per shot location and depend on the parametrization $\mathbf{m}$ of our wave propagator. The most common function for measuring the data misfit between the observed and modeled data is the $\ell_2$-norm, which leads to the following objective function [@LionsJL1971,@Tarantola]:

```math {#FWI}
	\mathop{\hbox{minimize}}_{\mathbf{m}} \hspace{.2cm} \Phi(\mathbf{m})= \sum_{i=1}^{n_s} \frac{1}{2} \left\lVert \mathbf{d}^\mathrm{pred}_i (\mathbf{m}, \mathbf{q}) - \mathbf{d}_i^\mathrm{obs} \right\rVert_2^2,
```

where the index $i$ runs over the total number of shots $n_s$ and the model parameters are the squared slowness. Optimization problems of this form are called non-linear least-squares problems, since the predicted data $\mathbf{d}_i^\mathrm{pred} = \mathbf{A}(\mathbf{m})^{-1}\mathbf{q}_i$ depends on the unknown parameters $\mathbf{m}$ non-linearly. The full derivation of the FWI gradient using the adjoint state method is outside the scope of this tutorial, but conceptually we obtain the gradient by applying the chain rule and taking the partial derivative of the inverse wave equation $\mathbf{A}(\mathbf{m})^{-1}$ with respect to $\mathbf{m}$, which yields the following expression [@Plessix2006; @Virieux]:

```math {#FWIgrad}
 \nabla \Phi (\mathbf{m})= -  \sum_{i=1}^{n_s}  \sum_{time=1}^{n_t} \ddot{\mathbf{u}}[\text{time}]\odot \mathbf{v}[\text{time}].
```

The inner sum $\text{time}=1,...,n_t$ runs over the number of computational time steps $n_t$ and $\ddot{\mathbf{u}}$ denotes the second temporal derivative of the forward wavefield $\mathbf{u}$. Computing the gradient of Equation #FWI, therefore corresponds to performing the point-wise multiplication $\odot$ of the adjoint wavefields with the second time derivative of the forward wavefield and summing over all time steps and source positions. 

In practice, the FWI gradient (for a single source) is calculated in the reverse time-loop while solving the adjoint wave equation. I.e., while updating the adjoint wavefield for the current time step $\mathbf{v}[\text{time}]$, we compute:

```math {#gradupd}
 \mathbf{g} = \mathbf{g} - \frac{\mathbf{u}[\text{time-dt}] - 2\mathbf{u}[\text{time}] + \mathbf{u}[\text{time+dt}]}{dt^2} \odot \mathbf{v}[\text{time}]
```

The second time derivative of the forward wavefield is computed with a second order finite-difference stencil and requires access to the forward wavefields of the previous, current and subsequent time step. To implement the FWI gradient with Devito, we first define the gradient as a dense data object `Function`, which has no time dependence, since the gradient is computed as the sum over all time steps. The update for the gradient as defined in Equations #FWIgrad and #gradupd is then implemented as the following symbolic expression:

```python
grad = Function(name="g", grid=model.grid)
grad_update = Eq(grad, grad - u.dt2 * v)
``` 

Since the gradient is calculated as part of the adjoint time loop, we add the expression for the gradient to our previously defined stencil for the adjoint propagator:

```python
	op_grad = Operator([stencil_v] + src_term + grad_update,
						time_axis=Backward)
```

Solving the adjoint wave equation by running `op_grad(time=nt, dt=model.critical_dt)` from the Python command line now includes computing the FWI gradient, which afterwards can be accessed with `op_grad.gradient`. To further demonstrate the gradient computation, we perform a small seismic transmission experiment with the Camembert model, a constant velocity model with a circular high velocity inclusion in its center. For a transmission experiment, we place 21 seismic sources on the left-hand side of the model and 101 receivers on the right-hand side. We then use the forward propagator from part 1 to independently model the 21 "observed" shot records using the true model. As the initial model for our gradient calculation, we use a constant velocity model with the same velocity as the true model, but without the circular velocity perturbation. We then model the 21 predicted shot records for the initial model, calculate the data residual and gradient for each shot and then sum all 21 gradients to obtain the full gradient (Figure #Gradient). This result can be reproduced with the notebook **`adjoint_gradient.ipynb`**.

**show true model and gradient. add sources and receivers to plots (with src/rec labels) and colorbar with colorbar labels**

####Figure: {#Gradient}
![](Figures/simplegrad.pdf){width=50%}
: Camembert velocity model and the FWI gradient for 21 source locations, where each shot is recorded by 101 receivers located on the right-hand side of the model. The initial model used to compute the predicted data and gradient is a constant velocity model with the background velocity of the true model. This result can be reproduced by running the script **`adjoint_gradient.ipynb`**.

The final step of the adjoint modeling and gradient part is unit testing, i.e. we ensure that the adjoints and gradients are implemented correctly. Incorrect adjoints can lead to unpredictable behaviour during and inversion and in the worst case cause slower convergence or convergence to wrong solutions. Since our forward-adjoint wave equation solvers implicitly correspond to forward-adjoint matrix-vector products $\mathbf{F} \mathbf{q}$ and $\mathbf{F}^\top\mathbf{\delta d}$, we need to ensure that the relationship $\delta \mathbf{d}^\top \mathbf{F} \mathbf{q} = \mathbf{q}^\top\mathbf{F}^\top\mathbf{\delta d}$ holds within machine precision (see **`tests/test_adjointA.py`** for the full adjoint test). Furthermore, we verify the correct implementation of the FWI gradient by ensuring that using the gradient leads to first order convergence. The gradient test can be found in **`tests/test_gradient.py`**.

## Conclusions

The gradient of the FWI objective function is computed by solving adjoint wave equations and summing the point-wise product of forward and adjoint wavefields over all time steps. Using Devito, the adjoint wave equation is set up in a similar fashion as the forward wave equation, with the main difference being the (adjoint) source, which is the residual between the observed and predicted shot records. The FWI gradient is computed as part of the adjoint time loop and implemented by adding its symbolic expression to the stencil for the adjoint propagator. With the ability to model shot records and compute gradients of the FWI objective function, we will demonstrate how to set up a full inversion framework in the next and final part of this tutorial series. 

## Installation

This tutorial is based on Devito version 3.1.0. It requires the installation of the full software with examples, not only the code generation API. To install Devito, run

	git clone -b v3.1.0 https://github.com/opesci/devito
	cd devito
	conda env create -f environment.yml
	source activate devito
	pip install -e .
 
### Useful links

- [Devito documentation](http://www.opesci.org/)
- [Devito source code and examples](https://github.com/opesci/Devito)
- [Tutorial notebooks with latest Devito/master](https://github.com/opesci/Devito/examples/seismic/tutorials)


## Acknowledgments

This research was carried out as part of the SINBAD II project with the support of the member organizations of the SINBAD Consortium. This work was financially supported in part by EPSRC grant EP/L000407/1 and the Imperial College London Intel Parallel Computing Centre.

## References


