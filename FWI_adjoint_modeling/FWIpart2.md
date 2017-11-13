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

This tutorial is the second part of a three part tutorial series on full-waveform inversion (FWI), in which we provide a step by step walkthrough of setting up forward and adjoint wave-equation solvers and an optimization framework for inversion. In part 1 [@louboutin2017fwi], we demonstrated how to discretize the acoustic wave-equation and how to set up a basic forward modeling scheme using [Devito], a domain-specific language (DSL) in Python for automated finite-difference (FD) computations [@lange2016dtg]. [Devito] allows us to define wave-equations as symbolic Python expressions [@Meurer17], from which optimized FD stencil code is automatically generated during run time. In part 1, we show how we can use [Devito] to set up and solve acoustic wave-equations with (impulsive) seismic sources and sample wavefields at the receiver locations to model shot records.

[Devito]:http://www.opesci.org/devito-public

In the second part of this tutorial series, we will discuss how to set up and solve adjoint wave-equations with [Devito] and from that, how we can calculate gradients and function values of the FWI objective function. The gradient of FWI is most commonly computed via the adjoint state method, by cross-correlating forward and adjoint wavefields and summing the contributions over all time steps (refer to @Plessix2006 in the context of seismic inversion). Calculating the gradient (for one source location) therefore consists of three steps:

* Solving a forward wave-equation to model a shot record and save the forward wavefields for each source.

* Computing the data residual between the predicted and observed data.

* Solving an adjoint wave-equation with the data residual as the adjoint source. In the adjoint (reverse) time loop, cross correlate the second time derivative of the forward wavefield with the adjoint wavefield and sum over time.


{++We start with the definition and derivation of the adjoint wave-equation and its [Devito] stencil, then we show how to compute the gradient of the conventional leas-square FWI misfit function. Finally we show a simple gradient on a 2D model and introduce a verification framework as well as a simple FWI gradient descent example in the notebook. As usual, this tutorial is accompanied by all the code you need to reproduce the figures. Go to  github.com/seg/tutorials-2017 and follow the links.++}

{++ Consider to add connection / evaluation sentence to connect to next section. ++}
## The adjoint wave-equation

{>>General remark. I think we need to think of ways to connect the theoretical framework earlier and better to the code. I understand that Devito does not offer our abstractions, we should do a better job to lure the readers in early.<<}

Conceptually, adjoint wave-equations seem like a very theoretical construct and understanding why we need adjoint wave-equations and what they are is not always intuitive. 

{>>I think this is a bit vague. Couple of things. First, we need to say somewhere that we follow the discretize and then optimized route, which makes it easier to derive verifiable codes with unit tests for gradients and adjoints. In the continuum case, there are many adjoints possible while for the discrete case it is unique. Also we need to say very briefly that the acoustic wave-equation is self adjoint but that there are many discrete wave-equations that do not have that property. I would add this somewhere since this is why we need to care. In particular since there are unstable "adjoints" floating around.<<} 

{>>**MLOU: I think this makes thing overly complicated and mentioning self-adjoint is just gonna make people walk away as "why care, it's self-adjoint" no matter what we say on top of it<<}

To get a better understanding of {==adjoint==} 

{>>May want to say that the adjoint is like the transpose of a matrix if we were to think of the wave-equation as a matrix.<<} {>>MLOU: next lines<<}}

wave-equations, we will revisit the forward simulation from the first part, but look at them from the linear algebra point of view ---i.e., as a matrix. In fact, we can think of solving a forward wave-equation as solving a linear system of equations

```math {#linWE}
    \mathbf{A}(\mathbf{m}) \mathbf{u} = \mathbf{q}, 
```

where the solution of this system is obtained by inverting the matrix $\mathbf{A}(\mathbf{m})$, i.e. $\mathbf{u} = \mathbf{A}(\mathbf{m})^{-1} \mathbf{q}$. The vector $\mathbf{q}$ is the seismic source, $\mathbf{u}$ is the vectorized wavefield for **all** time steps and $\mathbf{A}(\mathbf{m})$ represents the discretized acoustic wave-equation and consists of finite difference operators for spatial and temporal derivates, as well as the model $\mathbf{m}$ and damping terms. From the linear algebra

 {--{==and optimization==}--}
 
 {>>Needed here?<<} 
 
point of view, solving wave-equations therefore simply corresponds to multiplying the {==matrix==}

{>>I would reserve F for the modeling after restricting to the receivers. If you need a symbol for the action of Ainverse I would use G, which is the Green function. We need ``F`` and ``\Delta F`` for the forward modeling and its linearization. Now I am thinking about this a bit more I do think we may want to refrain from over emphasizing the matrix aspect and leave that for tutorial 3. I think just sticking to Ainverse should be fine and keeps it simple. <<}
 
$\mathbf{G} := \mathbf{A}(\mathbf{m})^{-1}$ with a vector $\mathbf{q}$. For calculating the gradient of (non-linear) {==least squares problems such as FWI==} [@Virieux]

{>>Is it at this point understood what this is?<<},
 
 we also need the action of the adjoint matrix $\mathbf{G}^\top$, which corresponds to solving the linear system 

```math {#adjWE}
    \mathbf{A}(\mathbf{m})^\top \mathbf{v} = \delta \mathbf{d},
```

with $\mathbf{v}$ being the adjoint wavefield and $\delta \mathbf{d}$ being the adjoint source typically the data residual $\delta \mathbf{d}$ for FWI. 

{--If we explicitly formed the full matrix $\mathbf{A}(\mathbf{m})$, we would see that $\mathbf{A}(\mathbf{m})$ is lower triangular, which means the only non-zero entries lie along or below the diagonal. However, due to its size ($\mathbf{A}(\mathbf{m})$ represents the full time-stepping scheme), the matrix is never explicitly formed and instead, we solve wave-equations by successively updating the wavefields at each time step using the (two) previous wavefields.--} 

{>>I think this is confusing and too early. Moreover, the restriction and injection operators are not yet formed so I would leave it more general and emphasize the adjoint aspect in the sense that we can thing of these discrete systems as matrices. <<} 

{>>Mlou:Restriction/injection are defined in part 1<<}


{--This process implicitly corresponds to inverting  $\mathbf{A}(\mathbf{m})$ via forward substitution, i.e. we invert the matrix time step by time step, starting at the top row of $\mathbf{A}(\mathbf{m})$ (the first time step).--}

{>>Again, I am afraid this is too much detail at this point. It is not a math tutorial. I would just explain that indeed the adjoint can be seen as a transpose of a matrix if we think of simulations as the inverse of the discretized wave-equation. One way to meet in the middle, would be to show the matrices with spy or something by making A explicit for a small problem. This could open the way to earlier bring in the reader to the numerical side of things. Problem is that we do most of our matrix restrictions in Julia. Perhaps we can do something in 1D with some utility functions albeit I am aware how big things become immediately but perhaps this could be a good set up for the next tutorial where we do everything with matrix-free abstractions.<<} 

{==For solving the adjoint wave-equation, we need to invert the transposed matrix, which accordingly is upper triangular, with all non-zero entries lying along or above the diagonal. Linear systems with upper triangular matrices are solved by backward substitution, which means we invert the matrix that represents the adjoint wave-equation by starting at the bottom row (the last time step) and by moving backwards in time (hence the expression reverse-time).==}

{>>This needs to be rewritten in simpler language and better tied in with the computational part of the tutorial. I suggest you do this along the lines I outlined above.<<}

To implement the adjoint wave-equation with [Devito], we once again start from the PDE in continuous form. With the variables defined as in part 1 and the data residual {==$\delta d(x,y,t; x_r, y_r)$, located at $x_r, y_r$, {++(receivers locations)++} ==} 

 {>>Not sure I follow this notation. It is the residual inserted at the receiver locations.<<} 
 
 as the adjoint source, the adjoint wave-equation is given by:

```math {#WEa}
 m(x,y) \frac{d^2 v(t,x,y)}{dt^2} - \nabla^2 v(t,x,y) - \eta(x,y) \frac{d v(t,x,y)}{dt}=\delta d(t,x,y;x_r, y_r)
```

{>>See my comments on the notation of part 1, it is inconsistent since we have ``m`` and ``\eta(x,y)``. I suggested to replace ``m`` by ``m(x,y)`` and introduce ``H(x,y;v)=\eta(x,y)\frac{\mathrm{d} v(t,x,y)}{\mathrm{d}t}``. Fix according in text and notice the correct ``\mathrm{d}`` versus ``d``. BTW ``H`` is capital ``\eta`` so notation not completely crazy perhaps. Coordinate w/ Mathias same problem in part 1.<<}

{>> MLOU: This H notation does not follow the implementation/definition in the stencil in Devito and makes it a bit obscure in my opinion.<<}

Since the acoustic wave-equation contains only second spatial and temporal derivatives, which are both self-adjoint, the adjoint acoustic wave-equation is equivalent to the forward equation with the exception of the damping term $\eta(x,y) \frac{d v(t,x,y)}{dt}$, which contains a first  {++ time ++} derivative and therefore has a change of sign in its adjoint. (A second derivative matrix is the same as its transpose, whereas a first derivative matrix is equal to its negative transpose and vice versa.)

Following part 1, we first define the discrete adjoint wavefield $\mathbf{v}$ as a [Devito] `TimeFunction` object and then symbolically set up the PDE and rearrange the expression:

```python
	# Discrete adjoint wavefield
	v = TimeFunction(name="v", grid=model.grid, time_order=2, space_order=2)
	
	# Define adjoint wave-equation and rearrange expression
	pde = model.m * v.dt2 - v.laplace - model.damp * v.dt
	stencil_v = Eq(v.backward, solve(pde, v.backward)[0])
```

Just as for the forward wave-equation, `stencil_v` defines the update for the adjoint wavefield of a single time step within the (reverse) time loop and corresponds to the expression: 

{>>Role of ``\text{time}`` unclear. It just to be an integer index but I am not longer sure. Was also an issue in paper 1. Add what interval it runs on.<<}{>>MLOU: \text{time} interval added, it follows Devito indexing naming convention<<}

```math {#WEdisadj}
\mathbf{v}[\text{time}-dt] = 2\mathbf{v}[\text{time}] - \mathbf{v}[\text{time}+dt] + \frac{dt^2}{\mathbf{m}}\Delta \mathbf{v}[\text{time}], \quad \text{time}=1 \cdots n_{t-1}
```

with {==``dt``==} {>>MLOU: smae as \text{time}, dt follows Devito var naming<<}
{>>I find this not the best notation since it is awfully close to ``\mathrm{d}t``<<}

 being the time stepping interval. Once again, this expression does not contain any (adjoint) source terms so far, which will be defined as a separate `SparseFunction` object. Since the source term for the adjoint wave-equation is the difference between an observed and modeled shot record, we first define an (empty) shot record `rec` with ``101`` receivers and coordinates defined in `rec_coords`. We then set the data field `rec.data` of our shot record to be the data residual between the observed data `d_obs` and the predicted data `d_pred`. The symbolic source expression `src_term` for our adjoint wave-equation is then obtained by *injecting* the data residual into the modeling scheme (`rec.inject`). Since we solve the time-stepping loop backwards in time, the `src_term` is used to update the previous adjoint wavefield `v.backward`, rather than the next wavefield. As in the forward modeling example, the source is scaled by  $\frac{dt^2}{\mathbf{m}}$. In Python, we have

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

{==In contrast to forward modeling, we do not need to define a (receiver) sampling operator, since we are only interested in the adjoint wavefield itself. ==} 
{>>Needed, we did not really introduce these operators, we only mentioned that we extract/save the wavefield at the receiver locations. We leave the operators to part 3.<<} 

The full script for setting up the adjoint wave-equation, including an animation of the adjoint wavefield is available in **`adjoint_modeling.ipynb`**.

{>>May want to say that we produce ``v`` everywhere and that that is an object we can typically not store so we evaluate it as part of the gradient calculations. <<}{>>MLOU: added in grad part<<}

**MLOU: I can add snapshot of adjoint wavefield if necessary and enough space**


## Computing the FWI gradient

The goal of FWI is to estimate a discrete parametrization of the subsurface by minimizing the misfit between the observed shot records of a seismic survey and numerically modeled shot records. The predicted shot records are obtained by solving an individual wave-equation per shot location and depend on the parametrization $\mathbf{m}$ of our wave propagator. The most common function for measuring the data misfit between the observed and modeled data is the $\ell_2$-norm, which leads to the following objective function [@LionsJL1971,@Tarantola]:

```math {#FWI}
	\mathop{\hbox{minimize}}_{\mathbf{m}} \hspace{.2cm} f(\mathbf{m})= \sum_{i=1}^{n_s} \frac{1}{2} \left\lVert \mathbf{d}^\mathrm{pred}_i (\mathbf{m}, \mathbf{q}_i) - \mathbf{d}_i^\mathrm{obs} \right\rVert_2^2,
```

{>>I very much prefer the ``f(m)``, ``\nabla f(m)`` notation for the objective and later in part 3 ``F(m)`` and ``\nabla F(m)`` notation for the forward modeling and its Jacobian.<<}

where the index $i$ runs over the total number of shots $n_s$ and the model parameters are the squared slowness. Optimization problems of this form are called non-linear least-squares problems, since the predicted {==data==} {>>Again I am not sure you define it as such since we are missing the restriction operators. Why not simply say that computing ``$\mathbf{d}_i^\mathrm{pred}`` involves inverting ``A``<<}

 $\mathbf{d}_i^\mathrm{pred} = \mathbf{A}(\mathbf{m})^{-1}\mathbf{q}_i$ depends on the unknown parameters $\mathbf{m}$ non-linearly. The full derivation of the FWI gradient using the adjoint state method is outside the scope of this tutorial, but conceptually we obtain the gradient by applying the chain rule and taking the partial derivative of the inverse wave-equation $\mathbf{A}(\mathbf{m})^{-1}$ with respect to $\mathbf{m}$, which yields the following expression [@Plessix2006; @Virieux]: 

```math {#FWIgrad}
 \nabla f (\mathbf{m})= -  \sum_{i=1}^{n_s}  \sum_{\text{time}=1}^{n_t} \ddot{\mathbf{u}}[\text{time}]\odot \mathbf{v}[\text{time}].
```
{>>See again call for clarification regarding the role of ``\text{time}``.<<}

The inner sum $\text{time}=1,...,n_t$ runs over the number of computational time steps $n_t$ and $\ddot{\mathbf{u}}$ denotes the second temporal derivative of the forward wavefield $\mathbf{u}$. Computing the gradient of Equation #FWI, therefore corresponds to performing the point-wise multiplication (denoted  by the symbol$\odot$) of the adjoint wavefields with the second time derivative of the forward wavefield and summing over all time steps and source positions. 

In practice, the FWI gradient (for a single source) is calculated in the reverse time-loop while solving the adjoint wave-equation. By updating the adjoint wavefield for the current time step $\mathbf{v}[\text{time}]$, we compute:

```math {#gradupd}
 \mathbf{g} = \mathbf{g} - \frac{\mathbf{u}[\text{time-dt}] - 2\mathbf{u}[\text{time}] + \mathbf{u}[\text{time+dt}]}{dt^2} \odot \mathbf{v}[\text{time}], \quad \text{time}=1 \cdots n_{t-1}
```

with ``\mathbf{g}`` the vector containing the gradient.

The second time derivative of the forward wavefield is computed with a second order finite-difference stencil and requires access to the forward wavefields of the previous, current and subsequent time step. To implement the FWI gradient with [Devito], we first define the gradient as a dense data object `Function`, which has no time dependence, since the gradient is computed as the sum over all time steps. The update for the gradient as defined in Equations #FWIgrad and #gradupd is then implemented in [Devito] by the following symbolic expression:

```python
grad = Function(name="g", grid=model.grid)
grad_update = Eq(grad, grad - u.dt2 * v)
``` 

{++In practice, saving only the forward wavefield ``\mathbf{u}`` is already a memory overburden, and saving both the forward and adjoint wavefield is not feasible. The gradient is therefore computed on-the-fly as part of the adjoint time loop. The definition of the gradient in [Devito] is straightforward and only requires to add the gradient update expression to the adjoint propagator (and turn off the illustrative saving flag on the adjoint wavefield):++}
{==Since the gradient is calculated as part of the adjoint time loop,==} 

{>>You need to add motivation why.<<}

{--we add the expression for the gradient to our previously defined stencil for the adjoint propagator:--}
 
 {>>This is very elegant and perhaps deserves a bit more text.<<}

```python
	op_grad = Operator([stencil_v] + src_term + grad_update,
						time_axis=Backward)
```

Solving the adjoint wave-equation by running `op_grad(time=nt, dt=model.critical_dt)` from the Python command line now includes computing the FWI gradient, which afterwards can be accessed with `grad.data`. 


#### Verification
The {--{==final==}--} {++next++}
{>>Bit strange I would do this before running the inversion. Also need to add a sentence why this is very important. Also may need a remark on what we do on the damping layer to make sure we pass gradient tests.<<} 

step of the adjoint modeling and gradient part is verification with unit testing, i.e. we ensure that the adjoints and gradients are implemented correctly. Incorrect adjoints can lead to unpredictable behaviour during and inversion and in the worst case cause slower convergence or convergence to wrong solutions. Since our forward-adjoint wave-equation solvers implicitly correspond to forward-adjoint matrix-vector products {--$\mathbf{G} \mathbf{q}$ and $\mathbf{G}^\top\mathbf{\delta d}$,--} we need to ensure that the relationship {--$\delta \mathbf{d}^\top \mathbf{G} \mathbf{q} = \mathbf{q}^\top\mathbf{G}^\top\mathbf{\delta d}$--} 

{>>For the sake of simplicity and the fact that we need to be careful since not all operators are introduced to included the math. We could also decide to leave this to part 3 since quasi-Newton methods need true gradients.... <<}

{>>MLOU: I would keep it as it is pure self contained verification of implementation."}

holds within machine precision (see **`tests/test_adjointA.py`** for the full adjoint test). Furthermore, we verify the correct implementation of the FWI gradient by ensuring that using the gradient leads to first order convergence. The gradient test can be found in **`tests/test_gradient.py`**.
 
  {>>Would move to part 3.<<}
  
#### Example

To further demonstrate the gradient computation, {==we perform==}

 {>>Yes but I would do this only after introducing a simple gradient decent possibly with a line search. Otherwise, we leave too much for part 3.<<}
 
 {>>MLOU: Simple gradient descent in the notebook<<}
 
a small seismic transmission experiment with the Camembert model, a constant velocity model with a circular high velocity inclusion in its centre. For a transmission experiment, we place ``21`` seismic sources on the left-hand side of the model and ``101`` receivers on the right-hand side. We then use the forward propagator from part 1 to independently model the ``21`` "observed" shot records using the true model. As the initial model for our gradient calculation, we use a constant velocity model with the same velocity as the true model, but without the circular velocity perturbation. We then model the ``21`` predicted shot records for the initial model, calculate the data residual and gradient for each shot and sum all ``21`` gradients

 {>>Really or are you summing on the fly.<<} 

to obtain the full gradient (Figure #Gradient). This result can be reproduced with the notebook **`adjoint_gradient.ipynb`**.

####Figure: {#Gradient}
![](Figures/camembert_true.pdf){width=33%}
![](Figures/camembert_init.pdf){width=33%}
![](Figures/simplegrad.pdf){width=33%}
: Camembert velocity true and initial model with sources (red dots), receivers (green dots) locations and the FWI gradient for 21 source locations, where each shot is recorded by 101 receivers located on the right-hand side of the model. The initial model used to compute the predicted data and gradient is a constant velocity model with the background velocity of the true model. This result can be reproduced by running the script **`adjoint_gradient.ipynb`**.

{++This gradient can then be used for a simple gradient descent as illustrated in the notebook. After each update a new gradient is computed for the new velocity model until sufficient decrease of the objective or chosen number of iteration is reached. More advanced algorithm will be described in the third part of this tutorial serie.++} 
## Conclusions

The gradient of the FWI objective function is computed by solving adjoint wave-equations and summing the point-wise product of forward and adjoint wavefields over all time steps. Using [Devito], the adjoint wave-equation is set up in a similar fashion as the forward wave-equation, with the main difference being the (adjoint) source, which is the residual between the observed and predicted shot records. The FWI gradient is computed as part of the adjoint time loop and implemented by adding its symbolic expression to the stencil for the adjoint propagator. With the ability to model shot records and compute gradients of the FWI objective function, we will demonstrate how to set up a full inversion framework in the next and final part of this tutorial series. 

{>>As I mentioned, I would add simple gradient descent and refer to more sophisticated optimization in part 3. <<}

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


