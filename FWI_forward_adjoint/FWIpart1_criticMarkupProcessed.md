---
title: Full-Waveform Inversion - Part 1``:`` forward and adjoint modeling
runninghead : Part 1, modeling for inversion
author: |
	Mathias Louboutin^1^\*, Philipp Witte^1^, Michael Lange^2^, Navjot Kurjeka^2^, Fabio Luporini^2^, Gerard Gorman^2^ and Felix J. Herrmann^1,3^\
	^1^ Seismic Laboratory for Imaging and Modeling (SLIM), The University of British Columbia \
	^2^ Imperial College London, London, UK\
	^3^ now at Georgia Institute of Technology, USA \
bibliography:
	- bib_tuto.bib
---

<span class="critic comment">Not sure where the order of the authors come from. Since you and Philipp are in my Lab and first authors, I suggest you put me last as is customary. You must also make sure that you get detailed input on this rewrite from all co-authors. Multiple times I suggested that the Python commands should be integrated more tightly to the text. Mathias really needs help from the other authors how to accomplish this. I think that the section in the Discretization would benefit mostly from this. I also took out all references to operators since this would only confuse the readers and should be left to part 2. Too much is left to the reader and that is not good for a tutorial like this.</span>

## Introduction

Full-waveform inversion (FWI) gained <del>tremendous</del><ins> a lot of</ins> attention in geophysical exploration since it was first introduced [@Pratt] <span class="critic comment">You need to say why just loose statement means nothing "because of its ...."</span>. However, the literature mostly contains specific and technical papers about applications and advanced <del>method</del><ins>methods</ins> and <ins> often </ins> lacks simple introductory resources for geophysical newcomers. Mathematical and geophysical FWI papers, as @Virieux, give excellent theoretical overviews, but typically <del>do not cover</del><ins>pay less attention to</ins> the implementation side of the problem. In this two part tutorial, we provide a hands-on walkthrough of FWI using Devito, a finite-difference domain-specific language that provides a concise and straightforward <del>interface</del><ins>computational framework</ins> for discretizing wave equations and generating <del>operators</del><ins>propagators</ins> for <ins> the </ins> forward and adjoint <del>modeling</del><ins>wave equation</ins>. <ins> Because Devito releases the user from .... , </ins> Devito ([@lange2016dtg]) allows the user to concentrate on the geophysical side of the problem, rather than the low-level implementation details of a wave-equation simulator. This tutorial covers the conventional adjoint-state formulation of full-waveform tomography <span class="critic comment">Add ref</span> <ins> that underlies most of the current methods generally referred to as full-waveform inversion.</ins> <ins>While </ins> other <del>methods</del><ins>formulations</ins> <del>exist</del><ins>have been developed</ins> to improve the convergence properties of the algorithm, <ins> we will concentrate on the standard formulation that relies  </ins> relies on <ins> the combination of a </ins> forward/adjoint pairs <ins> propagators and </ins> an correlation-based gradient <del>and should use the proposed framework</del><ins>calculations</ins>.

<ins>As part of this tutorial for FWI, we will first introduce</ins> 

 - propagators that simulate <ins> forward modeled </ins> synthetic data, <ins> which we compare with  </ins> <del>that can be compared to </del>field recorded data<ins>, and  </ins>
 - adjoint <del>operator</del><ins>propagators</ins> that back-propagate the data residual <del>and compute the cross-correlation</del><ins>followed by gradient computations via cross-correlation of the forward and back-propagated wavefields.</ins> <span class="critic comment">I think there is redundancy here and suggest you merge these items with the text above.</span>

<ins> To explain how FWI works, </ins> <del>We will illustrate the </del><ins>we describe a typical</ins>workflow on a <del>very</del> simplistic 2D model that can be run on a laptop or desktop PC. <ins> Unfortunately, </ins> larger and more realistic models come at a computational <ins> cost </ins> and memory <del>price</del><ins>requirements</ins> that <del>are</del><ins>easily go</ins> beyond <del>of</del> the <del>scope of </del><ins>the type of hardware people will have available to reproduce the results presented in </ins>this tutorial. <ins>However,</ins> the workflow we describe <ins> is general enough that it easily </ins> translates to <ins> much larger </ins> velocity models in 2D and 3D and <ins> to </ins> <del>any type of </del> <ins> more complicated </ins> wave equations  <del>with</del><ins>as long as their adjoints are known.</ins> <span class="critic comment">Please add ref.</span> <ins>For maximal access to our software framework, we divided the aforementioned workflow into the following Python notebooks (available at url):</ins>

<del>The workflow for full-waveform inversion, and the corresponding notebooks, is the following:</del>

<span class="critic comment">I renamed the note books since it is completely confusing. Split into forward modeling; adjoint modeling + residual calculation, gradient calculation.</span>

- **[forward_modeling.ipynb]** <span class="critic comment">Make this an active link</span> --- in this notebook we describe how to simulate synthetic data and how to save the corresponding wavefields;

- **[adjoint_modeling.ipynb]** --- here we demonstrate how to compute the data residual---i.e., the difference between the synthetic and observed data how to back-propagate this residual wavefield with a propagator computed with the adjoint wave equation that acts on the residual;
 
- **[gradient.ipynb]** --- here we describe how we calculate the gradient by cross-correlation of the forward and adjoint wavefields over time. In this notebook, we also show how to repeat this for all sources, at the different iterations of simple gradient descent algorithm.
 
<del>We start with the description of a modeling operator and then move on to the adjoint operator and gradient for FWI. </del> <span class="critic comment">Redundant</span> <ins> We refer to Part 2 of this tutorial, for a more complete description on how to minimize  FWI objectives and how to gain matrix-free access to the Jacobian and (Gauss-Newton) Hessian of FWI.</ins>

<del>A complete tutorial on how to optimize the FWI objective function, once the computational framework is in place, will be covered in the second part. This tutorials is linked to three notebooks that detail each step of the implementation from modeling to FWI. For clarity purposes, some details will be left out of the article but are fully detailed in the corresponding notebooks.</del>

## <del>Modelling</del><ins>Wave simulations for inversion</ins>

The acoustic wave equation for the squared slowness ``m``, defined as ``m=\frac{1}{c^2}`` with ``c`` being the speed of sound, and a source ``q`` is given by:

```math {#WE}
 m \frac{d^2 u(x,t)}{dt^2} - \nabla^2 u(x,t) + \eta \frac{d u(x,t)}{dt}=q 
```

where ``\eta`` is the dampening parameter for the absorbing boundary layer [@Cerjan]. The physical model is extended in every direction by `nbpml` grid points to mimic an infinite domain and the dampening term ``\eta \frac{d u(x,t)}{dt}`` attenuates the waves in the dampening layer [@Cerjan]. In Devito, the physical parameters ``m`` and ``\eta`` are contained in the `model` object with the relevant information such as the origin, grid spacing and size.

```python
	# Define a Devito model with physical size, velocity vp 
	# and absorbing layer width in number of grid points
	model = Model(origin, spacing, shape, vp, nbpml=nbpml)
```

The model parameters are illustrated on Figure #model\.

####Figure: {#model}
![Model](Figures/setup.png){}
: Representation of the model.


<mark> doesn't show but all this until discretization section is removed</mark>
<del><ins>In this formulation, we included</ins> zero initial conditions to guarantee <mark>unicity</mark> <span class="critic comment">What is this. Is this the right language for a tutorial?</span> of the solution. <del>The</del><ins>These</ins> boundary conditions <del>are</del><ins>correspond to</ins> Dirichlet conditions <ins> given by </ins> <span class="critic comment">Is this really needed this language will piss the geop community off unnecessarily. Know your audience!!!!</span>&nbsp;u(x,t)|_{\delta\Omega} = 0&nbsp;where ``\delta\Omega`` is the surface of the boundary of the model ``\Omega``.</del><del>&nbsp;<span class="critic comment">I am missing a clear connection between these conditions and what you describe in the paragraph below. It is your responsibility to make this very clear to the reader! It is not enough to just do a brain dump and have the reader connect the dots.</span>&nbsp;<ins>Aside from these ....</ins> the field, seismic wave propagate in all directions in an "infinite" medium. However, solving the wave equation in a mathematically/discrete infinite domain is not feasible. <del>Therefore,</del><ins>For this reason, we use,</ins> <span class="critic comment">How many times have we told you to avoid passive tense!</span> Absorbing Boundary Conditions (ABC) or Perfectly Matched Layers (PML) {--are used</del> in practice to mimic an infinite domain. <ins>The two methods</ins> <span class="critic comment">Again connection is not clear? What are these two methods and are they both necessary?</span> allow <ins> us </ins> to approximate an infinite medium by <mark>damping and absorbing </mark> <span class="critic comment">unclear what is causing what</span> the waves at the <mark>limit of</mark> <span class="critic comment">What is this the edges? Unclear</span> the domain to avoid reflections <ins>from the boundaries that may introduce unphysical prismatic waves.</ins>

<mark>The simplest of these methods is the absorbing damping mask.</mark> <span class="critic comment">I really not know why you are introducing these "two" methods. I am afraid it does nothing but confuse. Take out.</span> The core idea is to extend the physical domain and to add a sponge <del>mask</del><ins>layer</ins> <span class="critic comment">This is not a mask.</span> in this extension that will absorb the incident waves. In our case, we use ABC where ``\eta`` is the damping mask equal to ``0`` inside the physical domain and increasing inside the sponge layer. Multiple choice of profile can be chosen for ``\eta`` from linear to exponential [@Cerjan].

<span class="critic comment">I am sorry but the paragraph above is incomprensible. There are issues w/ the English and also just plain explanatory issues. The reader will not know what ``\eta`` is for instance. I suggest Philipp or others come with an acceptable reformulation of the two above paragraphs so it is crystal clear what is going on.</span>--}

### Discretization

<ins> As we mentioned earlier, </ins> we discretize the wave equation with Devito, a finite-difference DSL that <ins> is designed to define and solve </ins> <del>solves the</del> <del>discretized</del><ins>discrete</ins> wave-equations on <del>a</del> Cartesian grids. <ins> To arrive at the discretized wave equation, we use </ins> <del>The</del> finite-difference approximations <del>is</del> derived from Taylor expansions of the continuous <del>field</del><ins>wavefields as they appear in Equation and</ins>  after removing the error term <span class="critic comment">Is this really to only reference that is relevant? Hard to belief. Also doe the readership of this tutorial any idea what this error term is and is it relevant?</span> .


The first step <ins> in our discretization </ins> is to define a symbolic representation of the discrete wavefield. In Devtio, this is <del>represented</del><ins>done</ins> by <ins> instantiating </ins> a `TimeData` <span class="critic comment">Use single quotes for programmatic objects so they are not confused w/ math symbols. Fix throughout!</span> object <del>and</del><ins>that</ins> contains all <ins> necessary/relevant </ins> information for the discretization <del>such a</del><ins>including </ins> discretization orders in space and time<ins>---i.e., we run the following line in our notebook:</ins>

```python
	u = TimeData(name="u", shape=model.shape_domain, time_order=2, space_order=2, save=True, time_dim=nt)
```

<ins> To avoid unnecessary complications, we use </ins>
a second-order discretization in time. <del>, which is the most commonly used time discretization.</del> From the Taylor expansion of the continuous wavefield ``u`` in time, the second order discrete approximation of the second-order time derivative as a function of the discrete wavefield ``\vd{u}`` is <ins> given by </ins> <span class="critic comment">I am sorry but the notation is completely confusing since it mixes continuous and discrete (bold) quantities. There are also typos in the equations below and unnecessary repetition. I suggest this is cleaned up and presented in a way that is appropriate for the audience.</span>

```math {#timedis}
 \frac{d^2 u(i \Delta t)}{dt^2} = \frac{\vd{u}[i+1] - 2 \vd{u}[i] + \vd{u}[i-1]}{\Delta t^2} + \mathcal{O}(\Delta t^2).
```

and the finite-difference approximation is 
```math
	\vd{\ddot{u}}[i] =  \frac{\vd{u}[i+1] - 2 \vd{u}[i] + \vd{u}[i-1]}{\Delta t^2}.
``` 

<del>where ``\vd{u}`` is the discrete wavefield, <ins> The </ins></del> ``\Delta t`` is the discrete time-step (distance between two consecutive discrete times). <del>and ``\mathcal{O}(\Delta t^2)`` is the discretization error term. <mark>The discretized approximation of the second order time derivative is then given by dropping the error term. </mark> <span class="critic comment">I am sorry but this is confusing to me </span></del> <ins>  In Devito, we represent </ins> this time derivative <ins> symbolically as </ins> `u.dt2`.

Apart from the temporal derivative, the acoustic wave equation <ins> also </ins> contains spatial <del>(second) </del> derivatives. <ins> For the constant density acoustic wave equation, the spatial derivative involve the action of the Laplacian </ins><del>We therefore define the discrete Laplacian </del>``\Delta \vd{u}[i]``,  which after discretization we define as the sum of second- order spatial derivatives <del>in the three </del> <ins> along the Cartesian coordinate directions.  </ins> <del>dimensions</del>. Each second<ins> -order </ins> spatial derivative <ins> itself </ins> is discretized <del>with</del><ins>by</ins> a ``k^{th}`` order finite-difference <del>scheme</del><ins>approximation</ins>  (`space_order=k` in the `TimeData` object <del>creation</del><ins>instantiation</ins>) and also derive from <ins> the </ins> Taylor expansion <span class="critic comment">Is this redundant?</span>. <ins> We represent </ins> the Laplacian <del>is represented </del> in Devito by `u.laplace`. <del>and <ins> we </ins> follow the same theoretical derivation as in Equation #timedis applied to the space variables ``x,y,z``. <span class="critic comment">I am afraid I am not following the last part of this sentence.</span></del>

With the space and time discretization defined, we can <ins> now </ins> <del>fully</del> <ins> automatically instantiate the `Stencil` objects, which will later be used to automatically generate executable C code. `Stencil` implements the action of a single timestep according to our discretization scheme, which is second order in time and  ``k^{th}`` order in space---i.e., mathematically we generate executable code that implements the following expression:</ins>

<del> discretize the wave-equation with the combination of the temporal and spatial discretizations and obtain the following second order in time and ``k^{th}`` order in space discrete stencil to update one grid point at position ``x,y,z`` at time ``t``:
</del>
<span class="critic comment">How often do you need to be told to not be so incredible sloppy w/ math notation. Your co-authors should not have to tell you that ``\Delta t`` should not be bold. This sort of lack of attention to detail sends a very negative message I am afraid and you need to train yourself to fix these things yourself certainly if this has been mentioned to you before.</span>

```math {#WEdis}
\vd{u}[i+1] = 2\vd{u}[i] - \vd{u}[i] + \frac{\Delta t^2}{\vd{m}} \Big(\Delta \vd{u}[i]+ \vd{q}[i] \Big). 
```

for each ``i`` in the interval ``[0, n_t]``.
<del><span class="critic comment">Is this expression correct??????</span></del>


### Setting up the acquisition geometry

<ins>In Devito, we model monopole sources/receivers with the object `PointData`, which includes methods that interpolate between the computational grid on which the wave equation is discretized and possibly off-the-grid source/receiver locations. The code that implements the definition of ```101``` receiver and one source with locations collected in the arrays `rec_coords` and `src_coords` reads as</ins> <span class="critic comment">I removed the terms operators since many people will not know what you mean with this. I think we also should leave the definition of linear operators that do this for part 2. So for now I just refer to them as source and receiver arrays.</span>
```
	# create receiver array from receiver coordinates
	rec = Receiver(name='rec', npoint=101, ntime=nt, ndim=2, coordinates=rec_coords)
	rec_term = rec.interpolate(u, offset=model.nbpml)
	
	# define source injection array for given a source wavelet, coordinates and frequency
	src = RickerSource(name='src', ndim=2, f0=f0, time=time, coordinates=src_coords)
	src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m, offset=model.nbpml)
```

<span class="critic comment">What is this offset?????</span>

where `dt**2 / model.m` is derived from equation #WEdis\ and offset is the size of the absorbing layer as displayed on Figure #model (source position shifted by `offset`).


### Wave equation stencil

<del>Using</del><ins>In</ins> Devito, we can directly translate <ins> <del>this</del> expression #WEdis, and therefore create executable code by issuing the following commands in our notebook: </ins> <del>the discretized wave equation into a symbolic expression and define a ```stencil``` expression, which defines the update for the new wavefield at each time step:</del>

```python
	# Set up acoustic wave equation
	pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
	# Generation of the stencil
	stencil = Eq(u.forward, solve(pde, u.forward)[0])
```

where `u.forward` is a Devito shortcut for ``\vd{u}[i+1]``
<span class="critic comment">u.forward is not defined or I may have missed something.</span>

<del><ins>As we can see, this</ins> wave equation does not contain a source term and is solely defined through its initial conditions(<ins>see Equation</ins> #WE). <ins>So </ins> to simulate an <ins> actual </ins> seismic experiment, we still need to <del>define</del><ins>introduce</ins> a seismic source, <del>i.e. a seismic wavelet which </del><ins>e.g. a monopole with a temporal source signature that</ins> is injected into the model at a predefined source location. <ins>Aside from injecting wavefield at one or more sources, we also need to extract wavefields at the receivers.</ins> <ins> Like sources, we treat these receivers as "monopole" sinks, which record the modeled  wavefield as a function of time at predefined receiver locations. </ins> <span class="critic comment">In US English modeled is with one l so please change it everywhere.</span></del>



### Forward simulation 

With the source/receiver <del>projection operators</del><ins>geometry set</ins> and the wave-equation stencil, we can <ins> now </ins> define our <del>full modeling operator</del><ins>forward propagator</ins> by symbolically adding the source and receiver terms into our previously defined 'Stencil' object---i.e., we have

```python
	# Create forward propagator
	op_fwd_ = Operator([stencil] + src_term + rec_term,
	              subs={t.spacing: dt, x.spacing: spacing[0],
	                    y.spacing: spacing[1]})
```

and shot records can be modeled by simply calling `op_fwd.apply()`. Once the propagator executed, the wavefield and shortrecord are accessed via:
```python
	wavefield = u.data
	shotrecord = rec.data
```

 In Figure #Forward, we show the resulting shot record. A movie of snapshots of the forward wavefield can be <del>found</del><ins>generated by executing </ins> the last cell of this notebook.
 

####Figure: {#Forward}
![Shot record](Figures/shotrecord.pdf){width=45%}
: Shot record on a two layer model for a single source and split-spread receiver geometry from **modeling.ipynb**.

### Backward simulation

<span class="critic comment">As I mentioned at more than one time during conversations you need to discuss the adjoint wavefield as well including a movie. This can be done very succinctly. </span>

The adjoint wave-equation is:

```math {#WEa}
 m \frac{d^2 v(x,t)}{dt^2} - \nabla^2 v(x,t) - \eta \frac{d v(x,t)}{dt}= \delta \vd{d}
```

Implementation of the adjoint modeling is straightforward in the self-adjoint acoustic case (except for what happens at in the damping layer). The only detail to consider is to adjust the non self-adjoint boundary conditions, which corresponds simply to a change of sign. Using Devito, we can define the adjoint wave equation propagator in a similar manner injecting the data residual, ``\delta \vd{d}=\vd{d}^{\mathrm{syn}}(\vd{m};\vd{q}) - \vd{d}^{\mathrm{obs}}``, as a source. <span class="critic comment">While this may be the case, I would STRONLY advice against this since it goes against the whole philosophy of Devito.</span>

```python
	v = TimeData(name="v", shape=model.shape_domain, time_order=2, space_order=2)
	# Receiver setup
	rec = Receiver(name='rec', npoint=101, ntime=nt, ndim=2, coordinates=rec_coords)
	
	# This lime comes from nowhere and is incomprehensible.
	rec_term = rec.inject(field=v.backward,  expr=rec * dt**2 / model.m, offset=model.nbpml)
	
	# Define adjoint wave equation
	pde = model.m * v.dt2 - v.laplace + model.damp * v.dt
	stencil_v = Eq(v.backward, solve(pde, v.backward)[0])
	
	# Create propagator
	op_adj = Operator([stencil_v] + src_term + rec_term,
	              subs={t.spacing: dt, x.spacing: spacing[0],
	                    y.spacing: spacing[1]},
						time_axis=Backward)
```

An animation of the adjoint wavefield is available in **[adjoint_modeling.ipynb]**.

## Objective and gradient

Full-waveform inversion aims <del>at recovering </del><ins>to recover</ins> accurate <ins> estimates for the </ins> the discrete wave slowness vector, ``\vd{m} = \vd{c}^{-2}``, from a given set of measurements of the pressure wavefield ``\vd{u}``. Following [@LionsJL1971, @Virieux, @haber10TRemp, @Tarantola], <ins> inversion process corresponds to minimizing the following FWI objective:</ins> 

```math {#FWI}
	\mathop{\hbox{minimize}}_{\vd{m}} f(\vd{m})=\frac{1}{2}\left\lVert \vd{d}^{\mathrm{syn}}(\vd{m};\vd{q}) - \vd{d}^{\mathrm{obs}}\right\rVert_2^2,\\
```

where ``\vd{d}^{\mathrm{syn}}(\vd{m};\vd{q})`` is the synthetic data, which depends on the unknown slowness  vector ``\vd{m}`` and the discretized source function ``\vd{q}``, which we assume to be known. FWI aims to find a slowness vector ``\vd{m}`` that minimizes the energy between synthetic data and data measured in the field collected in the vector ``\vd{d}``. 

We minimize this objective by computing updates to the slowness that are given by the gradient of FWI objective with respect to ``\vd{m}``. Following work by (add refs), this gradient is given by the zero-lag term of the cross-correlation between the second time derivative of the forward wavefield, ``\vd{\ddot{u}}`` and the adjoint wavefield, ``\vd{v}``,---i.e. we have 

```math {#FWIgrad}
 \nabla f(\vd{m};\vd{q})= - \sum_{{i} =1}^{n_t}\vd{\ddot{u}}[i]\odot \vd{v}[i],
```

where the sum runs over all ``n_t`` time samples and ``\odot`` represents element-wise multiplication of two vectors that contain the spatial variations of the wavefield at discrete time index ``t``.

<del>The partial derivative of the modeling operator ``\frac{d \vd{A}(\vd{m}) \vd{u}}{d\vd{m}}`` in Equation #FWIgradLA is simply the second time derivative, since ``\vd{m}`` appears only in front of this term (equation #WEdis). The parameter ``n_t`` is the number of computational time steps, ``\delta\vd{d} = \left(\vd{P}_r \vd{u} - \vd{d} \right)`` is the data residual (difference between the measured data and the modeled data), ``\vd{J}`` is the Jacobian (i.e. the linearized modeling or demigration operator) and ``\vd{u}_{tt}`` is the second-order time derivative of the forward wavefield solving #linWE\.</del>

### Computing the gradient

While the derivation of the above expression for the gradient goes beyond the scope of this tutorial, it important to emphasize how the forward and adjoint wavefields are calculated with the forward and backward simulations introduced above. Mathematically, forward simulation to compute the forward wavefield ``\vd{u}`` for each source involves the the solution of the following linear system of equations:

```math {#linWE}
    \vd{A}(\vd{m}) \vd{u} = \vd{q}, 
```
where ``\vd{q}`` represents the known source. With the previous definition for the sources, solving this system corresponds in Devito to running the following commands:

```
	op_fwd.apply()
```
<span class="critic comment">Python code to compute u and then second derivative of u goes here.</span>

Solutions for the corresponding adjoint wavefields, ``\vd{v}``, are computed in a similar fashion

```math {#adjWE}
    \vd{A}(\vd{m})^\top \vd{v} = \delta \vd{d}.
```


<del>where the adjoint source is computed by injecting the data residual, ``\delta \vd{d}=\vd{d}^{\mathrm{syn}}(\vd{m};\vd{q}) - \vd{d}^{\mathrm{obs}}``, as a source.</del> In this expression, we obtain the expression for the backward propagating wavefield by transposing (denoted by the symbol ``^\top``) the linear system associated with the forward simulations. <del>To avoid numerical instability associated with the damping in the boundary, we ... </del>

In Devito, the computation of the adjoint wavefield is carried out by

```
 op_ad.apply()
 ```

<span class="critic comment">The Python code is very confusing since it in par uses the same name for certain "variables". Also you use u instead of v and you are running the forward equation backwards while it would make more sense to actually generate a separate code for the adjoint as outlined in the section above. </span>

<del>Solving the wave-equation is equivalent to solving the linear system ``\vd{Au}=\vd{q}`` where the vector ``\vd{u}`` is the discrete wavefield solution of the discrete wave-equation, ``\vd{q}`` is the source term and ``\vd{A}`` is the matrix representation of the discrete wave-equation. From Equation #WEdis we can see that the matrix ``\vd{A}`` is a lower triangular matrix that reflects the time-marching structure of the stencil. Simulation of the wavefield is equivalent to a forward elimination on the lower triangular matrix ``\vd{A}``. The adjoint of ``\vd{A}``, denoted as ``\vd{A}^T``, is then an upper triangular matrix and the solution ``\vd{v}`` of the discrete adjoint wave-equation ``\vd{A}^\top\vd{v}=\vd{q}_a`` for an adjoint source ``\vd{q}_a`` is equivalent to a backward elimination on the upper triangular matrix ``\vd{A}\top`` and is simulated backward in time starting from the last time-step. These matrices are never explicitly formed, but are instead matrix free operators with implicit implementation of ``\vd{u}=\vd{A}^{-1}\vd{q}``.</del> <span class="critic comment">All nice but irrelevant for a tutorial.</span>

<del>Implementation of the adjoint modeling is straightforward in the self-adjoint acoustic case (except for what happens at in the damping layer). The only detail to consider is to adjust the non self-adjoint boundary conditions, which corresponds simply to a change of sign. Using Devito, we can define the adjoint wave equation in the same fashion as the forward equation: <span class="critic comment">While this may be the case, I would STRONLY advice against this since it goes against the whole philosophy of Devito.</span>
</del>

When calculating the gradient, we need as explained in Equation #FWIgrad to simply sum the pointwise multiplication of the adjoint wavefield with the second-time derivative of the forward wavefield. In Devito, this is symbolically expressed by `grad_update = Eq(grad, grad - u.dt2 * v)`. The full script for calculating the gradient is given in the notebook **adjoint_gradient.ipynb**. The computation of the gradient is implemented adding the gradient  update expression to the adjoint propagator:

```python
	op_grad = Operator([stencil_v_] + src_term + rec_term + grad_update,
	              subs={t.spacing: dt, x.spacing: spacing[0],
	                    y.spacing: spacing[1]},
						time_axis=Backward)
```

Before we take a look at what the gradient for our test model looks like, we frist want to ensure that our implementations of the forward and adjoint wave equations are in fact a correct forward-adjoint pair. Not having correct adjoints can lead to wrong gradients, which in turn may lead to convergence to a wrong solution, or to slower convergence. To ensure that the discretized wave equations and associated propagators are implemented correctly, they need to pass the so-called **dot** and **gradient** tests, which can be found in the Devito test **tests/test_adjointA.py** and **tests/test_gradient.py**.

Having tested the forward/adjoint wave equations, we can now calculate the gradient of the FWI objective function for a simple 2D test model. The Camembert model consists of a constant medium with a circular high velocity zone in its centre and we perform a transmission experiment, with the source on one side of the model and receivers at the other side. The gradient for a constant starting model (without the circular perturbation) looks as follows:

####Figure: {#Gradient}
![Gradient for a transmission camembert model and a single source-receiver pair](Figures/banana.pdf){width=45%}
![Gradient for a transmission camembert model and a sfull shot record](Figures/simplegrad.pdf){width=45%}
:Gradients for a simple camembert transmission model **adjoint_gradient.ipynb**.

Finally, with the gradient implemented, we can easily setup the FWI objective function that can be used in an optimization toolbox as we will show in the next part of the tutorial. A example of an FWI objective function is given in cell 18 of **adjoint_gradient.ipynb**.


## Conclusions

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

<style>
	.scholmd-container {
		padding-top: 60px !important;
	}

	#criticnav {
	  position: fixed;
	  z-index: 1100;
	  top: 0;
	  left: 0;
	  width: 100%;
	  border-bottom: solid 1px #696f75;
	  margin: 0;
	  padding: 0;
	  background-color: rgba(255,255,255,0.95);
	  color: #696f75;
	  font-size: 14px;
	  font-family: "Helvetica Neue", helvetica, arial, sans-serif !important
	}

	#criticnav ul {
	  list-style-type: none;
	  width: 90%;
	  margin: 0 auto;
	  padding: 0
	}

	#criticnav ul li {
	  display: block;
	  width: 15%;
	  min-width: 100px;
	  text-align: center;
	  padding: 5px 0 3px !important;
	  margin: 5px 2px !important;
	  line-height: 1em;
	  float: left;
	  text-transform: uppercase;
	  cursor: pointer;
	  -webkit-user-select: none;
	  border-radius: 20px;
	  border: 1px solid rgba(255,255,255,0);
	  color: #777 !important
	}

	#criticnav ul li:before {
	  content: none !important
	}

	#criticnav ul li.active {
	  border: 1px solid #696f75
	}

	.original del {
		
			text-decoration: none;
	}	

	.original ins,
	.original span.popover,
	.original ins.break {
		display: none;
	}

	.edited ins {
		
			text-decoration: none;
	}	

	.edited del,
	.edited span.popover,
	.edited ins.break {
		display: none;
	}

	.original mark,
	.edited mark {
		background-color: transparent;
	}

	.markup mark {
	    background-color: #fffd38;
	    text-decoration: none;
	}

	.markup del {
	    background-color: rgba(183,47,47,0.4);
	    text-decoration: none;
	}

	.markup ins {
	    background-color: rgba(152,200,86,0.4);
	    text-decoration: none;
	}

	.markup ins.break {
		display: block;
		line-height: 2px;
		padding: 0 !important;
		margin: 0 !important;
	}

	.markup ins.break span {
		line-height: 1.5em;
	}

	.markup .popover {
		background-color: #e5b000;
		color: #fff;
	}

	.markup .popover .critic.comment {
	    display: none;
	}

	.markup .popover:hover span.critic.comment {
	    display: block;
	    position: absolute;
	    width: 200px;
	    left: 30%;
	    font-size: 0.8em; 
	    color: #ccc;
	    background-color: #333;
	    z-index: 10;
	    padding: 0.5em 1em;
	    border-radius: 0.5em;
	}

	@media print {
		#criticnav {
			display: none !important
		}
	}
}

</style>

<div id="criticnav">
<ul>
<li id="markup-button">Markup</li>
<li id="original-button">Original</li>
<li id="edited-button">Edited</li>
</ul>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.1.1/jquery.min.js"></script>
<script type="text/javascript">

	function critic() {

		$('.scholmd-container').addClass('markup');
		$('#markup-button').addClass('active');
		$('ins.break').unwrap();
		$('span.critic.comment').wrap('<span class="popover" />');
		$('span.critic.comment').before('&#8225;');

	}  

	function original() {
		$('#original-button').addClass('active');
		$('#edited-button').removeClass('active');
		$('#markup-button').removeClass('active');

		$('.scholmd-container').addClass('original');
		$('.scholmd-container').removeClass('edited');
		$('.scholmd-container').removeClass('markup');
	}

	function edited() {
		$('#original-button').removeClass('active');
		$('#edited-button').addClass('active');
		$('#markup-button').removeClass('active');

		$('.scholmd-container').removeClass('original');
		$('.scholmd-container').addClass('edited');
		$('.scholmd-container').removeClass('markup');
	} 

	function markup() {
		$('#original-button').removeClass('active');
		$('#edited-button').removeClass('active');
		$('#markup-button').addClass('active');

		$('.scholmd-container').removeClass('original');
		$('.scholmd-container').removeClass('edited');
		$('.scholmd-container').addClass('markup');
	}

	var o = document.getElementById("original-button");
	var e = document.getElementById("edited-button");
	var m = document.getElementById("markup-button");

	window.onload = critic;
	o.onclick = original;
	e.onclick = edited;
	m.onclick = markup;

</script>
