# MLE Implementation Summary

This note summarizes how the exact maximum likelihood estimation (MLE) method
from `Vanin et al. - 2023 - Exact Modeling of Non-Gaussian Measurement
Uncertainty in Distribution System State Estimation.pdf` is implemented in this
repository.

## Sources and Traceability

Inline citations throughout this note use paper sections/equations, repository
documentation paths, and implementation paths. Primary sources used for this
note are listed below. Code citations use `file:start-end` line ranges from the
current working tree:

- Paper: `Vanin et al. - 2023 - Exact Modeling of Non-Gaussian Measurement
  Uncertainty in Distribution System State Estimation.pdf`.
  - Section II.A, equations (1)-(14): constrained DSSE formulation, MLE
    likelihood/log-likelihood objective, Gaussian/WLS special case, MLE residual
    shift.
  - Section II.B: implementation in PowerModelsDistributionStateEstimation.jl,
    JuMP, Distributions.jl, solver separation, supported distribution families.
  - Section III: pseudo-measurement uncertainty models, Beta, polynomial
    log-pdf, GMM, Gaussian approximations, and reactive-power assumptions.
  - Section IV: numerical setup, ACR formulation choice, voltage-magnitude
    conversion, Ipopt usage, and solve-time/accuracy observations.
- Package docs:
  - `docs/src/problems.md`: state-estimation variable/constraint/objective
    structure and residual-function role.
  - `docs/src/se_criteria.md`: WLS/WLAV/MLE residual definitions, rescaler,
    supported MLE distributions, and derivative requirements.
  - `docs/src/measurements.md`: measured-variable, formulation-variable, and
    state-variable distinction plus measurement-conversion equations.
  - `docs/src/input_data_format.md`: `data["meas"]` schema, measurement CSV
    fields, `data["se_settings"]`, `rescaler`, `criterion`, and pseudo-measurement
    assumptions.
- Implementation files:
  - `src/prob/se.jl:49-201`: `solve_mc_se` and formulation-specific `build_mc_se`
    methods.
  - `src/prob/se_en.jl:1-54`: explicit-neutral IVR state-estimation builder.
  - `src/core/objective.jl:11-19`: residual-sum objective.
  - `src/core/constraint.jl:14-70`: WLS/WLAV/MLE residual constraints and MLE log-pdf
    registration.
  - `src/core/variable.jl:16-168`: residual variables, load variables, and auxiliary
    measurement variables.
  - `src/core/measurement_conversion.jl:85-515`: conversion constraints between
    measured variables and formulation variables.
  - `src/core/utils.jl:76-140`: criterion assignment, bounds helpers, residual upper
    bounds, and measurement utilities.
  - `src/io/measurement_parser.jl:39-100`: measurement CSV parsing and distribution
    object creation.
  - `src/io/distributions.jl:9-170`: `ExtendedBeta`, `heslogpdf`, and Gaussian mixture
    derivative support.
  - `src/io/polynomials.jl:3-6`: polynomial log-pdf, gradient, and Hessian support.
  - `test/runtests.jl:31-36`: typical Ipopt optimizer configuration used in tests.

External optimization background used for the solver explanation:

- Ipopt solves smooth nonlinear programs using a primal-dual interior-point
  method and KKT conditions. This is reflected in the paper's use of Ipopt and
  in JuMP's standard nonlinear-programming interface, but the detailed
  algorithmic explanation is general nonlinear optimization background rather
  than package-specific code.
- SciPy `L-BFGS-B` comparison is based on the method's standard scope: smooth
  objectives with simple bound constraints, not general nonlinear equality
  constraints. It is included as contextual solver guidance, not as a feature of
  this repository.

## High-Level Model

The paper formulates distribution system state estimation as a constrained
optimization problem [paper: Sec. II.A, Eqs. (1)-(4); `docs/src/problems.md`]:

```math
\begin{aligned}
\min        &\sum_j \rho_j \\
\text{s.t.} &\text{residual definitions} \\
            &\text{power-flow equality constraints} \\
            &\text{optional inequality constraints and bounds}
\end{aligned}
```

The implementation follows that structure. The package builds a JuMP model using
PowerModelsDistribution variables and constraints, adds state-estimation residual
variables, then minimizes the sum of those residuals [`src/prob/se.jl:49-201`;
`src/prob/se_en.jl:1-54`; `src/core/objective.jl:11-19`].

The core objective is implemented in `src/core/objective.jl:11-19`
[`src/core/objective.jl:11-19`]:

```julia
sum(residuals over all measurements and networks)
```

The formulation builders are in `src/prob/se.jl:49-201` and `src/prob/se_en.jl:1-54`
[`src/prob/se.jl:49-201`; `src/prob/se_en.jl:1-54`].

## Shape of the State and Variable Space

There is no single packed numerical state vector in the code. Instead, the
physical state and the extended optimization variables are JuMP variables stored
in PowerModelsDistribution dictionaries, keyed by network, component, component
id, and conductor/phase [`src/prob/se.jl:49-201`; `src/core/variable.jl:16-168`].

It is useful to distinguish two concepts:

- The physical state of interest is the network voltage state, i.e. the bus
  voltage phasors [paper: Sec. II.A, Eq. (8); `docs/src/measurements.md`].
- The optimization variable space is larger. It may include active and reactive
  powers, currents, branch flows, residuals, and auxiliary measurement variables
  because these make the constrained state-estimation model easier to express
  [paper: Sec. II.A; `docs/src/problems.md`; `src/prob/se.jl:49-201`].

In the paper's main case, the selected formulation is ACR, whose variable space
is [paper: Sec. II.A]:

```math
x \in X_{\mathrm{ACR}} = \{U^{re}, U^{im}, P, Q\}
```

Here, `x` is best understood as the optimization vector, not only the physical
state vector. The physical state is represented by the rectangular voltage
variables `U^{re}` and `U^{im}`. The active and reactive power variables `P` and
`Q` are part of the extended formulation space and are linked to the voltage
state by the network constraints and measurement model [paper: Sec. II.A,
Eqs. (5)-(7); `docs/src/problems.md`; `docs/src/measurements.md`].

In this repo, the exact variable space depends on the selected formulation:

- ACP/ACR bus-injection models create bus voltage variables, branch power,
  transformer power, generator power, load power, residual variables, and
  measurement-conversion variables [`src/prob/se.jl:49-201`].
- IVR models create voltage and current variables, plus current-based generator
  and load variables [`src/prob/se.jl:49-201`; `src/core/variable.jl:16-168`].
- Branch-flow, SDP, and LinDist3Flow-style models use their native PMD variable
  spaces [`src/prob/se.jl:49-201`; `docs/src/formulations.md`].
- Explicit-neutral IVR has a separate builder in `src/prob/se_en.jl:1-54`
  [`src/prob/se_en.jl:1-54`; `docs/src/explicit_neutral_models.md`].

So, when referring to this implementation, a precise phrasing is:

> The estimated physical state is the bus voltage phasor state. The JuMP
> optimization model uses an extended variable space that also contains `P`,
> `Q`, current, flow, residual, and measurement-conversion variables as required
> by the selected formulation.

## MLE Residual Definition

The MLE residual is implemented in `src/core/constraint.jl:14-70`, inside
`constraint_mc_residual` [`src/core/constraint.jl:14-70`].

For a measurement with:

```julia
meas["crit"] == "mle"
```

the implementation registers a scalar nonlinear JuMP function based on the
measurement distribution's log-pdf [`src/core/constraint.jl:14-70`]. Conceptually:

```math
\rho = \mathrm{rsc}\,(\mathrm{shift} - \log f(x))
```

where:

- `rho` is the residual variable minimized in the objective [`src/core/variable.jl:16-168`;
  `src/core/objective.jl:11-19`],
- `rsc` is `data["se_settings"]["rescaler"]` [`src/prob/se.jl:49-201`;
  `docs/src/input_data_format.md`],
- `f` is the measurement pdf [paper: Sec. II.A, Eqs. (9)-(12);
  `docs/src/se_criteria.md`],
- `x` is the JuMP variable corresponding to the measured quantity
  [`src/core/constraint.jl:14-70`; `src/core/measurement_conversion.jl:85-515`],
- `shift` moves the residual so its minimum is intended to be near zero.
  [paper: Sec. II.A, Eq. (14); `src/core/constraint.jl:14-70`]

The code computes the shift by solving a one-dimensional optimization problem
[`src/core/constraint.jl:14-70`]:

```julia
shf = abs(Optim.optimize(x -> -logpdf(distribution, x), lb, ub).minimum)
```

Then it registers:

```julia
fun(x) = rsc * (-shf + logpdf(distribution, x))
res == -fun(measured_variable)
```

Minimizing the sum of residuals is therefore equivalent to maximizing the sum of
log-likelihood terms, up to constant shifts [paper: Sec. II.A, Eqs. (9)-(14);
`docs/src/se_criteria.md`; `src/core/objective.jl:11-19`].

## Optimization Method

The repository does not implement a custom Gauss-Newton or custom MLE solver.
It constructs a nonlinear constrained JuMP model and delegates the solve to the
solver supplied by the user [paper: Sec. II.B; `src/prob/se.jl:49-201`].

The solve path is:

```julia
solve_mc_se(data, model_type, solver; kwargs...)
```

which calls:

```julia
PowerModelsDistribution.solve_mc_model(data, model_type, solver, build_mc_se; kwargs...)
```

The practical sequence is:

1. `solve_mc_se` checks and fills `data["se_settings"]` [`src/prob/se.jl:49-201`].
2. `PowerModelsDistribution.solve_mc_model` creates the JuMP model for the
   selected formulation [`src/prob/se.jl:49-201`].
3. The relevant `build_mc_se` method adds formulation variables, network
   constraints, residual variables, measurement-conversion variables, residual
   constraints, and the residual-sum objective [`src/prob/se.jl:49-201`;
   `src/prob/se_en.jl:1-54`; `src/core/objective.jl:11-19`].
4. JuMP passes the resulting nonlinear program through MathOptInterface (MOI) to
   the optimizer supplied by the user [paper: Sec. II.B; JuMP/MOI background].

For MLE measurements, the nonlinear part of the objective is not written
directly as a JuMP objective expression. Instead, each MLE term is represented
by a residual variable constrained to equal the shifted negative log-likelihood.
The log-pdf is registered as a scalar nonlinear function [`src/core/constraint.jl:14-70`]:

```julia
JuMP.register(pm.model, f, 1, fun, grd, hes)
JuMP.add_nonlinear_constraint(pm.model, :(res == -f(measured_variable)))
```

The implementation provides:

- `fun`: the shifted/scaled log-pdf value [`src/core/constraint.jl:14-70`],
- `grd`: the first derivative of the log-pdf [`src/core/constraint.jl:14-70`;
  `src/io/distributions.jl:9-170`; `src/io/polynomials.jl:3-6`],
- `hes`: the second derivative of the log-pdf [`src/core/constraint.jl:14-70`;
  `src/io/distributions.jl:9-170`; `src/io/polynomials.jl:3-6`].

Providing these derivatives is important because the nonlinear solver uses them
to build local linear/quadratic approximations of the nonlinear constrained
problem. This is also why the implementation requires smooth pdf/log-pdf models
[paper: Sec. II.B; `docs/src/se_criteria.md`; nonlinear optimization background].

The paper's numerical experiments used Ipopt [paper: Sec. IV]. The test suite
also primarily uses Ipopt [`test/runtests.jl:31-36`]. With Ipopt, the model is solved
as a nonlinear program using a primal-dual
interior-point method with line-search/filter globalization. Conceptually, Ipopt
iterates over candidate primal and dual variables, solves derivative-based
linearized subproblems, updates the variables, and drives the
Karush-Kuhn-Tucker (KKT) optimality, feasibility, and complementarity residuals
toward tolerance [Ipopt/nonlinear optimization background].

For this implementation, that means:

- PowerModelsDistributionStateEstimation defines the model equations and
  likelihood residuals [`src/prob/se.jl:49-201`; `src/core/constraint.jl:14-70`].
- JuMP/MOI translates the symbolic model to the solver interface [paper: Sec. II.B;
  JuMP/MOI background].
- Ipopt or another nonlinear optimizer chooses the iterates and solves the NLP
  [paper: Sec. IV; nonlinear optimization background].
- Solver settings such as tolerances, maximum time, print level, linear solver,
  and scaling are controlled by the optimizer attributes passed by the user
  [`test/runtests.jl:31-36`; JuMP optimizer-attribute interface].

The tests commonly construct Ipopt like this [`test/runtests.jl:31-36`]:

```julia
optimizer_with_attributes(
    Ipopt.Optimizer,
    "max_cpu_time" => 300.0,
    "obj_scaling_factor" => 1e3,
    "tol" => 1e-9,
    "print_level" => 0,
    "mu_strategy" => "adaptive",
)
```

This means the implementation gives a local nonlinear solution unless the chosen
problem class and solver provide stronger guarantees [Ipopt/nonlinear
optimization background; paper: Sec. IV].

The quality and robustness of the solve therefore depend on:

- the chosen formulation, such as ACR, ACP, IVR, or LinDist3Flow;
- whether the resulting model is convex or nonconvex [paper: Sec. II.A;
  `docs/src/formulations.md`];
- the smoothness and scaling of the log-pdf terms [paper: Sec. II.B;
  `docs/src/se_criteria.md`];
- variable bounds and initial values [`src/core/utils.jl:76-140`;
  `src/core/start_values_methods.jl:13-40`];
- measurement weighting/rescaling via `data["se_settings"]["rescaler"]`
  [`src/prob/se.jl:49-201`; `docs/src/input_data_format.md`];
- the nonlinear solver and its options [`test/runtests.jl:31-36`; JuMP/MOI background].

### Ipopt Compared With SciPy L-BFGS-B

Ipopt is a good fit for this implementation because the MLE state-estimation
model is a constrained nonlinear program. The model contains nonlinear
power-flow equality constraints, residual-definition constraints,
measurement-conversion constraints, variable bounds, and optionally additional
physical constraints [paper: Sec. II.A; `src/prob/se.jl:49-201`; `src/core/constraint.jl:14-70`;
`src/core/measurement_conversion.jl:85-515`].

Ipopt is designed for smooth nonlinear optimization problems with nonlinear
equality and inequality constraints. Through JuMP, it can use the model's
constraint Jacobians and nonlinear derivative information to search for a point
that satisfies the constraints and locally minimizes the residual objective
[Ipopt/nonlinear optimization background; JuMP/MOI background].

SciPy's `L-BFGS-B` solves a narrower problem class. It handles smooth objectives
with simple lower and upper bounds on variables, but it does not natively handle
general nonlinear equality constraints such as the power-flow equations
[SciPy L-BFGS-B method scope; nonlinear optimization background]. To use
`L-BFGS-B`, the model would need to be reformulated, for example by:

- eliminating the network constraints analytically, if possible;
- adding power-flow and measurement constraints as penalty terms in the
  objective;
- solving a different reduced problem where feasibility is not enforced by the
  optimizer itself.

Those reformulations would no longer match the clean constrained optimization
model used in the paper and in this package. In particular, penalty-based
approaches can trade off likelihood improvement against physical feasibility,
whereas the JuMP/Ipopt formulation enforces the network equations as constraints.
[paper: Sec. II.A; `src/prob/se.jl:49-201`]

In short:

| Solver | Fit for this implementation |
| --- | --- |
| Ipopt | Good fit: smooth constrained nonlinear program with equality/inequality constraints. |
| SciPy `L-BFGS-B` | Poor direct fit: supports bounds only; nonlinear constraints would require a different formulation. |

Ipopt still does not guarantee a global optimum for nonconvex formulations such
as ACP, ACR, or IVR. It should be understood as a local nonlinear optimizer. Even
so, it is much more aligned with the MLE implementation in this repository than
`L-BFGS-B` [Ipopt/nonlinear optimization background; `docs/src/formulations.md`].

## Measurement Representation

Measurements are stored under `data["meas"]` [`docs/src/input_data_format.md`;
`src/io/measurement_parser.jl:39-100`]. A typical MLE-capable measurement looks like:

```julia
data["meas"]["1"] = Dict{String,Any}(
    "var" => :pd,
    "cmp" => :load,
    "cmp_id" => 4,
    "dst" => Any[distribution_phase_1, distribution_phase_2, distribution_phase_3],
    "crit" => "mle"
)
```

Important fields:

- `"var"`: measured variable, such as `:pd`, `:qd`, `:vm`, `:pg`, `:qg`
  [`docs/src/input_data_format.md`; `src/io/measurement_parser.jl:39-100`].
- `"cmp"`: component type, such as `:load`, `:gen`, `:bus`, `:branch`
  [`docs/src/input_data_format.md`; `src/io/measurement_parser.jl:39-100`].
- `"cmp_id"`: component id in the mathematical PMD data model
  [`docs/src/input_data_format.md`; `src/io/measurement_parser.jl:39-100`].
- `"dst"`: vector of per-phase scalar distributions
  [`docs/src/input_data_format.md`; `src/io/measurement_parser.jl:39-100`].
- `"crit"`: residual model, set to `"mle"` for exact non-Gaussian MLE
  [`docs/src/input_data_format.md`; `docs/src/se_criteria.md`;
  `src/core/constraint.jl:14-70`].

Each measurement phase/conductor is treated separately. The implementation skips
the neutral conductor index `_N_IDX = 4` when building residuals
[`src/PowerModelsDistributionStateEstimation.jl:33-44`; `src/core/constraint.jl:14-70`].

## Measurement Conversion

The measured variable does not have to be native to the selected formulation.
If needed, `variable_mc_measurement` creates an auxiliary measurement variable
and links it to the formulation variables with a conversion constraint
[`docs/src/measurements.md`; `src/core/variable.jl:16-168`;
`src/core/measurement_conversion.jl:85-515`].

Examples:

- Voltage magnitude in ACR/IVR:

```math
v_m^2 = v_r^2 + v_i^2
```

- Active/reactive power in IVR:

```math
p = v_r i_r + v_i i_i
```

```math
q = v_i i_r - v_r i_i
```

These conversions are implemented in `src/core/measurement_conversion.jl:85-515`.
The MLE residual is then applied to the native or auxiliary variable associated
with the measured quantity [`src/core/variable.jl:16-168`; `src/core/constraint.jl:14-70`].

## Equality Constraints

Equality constraints are central to the implementation. They are added as hard
JuMP constraints, not as soft penalty terms in the objective [`src/prob/se.jl:49-201`;
`src/core/constraint.jl:14-70`; `src/core/measurement_conversion.jl:85-515`].

The build flow is:

```julia
solve_mc_se(...)
    -> PowerModelsDistribution.solve_mc_model(..., build_mc_se)
        -> build_mc_se(pm)
            -> create variables
            -> add network equality constraints
            -> add measurement-conversion constraints
            -> add residual definition constraints
            -> add objective
```

The resulting optimization problem has the structure:

```math
\begin{aligned}
\min        &\sum_m \rho_m \\
\text{s.t.} &h_{\mathrm{pf}}(x) = 0 \\
            &h_{\mathrm{conv}}(x, y) = 0 \\
            &h_{\mathrm{res}}(\rho, y) = 0 \\
            &\ell \leq x \leq u
\end{aligned}
```

where:

- `h_pf` are the power-flow and network physics equations [paper: Sec. II.A,
  Eqs. (3), (5)-(7); `src/prob/se.jl:49-201`],
- `h_conv` are measurement-conversion equations [`docs/src/measurements.md`;
  `src/core/measurement_conversion.jl:85-515`],
- `h_res` are residual-definition equations [paper: Sec. II.A, Eqs. (12)-(14);
  `docs/src/se_criteria.md`; `src/core/constraint.jl:14-70`],
- `x` denotes formulation variables [paper: Sec. II.A; `docs/src/problems.md`],
- `y` denotes native or auxiliary measured variables [`src/core/variable.jl:16-168`],
- `rho` denotes residual variables [`src/core/variable.jl:16-168`;
  `src/core/objective.jl:11-19`].

The main equality-constraint categories are:

- Network constraints, such as power balance, branch Ohm's-law constraints,
  transformer constraints, current balance, and voltage-drop constraints
  [`src/prob/se.jl:49-201`; `src/prob/se_en.jl:1-54`].
- Reference constraints, such as fixing voltage angles at reference buses
  [paper: Sec. II.A, Eq. (8); `src/prob/se.jl:49-201`; `docs/src/angular_ref.md`].
- Measurement-conversion constraints, such as mapping voltage magnitude to
  rectangular voltage variables [`docs/src/measurements.md`;
  `src/core/measurement_conversion.jl:85-515`].
- Residual-definition constraints, such as WLS, deterministic measurement, and
  MLE residual equations [`docs/src/se_criteria.md`; `src/core/constraint.jl:14-70`].

For example, deterministic measurements are enforced as [`src/core/constraint.jl:14-70`]:

```julia
var[c] == measured_value
res[idx] == 0.0
```

WLS residuals use equality constraints of the form [paper: Sec. II.A, Eq. (13);
`docs/src/se_criteria.md`; `src/core/constraint.jl:14-70`]:

```math
\rho \cdot \mathrm{rsc}^2 \sigma^2 = (x - \mu)^2
```

MLE residuals are enforced by registering the log-pdf as a nonlinear scalar
function and adding [`src/core/constraint.jl:14-70`]:

```math
\rho = -f(x)
```

where `f(x)` is the shifted/scaled log-pdf expression.
[paper: Sec. II.A, Eqs. (12)-(14); `docs/src/se_criteria.md`]

Equality constraints change the effective loss landscape. The raw objective is
the sum of residuals, but the optimizer can only move on the feasible set:

```math
\{x : h(x) = 0\}
```

So the relevant landscape is the residual objective restricted to the network's
physical feasibility manifold. A pseudo-measurement pdf may prefer a certain
power value, but the optimizer may only choose that value if it is compatible
with the power-flow equations, voltage state, bounds, and other measurements.
[paper: Sec. II.A; constrained optimization background]

Mathematically, this is handled through Lagrange multipliers. For equality
constraints `g(x) = 0`, the constrained first-order optimality conditions are
based on the Lagrangian:

```math
\mathcal{L}(x, \lambda) = f(x) + \lambda^T g(x)
```

where:

- `f(x)` is the residual objective,
- `g(x)` is the vector of equality constraints,
- `lambda` are the Lagrange multipliers.

At a constrained local optimum, the objective gradient does not generally need
to be zero by itself. Instead, it is balanced by the constraint gradients:

```math
\nabla f(x) + J_g(x)^T \lambda = 0
```

with:

```math
g(x) = 0
```

This is the key difference from a penalty formulation. The constraints are not
merely added to the loss with a large weight; they are separate equations with
associated multipliers that enforce feasibility and shape the stationarity
conditions [constrained optimization/KKT background].

In Ipopt, equality constraints enter the Karush-Kuhn-Tucker (KKT) system through
the constraint Jacobian. Conceptually, Ipopt solves linearized systems involving:

```math
\begin{bmatrix}
W & J^T \\
J & 0
\end{bmatrix}
```

where `J` is the equality-constraint Jacobian and `W` is the Hessian of the
Lagrangian. This means equality constraints directly affect both the search
direction and numerical conditioning [Ipopt/KKT background].

Ipopt does not solve the Lagrangian equations in one step. It uses an iterative
primal-dual interior-point method. During the solve, it updates:

- the primal variables, such as voltage, power, current, and residual variables;
- equality-constraint multipliers;
- inequality and bound multipliers;
- barrier parameters for inequalities and bounds.

So, in practical terms, hard JuMP equality constraints are implemented through
the constrained optimization machinery of the solver, and Lagrange multipliers
are the mathematical mechanism that represents their effect on the optimum
[Ipopt/KKT background; JuMP/MOI background].

### Numerical Stability Impact

Equality constraints can improve numerical behavior by removing impossible or
unphysical directions from the search space. For example, power balance prevents
the optimizer from fitting measurements with a state that violates network
physics. Reference-angle constraints remove rotational symmetry and improve
identifiability [paper: Sec. II.A; `src/prob/se.jl:49-201`; `docs/src/angular_ref.md`].

They can also hurt numerical stability when the constrained system is poorly
scaled, nearly redundant, highly nonlinear, or inconsistent [Ipopt/KKT
background; nonlinear optimization background].

Important stability mechanisms in this implementation:

- Scaling mismatch: voltage variables are typically around `1.0` p.u., while
  powers, residuals, and log-pdf values can have very different magnitudes. The
  `rescaler` setting and solver scaling options help manage this [`src/prob/se.jl:49-201`;
  `docs/src/input_data_format.md`; `test/runtests.jl:31-36`].
- Ill-conditioned constraint Jacobian: nearly dependent power-flow, conversion,
  or reference constraints can make Ipopt's KKT linear systems difficult to
  solve accurately [Ipopt/KKT background].
- Nonlinear conversion constraints: equations such as
  `vm^2 = vr^2 + vi^2`, current-power products, and division-style conversions
  introduce curvature. If the iterate is far from feasible, local
  linearizations can be poor [`docs/src/measurements.md`;
  `src/core/measurement_conversion.jl:85-515`; nonlinear optimization background].
- Hard feasibility: equality constraints must be satisfied tightly. If
  measurements, bounds, or deterministic constraints conflict with the network
  equations, the optimizer may report infeasibility or enter restoration phases
  [Ipopt/nonlinear optimization background].
- Reference constraints: these are necessary for identifiability, but an
  overly restrictive or inappropriate reference model can bias the solution or
  make convergence harder [`docs/src/angular_ref.md`; `src/prob/se.jl:49-201`].
- Deterministic measurements: exact constraints like `var == value` are strong.
  They can improve observability, but wrong deterministic values make the model
  brittle because they cannot be relaxed through residual minimization
  [`src/core/constraint.jl:14-70`].

Practical signs of equality-constraint-related numerical trouble include:

- `LOCALLY_INFEASIBLE` termination,
- restoration-phase messages from Ipopt,
- slow convergence,
- sensitivity to starting values,
- `ALMOST_LOCALLY_SOLVED`,
- large final constraint violation,
- materially different solutions after small scaling or initialization changes.

Good practice is to use per-unit scaling, reasonable variable bounds, consistent
measurement data, appropriate reference-angle modeling, and solver options suited
to the formulation and data scale [`docs/src/input_data_format.md`;
`src/core/utils.jl:76-140`; `src/core/start_values_methods.jl:13-40`; `test/runtests.jl:31-36`].

## Supported Distribution Types

The MLE path assumes each measurement uncertainty is a continuous univariate
distribution with:

- `logpdf`
- `gradlogpdf`
- `heslogpdf`

The implementation supports distributions from Distributions.jl and local
extensions, including:

- Normal
- Exponential
- Weibull
- LogNormal
- Beta
- Gamma
- ExtendedBeta
- Gaussian mixture models
- polynomial log-pdf fits

[paper: Sec. II.B; `docs/src/se_criteria.md`; `src/io/distributions.jl:9-170`;
`src/io/polynomials.jl:3-6`]

`ExtendedBeta` and Hessian helpers are implemented in `src/io/distributions.jl:9-170`.
Polynomial log-pdf support is in `src/io/polynomials.jl:3-6`.

For Gaussian mixture models, the implementation assumes the mixture components
are Normal distributions [`src/io/distributions.jl:9-170`].

## Gaussian and Mixed Criteria

The implementation can mix criteria per measurement. A measurement can use MLE
while another uses WLS, rWLS, WLAV, or rWLAV [paper: Sec. II.A;
`docs/src/se_criteria.md`; `src/core/constraint.jl:14-70`].

This matches the paper's setup where smart meter measurements can be modeled as
Gaussian while pseudo-measurements can be modeled with non-Gaussian pdfs
[paper: Sec. II.A, Eqs. (12)-(13)].

The helper `assign_basic_individual_criteria!` assigns:

- the chosen Gaussian criterion to measurements with Normal distributions,
- `"mle"` to non-Normal measurements.

[`src/core/utils.jl:76-140`]

## Main Assumptions

The implementation makes several practical assumptions:

- Measurement uncertainties are scalar and per phase/conductor
  [`docs/src/input_data_format.md`; `src/core/constraint.jl:14-70`].
- The likelihood is separable across measurements, so the total log-likelihood
  is a sum of independent scalar log-pdf terms [paper: Sec. II.A, Eqs. (9)-(10);
  `src/core/objective.jl:11-19`].
- Correlations are not represented as a joint probability distribution
  [paper: Sec. III.B; `src/core/constraint.jl:14-70`].
- Correlations or modeling choices such as constant power factor must be added
  as explicit constraints, not as correlated pdfs [paper: Sec. II.C, Sec. III.B,
  Sec. III.C].
- The measurement pdf or log-pdf must be smooth enough for nonlinear
  optimization [paper: Sec. II.B; `docs/src/se_criteria.md`].
- First and second derivatives must be available through `gradlogpdf` and
  `heslogpdf` [`src/core/constraint.jl:14-70`; `src/io/distributions.jl:9-170`;
  `src/io/polynomials.jl:3-6`].
- For GMM and polynomial log-pdf measurements, finite `"min"` and `"max"` bounds
  must be present in the measurement dictionary so the shift can be computed
  [`src/core/constraint.jl:14-70`].
- Bounds on voltages, loads, and generators are optional but important for
  numerical tractability [`src/core/utils.jl:76-140`; `docs/src/problems.md`].
- Exact MLE means exact use of the provided pdf/log-pdf in the optimization
  model. It does not imply globally optimal nonlinear optimization [paper:
  Sec. II.A; Ipopt/nonlinear optimization background].

## Practical Caveats

- `solve_mc_se` is typed as accepting a `String`, but it indexes
  `data["se_settings"]` before parsing, so file-path input likely does not work
  as typed [`src/prob/se.jl:49-201`].
- `variable_mc_residual` has a bug in the `res_max` path: it references
  `meas["res_max"]` even though `meas` is not defined in that scope
  [`src/core/variable.jl:16-168`].
- The shift calculation is intended to make residuals nonnegative, but exact
  zero-minimum behavior depends on the log-pdf value range and support bounds
  [paper: Sec. II.A, Eq. (14); `src/core/constraint.jl:14-70`].
- Non-Gaussian measurement parsing through the generic helpers is limited. The
  docs recommend users who rely heavily on non-Gaussian distributions build
  their own measurement creator/parser [`docs/src/input_data_format.md`].

## Bottom Line

The repo implements the paper's exact MLE idea by putting each non-Gaussian
measurement's log-pdf directly into the JuMP optimization model. The resulting
state-estimation problem is a constrained nonlinear program over the selected
PowerModelsDistribution formulation variables, with a residual objective that
maximizes the measurement likelihood while enforcing the network physics
[paper: Sec. II.A; paper: Sec. II.B; `src/prob/se.jl:49-201`;
`src/core/constraint.jl:14-70`; `src/core/objective.jl:11-19`].
