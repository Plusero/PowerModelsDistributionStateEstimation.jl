# Local MLE Experiments

This folder contains small, runnable scripts for inspecting the MLE state
estimation path outside the package test suite.

## Julia Installation

Use Juliaup for a modern Julia installation. Juliaup is the official installer
and version manager recommended by the Julia project; it installs Julia without
using the operating system package manager and lets you update or switch Julia
versions later.

On Linux or macOS:

```sh
curl -fsSL https://install.julialang.org | sh
```

After installation, restart the shell or source the profile file mentioned by
the installer, then verify:

```sh
julia --version
juliaup status
```

This repository currently declares compatibility with Julia `1.6 - 1.11`, so
install and use the Julia `1.11` channel for these experiments:

```sh
juliaup add 1.11
```

Useful Juliaup commands:

```sh
juliaup update
juliaup add lts
juliaup default release
```

Avoid installing Julia from Ubuntu `apt` or `snap` for this repo unless you have
a specific reason; those channels can lag behind the official Julia binaries.

## Run The Example

Prepare the experiment environment from the repository root:

```sh
julia +1.11 --project=local_experiments -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Then run the minimal example:

```sh
julia +1.11 --project=local_experiments local_experiments/mle_minimal_example.jl
```

The experiment environment keeps solver dependencies such as `Ipopt` local to
this folder instead of adding them to the package's main `Project.toml`.
