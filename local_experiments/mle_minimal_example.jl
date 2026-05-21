#!/usr/bin/env julia

# Minimal MLE state-estimation experiment.
#
# Run from the repository root:
#   julia +1.11 --project=local_experiments local_experiments/mle_minimal_example.jl
#
# The experiment reuses the tiny case3 test feeder and solves:
#   1. a relaxed WLS baseline with Normal measurement distributions,
#   2. an MLE run with the same Normal distributions,
#   3. a mixed run where one active-load measurement uses Gaussian-mixture MLE.

import PowerModelsDistributionStateEstimation as PMDSE
import PowerModelsDistribution as PMD
import Distributions as DST
import Ipopt

function make_solver()
    return PMDSE.optimizer_with_attributes(
        Ipopt.Optimizer,
        "max_cpu_time" => 300.0,
        "tol" => 1e-9,
        "print_level" => 0,
        "mu_strategy" => "adaptive",
    )
end

function load_case()
    repo_root = normpath(joinpath(@__DIR__, ".."))
    network_path = joinpath(repo_root, "test/data/extra/networks/case3_unbalanced.dss")
    measurement_path = joinpath(repo_root, "test/data/extra/measurements/case3_meas.csv")

    data = PMD.parse_file(network_path; data_model=PMD.MATHEMATICAL)
    PMDSE.add_measurements!(data, measurement_path; actual_meas=true)
    return data
end

function solve_case(data, solver, criterion)
    case_data = deepcopy(data)
    case_data["se_settings"] = Dict{String,Any}(
        "criterion" => criterion,
        "rescaler" => 1.0,
    )
    return PMDSE.solve_acp_red_mc_se(case_data, solver)
end

function solve_mixed_gmm_case(data, solver)
    case_data = deepcopy(data)

    # Replace one scalar active-load measurement by a non-Gaussian mixture of
    # Normal components. GMM MLE also needs finite min/max bounds for the shift
    # calculation in constraint_mc_residual.
    case_data["meas"]["4"]["dst"] = Any[
        DST.MixtureModel(
            [DST.Normal(0.010, 0.0015), DST.Normal(0.016, 0.0020)],
            [0.7, 0.3],
        ),
    ]
    case_data["meas"]["4"]["min"] = 0.0
    case_data["meas"]["4"]["max"] = 0.03

    case_data["se_settings"] = Dict{String,Any}("rescaler" => 1.0)
    PMDSE.assign_basic_individual_criteria!(case_data; chosen_criterion="rwls")

    return PMDSE.solve_acp_red_mc_se(case_data, solver)
end

function summarize(label, result)
    println(label)
    println("  termination_status = ", result["termination_status"])
    println("  objective          = ", result["objective"])

    if haskey(result, "solve_time")
        println("  solve_time         = ", result["solve_time"])
    end
end

function main()
    solver = make_solver()
    data = load_case()

    pf_result = PMD.solve_mc_pf(data, PMD.ACPUPowerModel, solver)

    rwls_result = solve_case(data, solver, "rwls")
    normal_mle_result = solve_case(data, solver, "mle")
    mixed_gmm_result = solve_mixed_gmm_case(data, solver)

    _, rwls_max_err, rwls_avg_err = PMDSE.calculate_voltage_magnitude_error(rwls_result, pf_result)
    _, mle_max_err, mle_avg_err = PMDSE.calculate_voltage_magnitude_error(normal_mle_result, pf_result)
    _, gmm_max_err, gmm_avg_err = PMDSE.calculate_voltage_magnitude_error(mixed_gmm_result, pf_result)

    summarize("Relaxed WLS baseline", rwls_result)
    println("  max_vm_error       = ", rwls_max_err)
    println("  avg_vm_error       = ", rwls_avg_err)
    println()

    summarize("Normal-distribution MLE", normal_mle_result)
    println("  max_vm_error       = ", mle_max_err)
    println("  avg_vm_error       = ", mle_avg_err)
    println()

    summarize("Mixed rwls + GMM MLE", mixed_gmm_result)
    println("  max_vm_error       = ", gmm_max_err)
    println("  avg_vm_error       = ", gmm_avg_err)
end

main()
