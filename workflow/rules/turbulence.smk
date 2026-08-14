TURBULENCE_DEFAULTS = {
    "enabled": False,
    "input": repo_path("inputs/turbulence_wendeln.in"),
    "dirname": "turbulence",
    # Keep configured multi-scheme comparisons separate from single-scheme results.
    "output_group": "mhd_comparison",
    "energy_plotting_script": repo_path(
        "workflow/diagnostics_scripts/turbulence/energies.py"
    ),
    "slice_plotting_script": repo_path(
        "workflow/diagnostics_scripts/turbulence/slices.py"
    ),
    "spectrum_plotting_script": repo_path(
        "workflow/diagnostics_scripts/turbulence/plot_power_spectra.py"
    ),
    "resolution": [64, 64, 64],
    "meshblock": [32, 32, 32],
    "tlim": 60.0,
    "history_dt": 0.1,
    "output_dt": 0.5,
    # Optional, case-specific Parthenon restart files. Cases not listed here start
    # from the input deck as usual.
    "restart_from": {},
    "cases": {
        "glm_weno3js": {
            "label": "AthenaPK WENO3",
            "fluid": "glmmhd",
            "riemann": "hlld",
            "reconstruction": "weno3js",
            "integrator": "rk3",
            "convergence_order": 2,
            "first_order_flux_correct": False,
            "nghost": 2,
        },
        "glm_weno5js": {
            "label": "AthenaPK WENO5",
            "fluid": "glmmhd",
            "riemann": "hlld",
            "reconstruction": "weno5js",
            "integrator": "rk3",
            "convergence_order": 2,
            "first_order_flux_correct": False,
            "nghost": 3,
        },
        "uct2_weno3": {
            "label": "UCT-HLLD 2nd",
            "fluid": "ucthlldmhd",
            "riemann": "hlld",
            "reconstruction": "weno3",
            "integrator": "rk2",
            "convergence_order": 2,
            "first_order_flux_correct": False,
            "nghost": 2,
        },
    },
    "spectra": {
        "enabled": False,
        "dump_ids": [118, 119, 120],
        "analysis_nodes": 1,
        "analysis_tasks": 8,
        "mem_mb_per_cpu": 4000,
        "runtime": 120,
        "kernels": ["Gauss"],
        "python": config["plotting_python"],
        "flow_analysis_script": repo_path(
            "workflow/diagnostics_scripts/turbulence/energy_transfer_analysis/"
            "run_analysis.py"
        ),
    },
    "nodes": 1,
    "tasks_per_node": 8,
    "mpi_tasks": 8,
    "mem_mb_per_cpu": 4000,
}
TURBULENCE = {
    **TURBULENCE_DEFAULTS,
    **config["tests"].get("turbulence", {}),
}
TURBULENCE_SPECTRA = {
    **TURBULENCE_DEFAULTS["spectra"],
    **TURBULENCE.get("spectra", {}),
}
TURBULENCE_SPECTRA["flow_analysis_script"] = repo_path(
    TURBULENCE_SPECTRA["flow_analysis_script"]
)
TURBULENCE_RESOLUTION = TURBULENCE["resolution"]
TURBULENCE_MB = TURBULENCE["meshblock"]
TURBULENCE_CASES = TURBULENCE["cases"]
TURBULENCE_CASE_NAMES = tuple(TURBULENCE_CASES)
TURBULENCE_RESTART_FROM = TURBULENCE["restart_from"]
TURBULENCE_OUTPUT_VARIABLES = output_variable_list(
    TURBULENCE, default=("prim", "acc")
)
TURBULENCE_SPECTRUM_DUMP_IDS = tuple(
    sorted(set(int(dump_id) for dump_id in TURBULENCE_SPECTRA["dump_ids"]))
)
TURBULENCE_SPECTRUM_DUMPS = tuple(
    f"{dump_id:05d}" for dump_id in TURBULENCE_SPECTRUM_DUMP_IDS
)

if len(TURBULENCE_RESOLUTION) != 3 or len(TURBULENCE_MB) != 3:
    raise ValueError("turbulence resolution and meshblock must have three entries")
if TURBULENCE["enabled"] and config["dimension"] != "3D":
    raise ValueError("the native turbulence workflow test is three-dimensional")
if not TURBULENCE_CASE_NAMES:
    raise ValueError("turbulence must define at least one numerical case")
unknown_restart_cases = set(TURBULENCE_RESTART_FROM).difference(TURBULENCE_CASE_NAMES)
if unknown_restart_cases:
    raise ValueError(
        "turbulence restart_from contains unknown cases: "
        f"{sorted(unknown_restart_cases)}"
    )
if TURBULENCE_SPECTRA["enabled"] and not TURBULENCE_SPECTRUM_DUMP_IDS:
    raise ValueError("enabled turbulence spectra require at least one dump ID")
if TURBULENCE_SPECTRA["analysis_tasks"] < 1:
    raise ValueError("turbulence spectra analysis_tasks must be positive")
if TURBULENCE_SPECTRA["analysis_nodes"] < 1:
    raise ValueError("turbulence spectra analysis_nodes must be positive")
if TURBULENCE_SPECTRA["analysis_tasks"] < TURBULENCE_SPECTRA["analysis_nodes"]:
    raise ValueError("turbulence spectra analysis_tasks must be >= analysis_nodes")
if TURBULENCE["output_dt"] <= 0.0:
    raise ValueError("turbulence output_dt must be positive")
if TURBULENCE_SPECTRA["enabled"] and (
    TURBULENCE_SPECTRUM_DUMP_IDS[-1] * TURBULENCE["output_dt"]
    > TURBULENCE["tlim"] + 1.0e-12
):
    raise ValueError("a requested turbulence spectrum dump occurs after tlim")
for case_name, case in TURBULENCE_CASES.items():
    required = {
        "label",
        "fluid",
        "riemann",
        "reconstruction",
        "integrator",
        "convergence_order",
        "first_order_flux_correct",
    }
    missing = required.difference(case)
    if missing:
        raise ValueError(f"turbulence case {case_name} is missing {sorted(missing)}")

TURBULENCE_NUM_BLOCKS = (
    ((TURBULENCE_RESOLUTION[0] + TURBULENCE_MB[0] - 1) // TURBULENCE_MB[0])
    * ((TURBULENCE_RESOLUTION[1] + TURBULENCE_MB[1] - 1) // TURBULENCE_MB[1])
    * ((TURBULENCE_RESOLUTION[2] + TURBULENCE_MB[2] - 1) // TURBULENCE_MB[2])
)
TURBULENCE_MPI_TASKS = min(TURBULENCE["mpi_tasks"], TURBULENCE_NUM_BLOCKS)
if TURBULENCE["nodes"] < 1 or TURBULENCE["tasks_per_node"] < 1:
    raise ValueError("turbulence nodes and tasks_per_node must be positive")
if TURBULENCE_MPI_TASKS != TURBULENCE["nodes"] * TURBULENCE["tasks_per_node"]:
    raise ValueError(
        "turbulence mpi_tasks must equal nodes * tasks_per_node and cannot exceed "
        "the number of meshblocks"
    )


def turbulence_out():
    return (
        f"{config['results_root']}/{config['dimension']}/"
        f"{TURBULENCE['output_group']}/{TURBULENCE['dirname']}"
    )


def turbulence_run_out(case):
    return f"{turbulence_out()}/{case}"


def turbulence_problem_id(case):
    return (
        f"turbulence_{case}_"
        f"Nx{TURBULENCE_RESOLUTION[0]}x{TURBULENCE_RESOLUTION[1]}x"
        f"{TURBULENCE_RESOLUTION[2]}"
    )


def turbulence_spectrum_out(case):
    return f"{turbulence_run_out(case)}/spectra"

def turbulence_restart_input(wildcards):
    restart = TURBULENCE_RESTART_FROM.get(wildcards.case)
    return [restart] if restart else []


turbulence_targets = []
if TURBULENCE["enabled"]:
    turbulence_targets = [
        f"{turbulence_out()}/turbulence_mean_energies.png",
        f"{turbulence_out()}/turbulence_summary.txt",
        f"{turbulence_out()}/turbulence_density_final.png",
        f"{turbulence_out()}/turbulence_Bmag_final.png",
        f"{turbulence_out()}/turbulence_velocity_final.png",
    ]
    if TURBULENCE_SPECTRA["enabled"]:
        turbulence_targets.extend(
            [
                f"{turbulence_out()}/turbulence_kinetic_power_spectra.png",
                f"{turbulence_out()}/turbulence_magnetic_power_spectra.png",
            ]
        )


rule run_turbulence:
    input:
        exe=config["athenapk_mpi_hdf5"],
        deck=TURBULENCE["input"],
        restart=turbulence_restart_input
    output:
        done=f"{turbulence_out()}/{{case}}/phdf-files/run.done"
    log:
        out=f"{turbulence_out()}/{{case}}/run.out",
        err=f"{turbulence_out()}/{{case}}/run.err"
    wildcard_constraints:
        case="|".join(re.escape(name) for name in TURBULENCE_CASE_NAMES)
    envmodules:
        "foss/2023a",
        "HDF5/1.14.0-gompi-2023a"
    params:
        rundir=lambda wc: f"{turbulence_run_out(wc.case)}/phdf-files",
        problem_id=lambda wc: turbulence_problem_id(wc.case),
        fluid=lambda wc: TURBULENCE_CASES[wc.case]["fluid"],
        riemann=lambda wc: TURBULENCE_CASES[wc.case]["riemann"],
        reconstruction=lambda wc: TURBULENCE_CASES[wc.case]["reconstruction"],
        integrator=lambda wc: TURBULENCE_CASES[wc.case]["integrator"],
        convergence_order=lambda wc: TURBULENCE_CASES[wc.case]["convergence_order"],
        first_order_flux_correct=lambda wc: str(
            TURBULENCE_CASES[wc.case]["first_order_flux_correct"]
        ).lower(),
        nghost=lambda wc: TURBULENCE_CASES[wc.case].get(
            "nghost",
            3
            if TURBULENCE_CASES[wc.case]["reconstruction"]
            in ("ppm", "wenoz", "weno5js")
            else 2,
        ),
        case_hydro_options=lambda wc: " ".join(
            option
            for option in (
                f"hydro/convergence_order="
                f"{TURBULENCE_CASES[wc.case]['convergence_order']}",
                hydro_cli_options(
                    TURBULENCE_CASES[wc.case], include_global=False
                ),
            )
            if option
        ),
        nx1=TURBULENCE_RESOLUTION[0],
        nx2=TURBULENCE_RESOLUTION[1],
        nx3=TURBULENCE_RESOLUTION[2],
        mb_nx1=TURBULENCE_MB[0],
        mb_nx2=TURBULENCE_MB[1],
        mb_nx3=TURBULENCE_MB[2],
        tlim=TURBULENCE["tlim"],
        history_dt=TURBULENCE["history_dt"],
        output_dt=TURBULENCE["output_dt"],
        mpi_tasks=TURBULENCE_MPI_TASKS,
        restart_arg=lambda wc: (
            f"-r {shlex.quote(TURBULENCE_RESTART_FROM[wc.case])}"
            if TURBULENCE_RESTART_FROM.get(wc.case)
            else ""
        )
    resources:
        runtime=4320,
        nodes=TURBULENCE["nodes"],
        tasks_per_node=TURBULENCE["tasks_per_node"],
        mpi="srun",
        mem_mb_per_cpu=TURBULENCE["mem_mb_per_cpu"]
    shell:
        """
        mkdir -p {params.rundir}
        if [[ -z "{params.restart_arg}" ]]; then
          rm -f {params.rundir}/*.phdf
          rm -f {params.rundir}/*.phdf.xdmf
          rm -f {params.rundir}/*.hst
          rm -f {params.rundir}/*.rhdf
          : > {log.out}
          : > {log.err}
        fi
        rm -f {output.done}
        cd {params.rundir}

        {resources.mpi} -n {params.mpi_tasks} {input.exe} \
          {params.restart_arg} -i {input.deck} \
          job/problem_id=turbulence \
          parthenon/job/problem_id={params.problem_id} \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3={params.nx3} \
          parthenon/meshblock/nx1={params.mb_nx1} \
          parthenon/meshblock/nx2={params.mb_nx2} \
          parthenon/meshblock/nx3={params.mb_nx3} \
          parthenon/mesh/nghost={params.nghost} \
          parthenon/mesh/packs_per_rank=1 \
          parthenon/time/tlim={params.tlim} \
          parthenon/time/integrator={params.integrator} \
          hydro/fluid={params.fluid} \
          hydro/riemann={params.riemann} \
          hydro/reconstruction={params.reconstruction} \
          hydro/first_order_flux_correct={params.first_order_flux_correct} \
          {params.case_hydro_options} \
          parthenon/output1/file_type=hst \
          parthenon/output1/dt={params.history_dt} \
          parthenon/output2/file_type=hdf5 \
          parthenon/output2/dt={params.output_dt} \
          parthenon/output2/variables={TURBULENCE_OUTPUT_VARIABLES} \
          parthenon/output2/include_in_final=true \
          parthenon/output2/use_final_label=false \
          >> {log.out} 2>> {log.err}

        touch {output.done}
        """


rule compute_turbulence_flow_analysis:
    input:
        done=f"{turbulence_out()}/{{case}}/phdf-files/run.done",
        analysis=TURBULENCE_SPECTRA["flow_analysis_script"],
        modules=[
            repo_path(
                "workflow/diagnostics_scripts/turbulence/energy_transfer_analysis/"
                + filename
            )
            for filename in (
                "EnergyTransfer.py",
                "FFTHelperFuncs.py",
                "FlowAnalysis.py",
                "IOhelperFuncs.py",
                "MPIderivHelperFuncs.py",
            )
        ]
    output:
        flow=f"{turbulence_out()}/{{case}}/spectra/flow-{{dump}}-out.hdf5"
    log:
        out=f"{turbulence_out()}/{{case}}/spectra/flow-{{dump}}.out",
        err=f"{turbulence_out()}/{{case}}/spectra/flow-{{dump}}.err"
    wildcard_constraints:
        case="|".join(re.escape(name) for name in TURBULENCE_CASE_NAMES),
        dump="|".join(TURBULENCE_SPECTRUM_DUMPS)
    params:
        outdir=lambda wc: turbulence_spectrum_out(wc.case),
        analysis_tasks=TURBULENCE_SPECTRA["analysis_tasks"],
        snapshot=lambda wc: (
            f"{turbulence_run_out(wc.case)}/phdf-files/"
            f"{turbulence_problem_id(wc.case)}.prim.{wc.dump}.phdf"
        ),
        kernel_args=(
            "--kernels " + " ".join(TURBULENCE_SPECTRA["kernels"])
            if TURBULENCE_SPECTRA["kernels"]
            else ""
        )
    resources:
        runtime=TURBULENCE_SPECTRA["runtime"],
        nodes=TURBULENCE_SPECTRA["analysis_nodes"],
        tasks=TURBULENCE_SPECTRA["analysis_tasks"],
        mpi="srun",
        mem_mb_per_cpu=TURBULENCE_SPECTRA["mem_mb_per_cpu"],
        turbulence_spectra_slots=1
    shell:
        """
        test -f {params.snapshot}
        mkdir -p {params.outdir}
        {resources.mpi} -n {params.analysis_tasks} \
          {TURBULENCE_SPECTRA[python]} {input.analysis} \
          --res {TURBULENCE_RESOLUTION[0]} \
          --data_path {params.snapshot} \
          --data_type AthenaPK \
          {params.kernel_args} \
          --type flow \
          --outfile {output.flow} \
          --eos adiabatic \
          --gamma 1.0001 \
          -forced -b \
          > {log.out} 2> {log.err}
        """


rule plot_turbulence_power_spectra:
    input:
        flows=expand(
            f"{turbulence_out()}/{{case}}/spectra/flow-{{dump}}-out.hdf5",
            case=TURBULENCE_CASE_NAMES,
            dump=TURBULENCE_SPECTRUM_DUMPS,
        ),
        script=TURBULENCE["spectrum_plotting_script"]
    output:
        kinetic=report(
            f"{turbulence_out()}/turbulence_kinetic_power_spectra.png",
            caption="../report/turbulence.rst",
            category="3D Tests",
            subcategory="Driven MHD Turbulence / power spectra",
            labels={"quantity": "compensated kinetic power spectrum"},
        ),
        magnetic=report(
            f"{turbulence_out()}/turbulence_magnetic_power_spectra.png",
            caption="../report/turbulence.rst",
            category="3D Tests",
            subcategory="Driven MHD Turbulence / power spectra",
            labels={"quantity": "compensated magnetic power spectrum"},
        )
    params:
        case_args=lambda wc: " ".join(
            "--case "
            + shlex.quote(
                f"{TURBULENCE_CASES[name]['label']}="
                f"{turbulence_spectrum_out(name)}"
            )
            for name in TURBULENCE_CASE_NAMES
        ),
        dump_args=" ".join(
            f"--dump-id {dump_id}" for dump_id in TURBULENCE_SPECTRUM_DUMP_IDS
        )
    resources:
        runtime=30,
        mem_mb=2000
    shell:
        """
        {config[plotting_python]} {input.script} \
          {params.case_args} \
          {params.dump_args} \
          --resolution {TURBULENCE_RESOLUTION[0]} \
          --kinetic {output.kinetic} \
          --magnetic {output.magnetic}
        """


rule plot_turbulence_energies:
    input:
        done=expand(
            f"{turbulence_out()}/{{case}}/phdf-files/run.done",
            case=TURBULENCE_CASE_NAMES,
        )
    output:
        plot=report(
            f"{turbulence_out()}/turbulence_mean_energies.png",
            caption="../report/turbulence.rst",
            category="3D Tests",
            subcategory="Driven MHD Turbulence / configured schemes",
            labels={"quantity": "mean kinetic and magnetic energy"},
        ),
        summary=report(
            f"{turbulence_out()}/turbulence_summary.txt",
            caption="../report/turbulence.rst",
            category="3D Tests",
            subcategory="Driven MHD Turbulence / configured schemes",
            labels={"quantity": "late-time summary"},
        )
    params:
        case_args=lambda wc: " ".join(
            "--case "
            + shlex.quote(
                f"{TURBULENCE_CASES[name]['label']}="
                f"{turbulence_run_out(name)}/phdf-files"
            )
            for name in TURBULENCE_CASE_NAMES
        )
    resources:
        runtime=30,
        mem_mb=2000
    shell:
        """
        {config[plotting_python]} {TURBULENCE[energy_plotting_script]} \
          {params.case_args} \
          --plot {output.plot} \
          --summary {output.summary}
        """


rule plot_turbulence_slices:
    input:
        done=expand(
            f"{turbulence_out()}/{{case}}/phdf-files/run.done",
            case=TURBULENCE_CASE_NAMES,
        )
    output:
        density=report(
            f"{turbulence_out()}/turbulence_density_final.png",
            caption="../report/turbulence.rst",
            category="3D Tests",
            subcategory="Driven MHD Turbulence / configured schemes",
            labels={"quantity": "density", "slice": "z = 0.5"},
        ),
        bmag=report(
            f"{turbulence_out()}/turbulence_Bmag_final.png",
            caption="../report/turbulence.rst",
            category="3D Tests",
            subcategory="Driven MHD Turbulence / configured schemes",
            labels={"quantity": "magnetic-field magnitude", "slice": "z = 0.5"},
        ),
        velocity=report(
            f"{turbulence_out()}/turbulence_velocity_final.png",
            caption="../report/turbulence.rst",
            category="3D Tests",
            subcategory="Driven MHD Turbulence / configured schemes",
            labels={"quantity": "velocity magnitude", "slice": "z = 0.5"},
        )
    params:
        case_args=lambda wc: " ".join(
            "--case "
            + shlex.quote(
                f"{TURBULENCE_CASES[name]['label']}="
                f"{turbulence_run_out(name)}/phdf-files"
            )
            for name in TURBULENCE_CASE_NAMES
        ),
        expected_final_time=TURBULENCE["tlim"]
    resources:
        runtime=60,
        mem_mb=16000
    shell:
        """
        {config[plotting_python]} {TURBULENCE[slice_plotting_script]} \
          {params.case_args} \
          --density {output.density} \
          --bmag {output.bmag} \
          --velocity {output.velocity} \
          --expected-final-time {params.expected_final_time}
        """
