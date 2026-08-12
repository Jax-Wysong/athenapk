TURBULENCE_DEFAULTS = {
    "enabled": False,
    "input": (
        "/mnt/home/wysongj2/athenapk-fourth-Berta24/inputs/"
        "turbulence_weno_compare.in"
    ),
    "dirname": "turbulence",
    # Keep the four-way comparison separate from legacy GLM-only results.
    "output_group": "mhd_comparison",
    "energy_plotting_script": (
        "/mnt/home/wysongj2/athenapk-fourth-Berta24/workflow/"
        "diagnostics_scripts/turbulence_energies.py"
    ),
    "slice_plotting_script": (
        "/mnt/home/wysongj2/athenapk-fourth-Berta24/workflow/"
        "diagnostics_scripts/turbulence_slices.py"
    ),
    "resolution": [128, 128, 128],
    "meshblock": [64, 64, 32],
    "tlim": 15.0,
    "history_dt": 0.005,
    "output_dt": 5.0,
    "cases": {
        "weno3": {
            "label": "GLM-HLLD WENO3 + VL2",
            "fluid": "glmmhd",
            "riemann": "hlld",
            "reconstruction": "weno3",
            "integrator": "vl2",
            "convergence_order": 2,
            "first_order_flux_correct": True,
        },
        "weno5": {
            "label": "GLM-HLLD WENO5 + VL2",
            "fluid": "glmmhd",
            "riemann": "hlld",
            "reconstruction": "wenoz",
            "integrator": "vl2",
            "convergence_order": 2,
            "first_order_flux_correct": True,
        },
        "uct2_weno3": {
            "label": "UCT-HLLD WENO3 + RK2",
            "fluid": "ucthlldmhd",
            "riemann": "hlld",
            "reconstruction": "weno3",
            "integrator": "rk2",
            "convergence_order": 2,
            "first_order_flux_correct": False,
        },
        "uct4_berta": {
            "label": "UCT-HLLD Berta4 + RK4",
            "fluid": "ucthlldmhd",
            "riemann": "hlld",
            "reconstruction": "wenoz_point",
            "integrator": "rk4",
            "convergence_order": 4,
            "first_order_flux_correct": False,
        },
    },
    "mpi_tasks": 16,
    "mem_mb_per_cpu": 4000,
}
TURBULENCE = {
    **TURBULENCE_DEFAULTS,
    **config["tests"].get("turbulence", {}),
}
TURBULENCE_RESOLUTION = TURBULENCE["resolution"]
TURBULENCE_MB = TURBULENCE["meshblock"]
TURBULENCE_CASES = TURBULENCE["cases"]
TURBULENCE_CASE_NAMES = (
    "weno3",
    "weno5",
    "uct2_weno3",
    "uct4_berta",
)

if len(TURBULENCE_RESOLUTION) != 3 or len(TURBULENCE_MB) != 3:
    raise ValueError("turbulence resolution and meshblock must have three entries")
if TURBULENCE["enabled"] and config["dimension"] != "3D":
    raise ValueError("the native turbulence workflow test is three-dimensional")
if set(TURBULENCE_CASES) != set(TURBULENCE_CASE_NAMES):
    raise ValueError(
        "turbulence cases must define weno3, weno5, "
        "uct2_weno3, and uct4_berta"
    )
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


def turbulence_out():
    return (
        f"{config['results_root']}/{config['dimension']}/"
        f"{TURBULENCE['output_group']}/{TURBULENCE['dirname']}"
    )


def turbulence_run_out(case):
    return f"{turbulence_out()}/{case}"


turbulence_targets = []
if TURBULENCE["enabled"]:
    turbulence_targets = [
        f"{turbulence_out()}/turbulence_mean_energies.png",
        f"{turbulence_out()}/turbulence_summary.txt",
        f"{turbulence_out()}/turbulence_density_final.png",
        f"{turbulence_out()}/turbulence_Bmag_final.png",
        f"{turbulence_out()}/turbulence_velocity_final.png",
    ]


rule run_turbulence:
    input:
        exe=config.get(
            "athenapk_mpi_hdf5",
            "/mnt/home/wysongj2/athenapk-fourth-Berta24/"
            "build-mpi-site-hdf5/bin/athenaPK",
        ),
        deck=TURBULENCE["input"]
    output:
        done=f"{turbulence_out()}/{{case}}/phdf-files/run.done"
    log:
        out=f"{turbulence_out()}/{{case}}/run.out",
        err=f"{turbulence_out()}/{{case}}/run.err"
    wildcard_constraints:
        case="|".join(TURBULENCE_CASE_NAMES)
    envmodules:
        "foss/2023a",
        "HDF5/1.14.0-gompi-2023a"
    params:
        rundir=lambda wc: f"{turbulence_run_out(wc.case)}/phdf-files",
        problem_id=lambda wc: (
            f"turbulence_{wc.case}_"
            f"Nx{TURBULENCE_RESOLUTION[0]}x{TURBULENCE_RESOLUTION[1]}x"
            f"{TURBULENCE_RESOLUTION[2]}"
        ),
        fluid=lambda wc: TURBULENCE_CASES[wc.case]["fluid"],
        riemann=lambda wc: TURBULENCE_CASES[wc.case]["riemann"],
        reconstruction=lambda wc: TURBULENCE_CASES[wc.case]["reconstruction"],
        integrator=lambda wc: TURBULENCE_CASES[wc.case]["integrator"],
        convergence_order=lambda wc: TURBULENCE_CASES[wc.case]["convergence_order"],
        first_order_flux_correct=lambda wc: str(
            TURBULENCE_CASES[wc.case]["first_order_flux_correct"]
        ).lower(),
        uct_options=lambda wc: (
            f"hydro/convergence_order={TURBULENCE_CASES[wc.case]['convergence_order']} "
            "hydro/discontinuity_detector=none"
            if TURBULENCE_CASES[wc.case]["fluid"] == "ucthlldmhd"
            else ""
        ),
        nx1=TURBULENCE_RESOLUTION[0],
        nx2=TURBULENCE_RESOLUTION[1],
        nx3=TURBULENCE_RESOLUTION[2],
        mb_nx1=TURBULENCE_MB[0],
        mb_nx2=TURBULENCE_MB[1],
        mb_nx3=TURBULENCE_MB[2],
        tlim=TURBULENCE["tlim"],
        history_dt=TURBULENCE["history_dt"],
        output_dt=TURBULENCE["output_dt"]
    resources:
        runtime=720,
        nodes=1,
        tasks=TURBULENCE_MPI_TASKS,
        mpi="srun",
        mem_mb_per_cpu=TURBULENCE["mem_mb_per_cpu"]
    shell:
        """
        mkdir -p {params.rundir}
        rm -f {params.rundir}/*.phdf
        rm -f {params.rundir}/*.phdf.xdmf
        rm -f {params.rundir}/*.hst
        rm -f {params.rundir}/*.rhdf
        rm -f {output.done}
        cd {params.rundir}

        {resources.mpi} -n {resources.tasks} {input.exe} -i {input.deck} \
          job/problem_id=turbulence \
          parthenon/job/problem_id={params.problem_id} \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3={params.nx3} \
          parthenon/meshblock/nx1={params.mb_nx1} \
          parthenon/meshblock/nx2={params.mb_nx2} \
          parthenon/meshblock/nx3={params.mb_nx3} \
          parthenon/mesh/packs_per_rank=1 \
          parthenon/time/tlim={params.tlim} \
          parthenon/time/integrator={params.integrator} \
          hydro/fluid={params.fluid} \
          hydro/riemann={params.riemann} \
          hydro/reconstruction={params.reconstruction} \
          hydro/first_order_flux_correct={params.first_order_flux_correct} \
          {params.uct_options} \
          parthenon/output1/file_type=hst \
          parthenon/output1/dt={params.history_dt} \
          parthenon/output2/file_type=hdf5 \
          parthenon/output2/dt={params.output_dt} \
          parthenon/output2/variables=prim,acc \
          parthenon/output2/include_in_final=true \
          parthenon/output2/use_final_label=true \
          > {log.out} 2> {log.err}

        touch {output.done}
        """


rule plot_turbulence_energies:
    input:
        glm_weno3_done=f"{turbulence_run_out('weno3')}/phdf-files/run.done",
        glm_weno5_done=f"{turbulence_run_out('weno5')}/phdf-files/run.done",
        uct2_done=f"{turbulence_run_out('uct2_weno3')}/phdf-files/run.done",
        uct4_done=f"{turbulence_run_out('uct4_berta')}/phdf-files/run.done"
    output:
        plot=report(
            f"{turbulence_out()}/turbulence_mean_energies.png",
            caption="../report/turbulence.rst",
            category="3D Tests",
            subcategory="Driven MHD Turbulence / GLM and UCT-HLLD",
            labels={"quantity": "mean kinetic and magnetic energy"},
        ),
        summary=report(
            f"{turbulence_out()}/turbulence_summary.txt",
            caption="../report/turbulence.rst",
            category="3D Tests",
            subcategory="Driven MHD Turbulence / GLM and UCT-HLLD",
            labels={"quantity": "late-time summary"},
        )
    params:
        glm_weno3_dir=f"{turbulence_run_out('weno3')}/phdf-files",
        glm_weno5_dir=f"{turbulence_run_out('weno5')}/phdf-files",
        uct2_dir=f"{turbulence_run_out('uct2_weno3')}/phdf-files",
        uct4_dir=f"{turbulence_run_out('uct4_berta')}/phdf-files"
    resources:
        runtime=30,
        mem_mb=2000
    shell:
        """
        {config[plotting_python]} {TURBULENCE[energy_plotting_script]} \
          --glm-weno3-dir {params.glm_weno3_dir} \
          --glm-weno5-dir {params.glm_weno5_dir} \
          --uct2-dir {params.uct2_dir} \
          --uct4-dir {params.uct4_dir} \
          --plot {output.plot} \
          --summary {output.summary}
        """


rule plot_turbulence_slices:
    input:
        glm_weno3_done=f"{turbulence_run_out('weno3')}/phdf-files/run.done",
        glm_weno5_done=f"{turbulence_run_out('weno5')}/phdf-files/run.done",
        uct2_done=f"{turbulence_run_out('uct2_weno3')}/phdf-files/run.done",
        uct4_done=f"{turbulence_run_out('uct4_berta')}/phdf-files/run.done"
    output:
        density=report(
            f"{turbulence_out()}/turbulence_density_final.png",
            caption="../report/turbulence.rst",
            category="3D Tests",
            subcategory="Driven MHD Turbulence / GLM and UCT-HLLD",
            labels={"quantity": "density", "slice": "z = 0.5"},
        ),
        bmag=report(
            f"{turbulence_out()}/turbulence_Bmag_final.png",
            caption="../report/turbulence.rst",
            category="3D Tests",
            subcategory="Driven MHD Turbulence / GLM and UCT-HLLD",
            labels={"quantity": "magnetic-field magnitude", "slice": "z = 0.5"},
        ),
        velocity=report(
            f"{turbulence_out()}/turbulence_velocity_final.png",
            caption="../report/turbulence.rst",
            category="3D Tests",
            subcategory="Driven MHD Turbulence / GLM and UCT-HLLD",
            labels={"quantity": "velocity magnitude", "slice": "z = 0.5"},
        )
    params:
        glm_weno3_dir=f"{turbulence_run_out('weno3')}/phdf-files",
        glm_weno5_dir=f"{turbulence_run_out('weno5')}/phdf-files",
        uct2_dir=f"{turbulence_run_out('uct2_weno3')}/phdf-files",
        uct4_dir=f"{turbulence_run_out('uct4_berta')}/phdf-files",
        expected_final_time=TURBULENCE["tlim"]
    resources:
        runtime=60,
        mem_mb=16000
    shell:
        """
        {config[plotting_python]} {TURBULENCE[slice_plotting_script]} \
          --glm-weno3-dir {params.glm_weno3_dir} \
          --glm-weno5-dir {params.glm_weno5_dir} \
          --uct2-dir {params.uct2_dir} \
          --uct4-dir {params.uct4_dir} \
          --density {output.density} \
          --bmag {output.bmag} \
          --velocity {output.velocity} \
          --expected-final-time {params.expected_final_time}
        """
