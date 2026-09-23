ORSZAG_TANG = config["tests"]["orszag_tang"]
ORSZAG_TANG_RESOLUTION = ORSZAG_TANG["resolution"]
ORSZAG_TANG_MB = ORSZAG_TANG["meshblock"]
ORSZAG_TANG_HYDRO_OPTIONS = hydro_cli_options(ORSZAG_TANG)
ORSZAG_TANG_OUTPUT_VARIABLES = output_variable_list(ORSZAG_TANG)
ORSZAG_TANG_REQUESTED_MPI_TASKS = int(ORSZAG_TANG.get("mpi_tasks", 1))
ORSZAG_TANG_NODES = int(ORSZAG_TANG.get("nodes", 1))
ORSZAG_TANG_MEM_MB_PER_CPU = int(ORSZAG_TANG.get("mem_mb_per_cpu", 10000))
ORSZAG_TANG_RUNTIME = int(ORSZAG_TANG.get("runtime", 120))
ORSZAG_TANG_DIAGNOSTIC_MEM_MB = int(
    ORSZAG_TANG.get("diagnostic_mem_mb", 4000)
)
ORSZAG_TANG_DIAGNOSTIC_RUNTIME = int(
    ORSZAG_TANG.get("diagnostic_runtime", 60)
)
ORSZAG_TANG_CFL = float(ORSZAG_TANG.get("cfl", 0.3))
ORSZAG_TANG_TLIM = float(ORSZAG_TANG.get("tlim", 1.0))
ORSZAG_TANG_OUTPUT_DT = float(ORSZAG_TANG.get("output_dt", 0.01))
ORSZAG_TANG_LAUNCHER_MODE = ORSZAG_TANG.get("launcher", "srun")

if len(ORSZAG_TANG_RESOLUTION) != 2 or len(ORSZAG_TANG_MB) != 2:
    raise ValueError("orszag_tang resolution and meshblock must have two entries")
if any(int(n) <= 0 for n in ORSZAG_TANG_RESOLUTION + ORSZAG_TANG_MB):
    raise ValueError("orszag_tang resolution and meshblock entries must be positive")
if any(
    int(global_n) % int(block_n) != 0
    for global_n, block_n in zip(ORSZAG_TANG_RESOLUTION, ORSZAG_TANG_MB)
):
    raise ValueError(
        f"orszag_tang grid {ORSZAG_TANG_RESOLUTION} is not divisible by "
        f"meshblock {ORSZAG_TANG_MB}"
    )
if ORSZAG_TANG_REQUESTED_MPI_TASKS <= 0:
    raise ValueError("orszag_tang mpi_tasks must be positive")
if ORSZAG_TANG_NODES <= 0:
    raise ValueError("orszag_tang nodes must be positive")
if ORSZAG_TANG_MEM_MB_PER_CPU <= 0:
    raise ValueError("orszag_tang mem_mb_per_cpu must be positive")
if ORSZAG_TANG_RUNTIME <= 0:
    raise ValueError("orszag_tang runtime must be positive")
if ORSZAG_TANG_DIAGNOSTIC_MEM_MB <= 0:
    raise ValueError("orszag_tang diagnostic_mem_mb must be positive")
if ORSZAG_TANG_DIAGNOSTIC_RUNTIME <= 0:
    raise ValueError("orszag_tang diagnostic_runtime must be positive")
if ORSZAG_TANG_CFL <= 0.0:
    raise ValueError("orszag_tang cfl must be positive")
if ORSZAG_TANG_TLIM <= 0.0:
    raise ValueError("orszag_tang tlim must be positive")
if ORSZAG_TANG_OUTPUT_DT <= 0.0:
    raise ValueError("orszag_tang output_dt must be positive")
if ORSZAG_TANG_LAUNCHER_MODE not in ("direct", "srun"):
    raise ValueError("orszag_tang launcher must be 'direct' or 'srun'")
if ORSZAG_TANG["enabled"] and config["dimension"] != "2D":
    raise ValueError("orszag_tang is a two-dimensional test")

ORSZAG_TANG_NUM_BLOCKS = (
    (int(ORSZAG_TANG_RESOLUTION[0]) // int(ORSZAG_TANG_MB[0]))
    * (int(ORSZAG_TANG_RESOLUTION[1]) // int(ORSZAG_TANG_MB[1]))
)
ORSZAG_TANG_MPI_TASKS = min(
    ORSZAG_TANG_REQUESTED_MPI_TASKS, ORSZAG_TANG_NUM_BLOCKS
)
if ORSZAG_TANG_LAUNCHER_MODE == "direct":
    if ORSZAG_TANG_MPI_TASKS != 1:
        raise ValueError("orszag_tang launcher=direct requires mpi_tasks=1")
    ORSZAG_TANG_MPI_RESOURCE = False
    ORSZAG_TANG_LAUNCHER = ""
else:
    ORSZAG_TANG_MPI_RESOURCE = "srun" if ORSZAG_TANG_MPI_TASKS > 1 else False
    ORSZAG_TANG_LAUNCHER = f"srun -n {ORSZAG_TANG_MPI_TASKS}"


def orszag_tang_out(fluid):
    return f"{config['results_root']}/{config['dimension']}/{fluid}/{ORSZAG_TANG['dirname']}"

orszag_tang_targets = []
if ORSZAG_TANG["enabled"]:
    orszag_tang_targets = expand(
        "{outdir}/{name}",
        outdir=[orszag_tang_out(fluid) for fluid in config["fluids"]],
        name=[
            "orszag_tang_pressure.gif",
            "orszag_tang_pressure_final.png",
            "orszag_tang_Bmag.gif",
            "orszag_tang_pressure_rot180_abs.gif",
        ],
    )

rule run_orszag_tang:
    input:
        exe=config["athenapk_mpi_hdf5"],
        deck=ORSZAG_TANG["input"]
    output:
        done=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG['dirname']}/phdf-files/run.done"
    log:
        out=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG['dirname']}/run.out",
        err=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG['dirname']}/run.err"
    envmodules:
        *SIMULATION_ENVMODULES
    params:
        outdir=lambda wc: orszag_tang_out(wc.fluid),
        rundir=lambda wc: f"{orszag_tang_out(wc.fluid)}/phdf-files",
        problem_id=lambda wc: f"orszag_tang_{wc.fluid}_Nx{ORSZAG_TANG_RESOLUTION[0]}x{ORSZAG_TANG_RESOLUTION[1]}",
        nx1=ORSZAG_TANG_RESOLUTION[0],
        nx2=ORSZAG_TANG_RESOLUTION[1],
        nx1_mb=ORSZAG_TANG_MB[0],
        nx2_mb=ORSZAG_TANG_MB[1],
        launcher=ORSZAG_TANG_LAUNCHER,
        cfl=ORSZAG_TANG_CFL,
        tlim=ORSZAG_TANG_TLIM,
        output_dt=ORSZAG_TANG_OUTPUT_DT,
    resources:
        runtime=ORSZAG_TANG_RUNTIME,
        nodes=ORSZAG_TANG_NODES,
        tasks=ORSZAG_TANG_MPI_TASKS,
        mpi=ORSZAG_TANG_MPI_RESOURCE,
        mem_mb_per_cpu=ORSZAG_TANG_MEM_MB_PER_CPU,
        gpu=SIMULATION_GPU_COUNT,
        tasks_per_gpu=SIMULATION_TASKS_PER_GPU,
        cpus_per_gpu=SIMULATION_CPUS_PER_GPU,
        constraint=SIMULATION_CONSTRAINT,
        slurm_extra=SIMULATION_SLURM_EXTRA
    shell:
        """
        mkdir -p {params.rundir}
        rm -f {params.rundir}/*.phdf
        rm -f {params.rundir}/*.phdf.xdmf
        rm -f {params.rundir}/*.hst
        rm -f {output.done}
        cd {params.rundir}

        {params.launcher} {input.exe} -i {input.deck} \
          parthenon/job/problem_id={params.problem_id} \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3=1 \
          parthenon/mesh/nghost={NGHOST} \
          parthenon/meshblock/nx1={params.nx1_mb} \
          parthenon/meshblock/nx2={params.nx2_mb} \
          parthenon/meshblock/nx3=1 \
          parthenon/time/tlim={params.tlim} \
          parthenon/time/cfl={params.cfl} \
          parthenon/time/integrator={config[integrator]} \
          hydro/fluid={wildcards.fluid} \
          hydro/riemann={config[riemann]} \
          hydro/reconstruction={config[reconstruction]} \
          hydro/convergence_order={config[convergence_order]} \
          {ORSZAG_TANG_HYDRO_OPTIONS} \
          hydro/gamma=1.666666666666667 \
          hydro/scratch_level=1 \
          hydro/pfloor=1e-15 \
          parthenon/output0/file_type=hdf5 \
          parthenon/output0/dt={params.output_dt} \
          parthenon/output0/variables={ORSZAG_TANG_OUTPUT_VARIABLES} \
          > {log.out} 2> {log.err}
    
        touch {output.done}
        """

rule make_orszag_tang_videos:
    input:
        done=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG['dirname']}/phdf-files/run.done"
    output:
        pressure=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG['dirname']}/orszag_tang_pressure.gif",
            caption="../report/orszag_tang_pressure.rst",
            category="2D Tests",
            subcategory="{fluid} / Orszag-Tang",
            labels={"fluid": "{fluid}", "quantity": "pressure"}
        ),
        bmag=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG['dirname']}/orszag_tang_Bmag.gif",
            caption="../report/orszag_tang_Bmag.rst",
            category="2D Tests",
            subcategory="{fluid} / Orszag-Tang",
            labels={"fluid": "{fluid}", "quantity": "magnetic field magnitude"}
        ),
        pressure_rot180_abs=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG['dirname']}/orszag_tang_pressure_rot180_abs.gif",
            caption="../report/orszag_tang_pressure_rot180_abs.rst",
            category="2D Tests",
            subcategory="{fluid} / Orszag-Tang",
            labels={"fluid": "{fluid}", "quantity": "180 degree pressure symmetry error"}
        )
    params:
        outdir=lambda wc: orszag_tang_out(wc.fluid),
        phdf=lambda wc: f"{orszag_tang_out(wc.fluid)}/phdf-files",
    resources:
        runtime=ORSZAG_TANG_DIAGNOSTIC_RUNTIME,
        mem_mb=ORSZAG_TANG_DIAGNOSTIC_MEM_MB
    shell:
        """
        cd {params.outdir}
        {config[plotting_python]} {ORSZAG_TANG[plotting_script]} \
          {params.phdf} \
          -o orszag_tang
        """

rule make_orszag_tang_final_plot:
    input:
        done=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG['dirname']}/phdf-files/run.done"
    output:
        pressure=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{ORSZAG_TANG['dirname']}/orszag_tang_pressure_final.png",
            caption="../report/orszag_tang_pressure.rst",
            category="2D Tests",
            subcategory="{fluid} / Orszag-Tang",
            labels={
                "fluid": "{fluid}",
                "quantity": "pressure",
                "snapshot": "final time",
            }
        )
    params:
        outdir=lambda wc: orszag_tang_out(wc.fluid),
        phdf=lambda wc: f"{orszag_tang_out(wc.fluid)}/phdf-files",
    resources:
        runtime=ORSZAG_TANG_DIAGNOSTIC_RUNTIME,
        mem_mb=ORSZAG_TANG_DIAGNOSTIC_MEM_MB
    shell:
        """
        cd {params.outdir}
        {config[plotting_python]} {ORSZAG_TANG[plotting_script]} \
          {params.phdf} \
          --final-only \
          --pressure-final {output.pressure}
        """
