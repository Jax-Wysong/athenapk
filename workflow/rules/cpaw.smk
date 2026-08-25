CPAW = config["tests"]["cpaw"]
CPAW_RESOLUTIONS = CPAW["resolutions"]
CPAW_HYDRO_OPTIONS = hydro_cli_options(CPAW)
CPAW_OUTPUT_VARIABLES = output_variable_list(CPAW)
CPAW_DIMENSION = config["dimension"]
CPAW_MESH = CPAW.get("mesh", {}).get(CPAW_DIMENSION, {})
CPAW_CONFIGURED_MB = CPAW_MESH.get("meshblock")
CPAW_REQUESTED_MPI_TASKS = int(CPAW_MESH.get("mpi_tasks", 1))
CPAW_MEM_MB_PER_CPU = int(CPAW_MESH.get("mem_mb_per_cpu", 4000))

if CPAW_DIMENSION not in ("2D", "3D"):
    raise ValueError("cpaw supports only dimension: 2D or 3D")
if CPAW_CONFIGURED_MB is not None:
    if len(CPAW_CONFIGURED_MB) != 3:
        raise ValueError("cpaw meshblock must have three entries")
    if any(int(n) <= 0 for n in CPAW_CONFIGURED_MB):
        raise ValueError("cpaw meshblock entries must be positive")
if CPAW_REQUESTED_MPI_TASKS <= 0:
    raise ValueError("cpaw mpi_tasks must be positive")
if CPAW_MEM_MB_PER_CPU <= 0:
    raise ValueError("cpaw mem_mb_per_cpu must be positive")


def cpaw_grid_shape(n):
    n = int(n)
    return (2 * n, n, 1 if CPAW_DIMENSION == "2D" else n)


def cpaw_meshblock_shape(n):
    if CPAW_CONFIGURED_MB is not None:
        meshblock = tuple(int(value) for value in CPAW_CONFIGURED_MB)
    else:
        # Preserve the legacy CPAW layout for configs that do not yet provide
        # explicit dimension-specific resource settings.
        n = int(n)
        meshblock = (2 * (n // 4), n // 2, 1 if CPAW_DIMENSION == "2D" else n // 2)

    grid = cpaw_grid_shape(n)
    if any(global_n % block_n != 0 for global_n, block_n in zip(grid, meshblock)):
        raise ValueError(
            f"cpaw {CPAW_DIMENSION} grid {grid} is not divisible by "
            f"meshblock {meshblock} at N={n}"
        )
    return meshblock


def cpaw_num_blocks(n):
    grid = cpaw_grid_shape(n)
    meshblock = cpaw_meshblock_shape(n)
    return (
        (grid[0] // meshblock[0])
        * (grid[1] // meshblock[1])
        * (grid[2] // meshblock[2])
    )


def cpaw_mpi_tasks(wildcards):
    return min(CPAW_REQUESTED_MPI_TASKS, cpaw_num_blocks(wildcards.N))


def cpaw_mpi_resource(wildcards):
    # The Slurm executor rejects mpi="srun" when tasks=1. The shell command
    # still explicitly launches that singleton with srun below.
    return "srun" if cpaw_mpi_tasks(wildcards) > 1 else False


def cpaw_launcher(wildcards):
    tasks = cpaw_mpi_tasks(wildcards)
    # This executable is MPI-enabled even for a one-block run. Keep an explicit
    # launcher for one rank as well; otherwise Snakemake's Slurm jobstep directly
    # launches the binary in an srun allocation without the site's MPI setup.
    return f"srun -n {tasks}"


def cpaw_out(fluid):
    return f"{config['results_root']}/{config['dimension']}/{fluid}/{CPAW['dirname']}"

cpaw_targets = []
if CPAW["enabled"]:
    cpaw_targets = expand(
        "{outdir}/cpaw_orders.txt",
        outdir=[cpaw_out(fluid) for fluid in config["fluids"]],
    )


rule run_cpaw:
    input:
        exe=config["athenapk_mpi"],
        deck=CPAW["input"]
    output:
        dat=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CPAW['dirname']}/runs/N{{N}}/cpaw-errors.dat"
    log:
        out=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CPAW['dirname']}/runs/N{{N}}/run.out",
        err=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CPAW['dirname']}/runs/N{{N}}/run.err"
    envmodules:
        *SIMULATION_ENVMODULES
    params:
        outdir=lambda wildcards: cpaw_out(wildcards.fluid),
        rundir=lambda wildcards: f"{cpaw_out(wildcards.fluid)}/runs/N{wildcards.N}",
        problem_id=lambda wildcards: f"cpaw_{wildcards.fluid}_N{wildcards.N}",
        nx1=lambda wildcards: 2 * int(wildcards.N),
        nx2=lambda wildcards: int(wildcards.N),
        nx3=lambda wc: 1 if config["dimension"] == "2D" else int(wc.N),
        ang_2=lambda wc: CPAW_MESH.get("ang_2", 0.0) if CPAW_DIMENSION == "2D" else -999.9,
        nx1_mb=lambda wc: cpaw_meshblock_shape(wc.N)[0],
        nx2_mb=lambda wc: cpaw_meshblock_shape(wc.N)[1],
        nx3_mb=lambda wc: cpaw_meshblock_shape(wc.N)[2],
        launcher=cpaw_launcher
    resources:
        runtime=120,
        nodes=1,
        tasks=cpaw_mpi_tasks,
        mpi=cpaw_mpi_resource,
        mem_mb_per_cpu=CPAW_MEM_MB_PER_CPU,
        gpu=SIMULATION_GPU_COUNT,
        tasks_per_gpu=SIMULATION_TASKS_PER_GPU,
        cpus_per_gpu=SIMULATION_CPUS_PER_GPU,
        constraint=SIMULATION_CONSTRAINT,
        slurm_extra=SIMULATION_SLURM_EXTRA
    shell:
        """
        mkdir -p {params.rundir}
        cd {params.rundir}
        rm -f {output.dat}

        {params.launcher} {input.exe} -i {input.deck} \
          parthenon/job/problem_id={params.problem_id} \
          problem/cpaw/compute_error=true \
          problem/cpaw/ang_2={params.ang_2} \
          problem/cpaw/v_par=1.0 \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3={params.nx3} \
          parthenon/mesh/nghost=3 \
          parthenon/meshblock/nx1={params.nx1_mb} \
          parthenon/meshblock/nx2={params.nx2_mb} \
          parthenon/meshblock/nx3={params.nx3_mb} \
          parthenon/time/tlim=1.0 \
          parthenon/time/cfl=0.3 \
          parthenon/time/integrator={config[integrator]} \
          hydro/fluid={wildcards.fluid} \
          hydro/riemann={config[riemann]} \
          hydro/reconstruction={config[reconstruction]} \
          hydro/convergence_order={config[convergence_order]} \
          {CPAW_HYDRO_OPTIONS} \
          hydro/gamma=1.666666666666667 \
          parthenon/output0/file_type=hdf5 \
          parthenon/output0/dt=-0.01 \
          parthenon/output0/variables={CPAW_OUTPUT_VARIABLES} \
          > {log.out} 2> {log.err}
        """

rule combine_cpaw_errors:
    input:
        lambda wildcards: expand(
            f"{cpaw_out(wildcards.fluid)}/runs/N{{N}}/cpaw-errors.dat",
            N=CPAW_RESOLUTIONS,
        )
    output:
        dat=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CPAW['dirname']}/cpaw-errors.dat"
    resources:
        runtime=10
    shell:
        """
        cat {input} > {output.dat}
        """

rule compute_cpaw_orders:
    input:
        dat=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CPAW['dirname']}/cpaw-errors.dat"
    output:
        txt=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CPAW['dirname']}/cpaw_orders.txt",
            caption="../report/cpaw_orders.rst",
            category=f"{config['dimension']} Tests",
            subcategory="{fluid} / Circularly Polarized Traveling Alfven Wave",
            labels={"fluid": "{fluid}", "mesh": "uniform", "diagnostic": "convergence orders"}
        )
    params:
        outdir=lambda wildcards: cpaw_out(wildcards.fluid),
    resources:
        runtime=10
    shell:
        """
        cd {params.outdir}
        python {CPAW[order_script]} {input.dat} > {output.txt}
        """
