SMOOTH = config["tests"]["smooth_vortex"]
SMOOTH_RESOLUTIONS = SMOOTH["resolutions"]
SMOOTH_HYDRO_OPTIONS = hydro_cli_options(SMOOTH)
SMOOTH_OUTPUT_VARIABLES = output_variable_list(SMOOTH)
SMOOTH_REQUESTED_MPI_TASKS = int(SMOOTH.get("mpi_tasks", 4))
SMOOTH_MEM_MB_PER_CPU = int(SMOOTH.get("mem_mb_per_cpu", 4000))
SMOOTH_LAUNCHER_MODE = SMOOTH.get("launcher", "srun")
SMOOTH_CFL = float(SMOOTH.get("cfl", 0.3))
SMOOTH_TLIM = float(SMOOTH.get("tlim", 10.0))
SMOOTH_MPI_TASKS = min(SMOOTH_REQUESTED_MPI_TASKS, 4)
SMOOTH_MPI_RESOURCE = (
    "srun"
    if SMOOTH_LAUNCHER_MODE == "srun" and SMOOTH_MPI_TASKS > 1
    else False
)
SMOOTH_LAUNCHER = (
    "" if SMOOTH_LAUNCHER_MODE == "direct" else f"srun -n {SMOOTH_MPI_TASKS}"
)

if SMOOTH_REQUESTED_MPI_TASKS < 1:
    raise ValueError("smooth_vortex mpi_tasks must be positive")
if SMOOTH_MEM_MB_PER_CPU < 1:
    raise ValueError("smooth_vortex mem_mb_per_cpu must be positive")
if SMOOTH_CFL <= 0.0:
    raise ValueError("smooth_vortex cfl must be positive")
if SMOOTH_TLIM <= 0.0:
    raise ValueError("smooth_vortex tlim must be positive")
if SMOOTH_LAUNCHER_MODE not in ("direct", "srun"):
    raise ValueError("smooth_vortex launcher must be 'direct' or 'srun'")
if SMOOTH_LAUNCHER_MODE == "direct" and SMOOTH_REQUESTED_MPI_TASKS != 1:
    raise ValueError("smooth_vortex launcher=direct requires mpi_tasks=1")

def smooth_vortex_out(fluid):
    return f"{config['results_root']}/{config['dimension']}/{fluid}/{SMOOTH['dirname']}"

smooth_vortex_targets = []
if SMOOTH["enabled"]:
    smooth_vortex_targets = expand(
        "{outdir}/smooth_vortex_orders.txt",
        outdir=[smooth_vortex_out(fluid) for fluid in config["fluids"]],
    )

rule run_smooth_vortex:
    output:
        dat=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{SMOOTH['dirname']}/runs/N{{N}}/smoothVortexMHD-errors.dat"
    log:
        out=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{SMOOTH['dirname']}/runs/N{{N}}/run.out",
        err=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{SMOOTH['dirname']}/runs/N{{N}}/run.err"
    envmodules:
        *SIMULATION_ENVMODULES
    params:
        outdir=lambda wildcards: smooth_vortex_out(wildcards.fluid),
        rundir=lambda wildcards: f"{smooth_vortex_out(wildcards.fluid)}/runs/N{wildcards.N}",
        problem_id=lambda wildcards: f"smooth_mhd_vortex_{wildcards.fluid}_N{wildcards.N}",
        nx1_mb=lambda wildcards: (int(wildcards.N) // 2),
        nx2_mb=lambda wildcards: (int(wildcards.N) // 2),
        cfl=SMOOTH_CFL,
        tlim=SMOOTH_TLIM,
        launcher=SMOOTH_LAUNCHER
    resources:
        runtime=240,
        nodes=1,
        tasks=SMOOTH_MPI_TASKS,
        mpi=SMOOTH_MPI_RESOURCE,
        mem_mb_per_cpu=SMOOTH_MEM_MB_PER_CPU,
        gpu=SIMULATION_GPU_COUNT,
        tasks_per_gpu=SIMULATION_TASKS_PER_GPU,
        cpus_per_gpu=SIMULATION_CPUS_PER_GPU,
        constraint=SIMULATION_CONSTRAINT,
        slurm_extra=SIMULATION_SLURM_EXTRA
        #mem_mb=(default for now) mb means megabyte
        #slurm_partition=(default for now)
        # see https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html
        # for more options (number of tasks per job, cpu per task, mem per cpu, etc)
        #hydro/pfloor=1e-15 \
    shell:
        """
        mkdir -p {params.rundir}
        cd {params.rundir}
        rm -f {output.dat}

        {params.launcher} {config[athenapk_mpi]} -i {SMOOTH[input]} \
          parthenon/job/problem_id={params.problem_id} \
          parthenon/mesh/nx1={wildcards.N} \
          parthenon/mesh/nx2={wildcards.N} \
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
          {SMOOTH_HYDRO_OPTIONS} \
          hydro/gamma=1.666666666666667 \
          parthenon/output0/file_type=hdf5 \
          parthenon/output0/dt=-0.01 \
          parthenon/output0/variables={SMOOTH_OUTPUT_VARIABLES} \
          > {log.out} 2> {log.err}
        """

rule combine_smooth_vortex_errors:
    input:
        lambda wildcards: expand(
            f"{smooth_vortex_out(wildcards.fluid)}/runs/N{{N}}/smoothVortexMHD-errors.dat",
            N=SMOOTH_RESOLUTIONS,
        )
    output:
        f"{config['results_root']}/{config['dimension']}/{{fluid}}/{SMOOTH['dirname']}/smoothVortexMHD-errors.dat"
    resources:
        runtime=10
    shell:
        """
        cat {input} > {output}
        """

rule compute_smooth_vortex_orders:
    input:
        dat=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{SMOOTH['dirname']}/smoothVortexMHD-errors.dat"
    output:
        txt=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{SMOOTH['dirname']}/smooth_vortex_orders.txt",
            caption="../report/smooth_vortex_orders.rst",
            category="2D Tests",
            subcategory="{fluid} / Smooth MHD Vortex",
            labels={"fluid": "{fluid}", "diagnostic": "convergence orders"}
        )
    params:
        outdir=lambda wildcards: smooth_vortex_out(wildcards.fluid),
    resources:
        runtime=10
    shell:
        """
        cd {params.outdir}
        python {SMOOTH[order_script]} {input.dat} > {output.txt}
        """
