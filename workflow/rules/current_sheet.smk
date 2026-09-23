CURRENT_SHEET = config["tests"]["current_sheet"]
CURRENT_SHEET_RESOLUTION = CURRENT_SHEET["resolution"]
CURRENT_SHEET_MB = CURRENT_SHEET["meshblock"]
CURRENT_SHEET_HYDRO_OPTIONS = hydro_cli_options(CURRENT_SHEET)
CURRENT_SHEET_OUTPUT_VARIABLES = output_variable_list(CURRENT_SHEET)
CURRENT_SHEET_FLUIDS = (
    config["fluids"]
    if ENABLED_TESTS is not None
    else CURRENT_SHEET.get("fluids", config["fluids"])
)
CURRENT_SHEET_CFL = float(CURRENT_SHEET.get("cfl", 0.3))
CURRENT_SHEET_TLIM = float(CURRENT_SHEET.get("tlim", 10.0))
CURRENT_SHEET_OUTPUT_DT = float(CURRENT_SHEET.get("output_dt", 0.05))

if len(CURRENT_SHEET_RESOLUTION) != 2 or len(CURRENT_SHEET_MB) != 2:
    raise ValueError("current_sheet resolution and meshblock must have two entries")
if any(int(n) <= 0 for n in CURRENT_SHEET_RESOLUTION + CURRENT_SHEET_MB):
    raise ValueError("current_sheet resolution and meshblock entries must be positive")
if any(
    int(global_n) % int(block_n) != 0
    for global_n, block_n in zip(CURRENT_SHEET_RESOLUTION, CURRENT_SHEET_MB)
):
    raise ValueError(
        f"current_sheet grid {CURRENT_SHEET_RESOLUTION} is not divisible by "
        f"meshblock {CURRENT_SHEET_MB}"
    )
if CURRENT_SHEET_CFL <= 0.0:
    raise ValueError("current_sheet cfl must be positive")
if CURRENT_SHEET_TLIM <= 0.0:
    raise ValueError("current_sheet tlim must be positive")
if CURRENT_SHEET_OUTPUT_DT <= 0.0:
    raise ValueError("current_sheet output_dt must be positive")
if CURRENT_SHEET["enabled"] and config["dimension"] != "2D":
    raise ValueError("current_sheet is a two-dimensional test")

def current_sheet_out(fluid):
    return f"{config['results_root']}/{config['dimension']}/{fluid}/{CURRENT_SHEET['dirname']}"

current_sheet_targets = []
if CURRENT_SHEET["enabled"]:
    current_sheet_targets = expand(
        "{outdir}/{name}",
        outdir=[current_sheet_out(fluid) for fluid in CURRENT_SHEET_FLUIDS],
        name=[
            "current_sheet_By.gif",
            "current_sheet_field_lines_initial.png",
            "current_sheet_field_lines_middle.png",
            "current_sheet_field_lines_final.png",
        ],
    )

rule run_current_sheet:
    input:
        exe=config["athenapk"],
        deck=CURRENT_SHEET["input"]
    output:
        done=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CURRENT_SHEET['dirname']}/phdf-files/run.done"
    log:
        out=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CURRENT_SHEET['dirname']}/run.out",
        err=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CURRENT_SHEET['dirname']}/run.err"
    envmodules:
        *SIMULATION_ENVMODULES
    params:
        outdir=lambda wc: current_sheet_out(wc.fluid),
        rundir=lambda wc: f"{current_sheet_out(wc.fluid)}/phdf-files",
        problem_id=lambda wc: f"current_sheet_{wc.fluid}_Nx{CURRENT_SHEET_RESOLUTION[0]}x{CURRENT_SHEET_RESOLUTION[1]}",
        nx1=CURRENT_SHEET_RESOLUTION[0],
        nx2=CURRENT_SHEET_RESOLUTION[1],
        nx1_mb=CURRENT_SHEET_MB[0],
        nx2_mb=CURRENT_SHEET_MB[1],
        cfl=CURRENT_SHEET_CFL,
        tlim=CURRENT_SHEET_TLIM,
        output_dt=CURRENT_SHEET_OUTPUT_DT,
    resources:
        runtime=240,
        mem_mb=16000,
        gpu=SIMULATION_GPU_COUNT,
        tasks_per_gpu=SIMULATION_TASKS_PER_GPU,
        cpus_per_gpu=SIMULATION_CPUS_PER_GPU,
        constraint=SIMULATION_CONSTRAINT,
        slurm_extra=SIMULATION_SLURM_EXTRA
        #nodes=1,
        #tasks=16,
        #mpi="srun",
        #mem_mb_per_cpu=2000
        #mem_mb=(default for now) mb means megabyte
        #slurm_partition=(default for now)
        # see https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html
        # for more options (number of tasks per job, cpu per task, mem per cpu, etc)
        #hydro/pfloor=1e-15 \
        #hydro/dfloor=1e-15 \
        #hydro/ct_energy_correction=true \

    shell:
        """
        mkdir -p {params.rundir}
        rm -f {params.rundir}/*.phdf
        rm -f {params.rundir}/*.phdf.xdmf
        rm -f {params.rundir}/*.hst
        rm -f {output.done}
        cd {params.rundir}

        {input.exe} -i {input.deck} \
          parthenon/job/problem_id={params.problem_id} \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3=1 \
          parthenon/meshblock/nx1={params.nx1_mb} \
          parthenon/meshblock/nx2={params.nx2_mb} \
          parthenon/meshblock/nx3=1 \
          parthenon/mesh/nghost={NGHOST} \
          parthenon/time/tlim={params.tlim} \
          parthenon/time/cfl={params.cfl} \
          parthenon/time/integrator={config[integrator]} \
          hydro/fluid={wildcards.fluid} \
          hydro/riemann={config[riemann]} \
          hydro/reconstruction={config[reconstruction]} \
          hydro/convergence_order={config[convergence_order]} \
          {CURRENT_SHEET_HYDRO_OPTIONS} \
          hydro/gamma=1.666666666666667 \
          hydro/scratch_level=1 \
          hydro/pfloor=1e-15 \
          parthenon/output0/file_type=hdf5 \
          parthenon/output0/dt={params.output_dt} \
          parthenon/output0/variables={CURRENT_SHEET_OUTPUT_VARIABLES} \
          > {log.out} 2> {log.err}
    
        touch {output.done}
        """

rule make_current_sheet_videos:
    input:
        done=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CURRENT_SHEET['dirname']}/phdf-files/run.done"
    output:
        gif=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CURRENT_SHEET['dirname']}/current_sheet_By.gif",
            caption="../report/current_sheet_By.rst",
            category="2D Tests",
            subcategory="{fluid} / Current Sheet",
            labels={"fluid": "{fluid}", "quantity": "magnetic field By"}
        )
    params:
        outdir=lambda wc: current_sheet_out(wc.fluid),
        phdf=lambda wc: f"{current_sheet_out(wc.fluid)}/phdf-files",
    resources:
        runtime=30,
        mem_mb=10000
    shell:
        """
        cd {params.outdir}
        {config[plotting_python]} {CURRENT_SHEET[plotting_script]} {params.phdf} -o current_sheet
        """

rule make_current_sheet_field_lines:
    input:
        done=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CURRENT_SHEET['dirname']}/phdf-files/run.done"
    output:
        initial=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CURRENT_SHEET['dirname']}/current_sheet_field_lines_initial.png",
            caption="../report/current_sheet_field_lines.rst",
            category="2D Tests",
            subcategory="{fluid} / Current Sheet",
            labels={"fluid": "{fluid}", "snapshot": "initial", "diagnostic": "magnetic field lines"}
        ),
        middle=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CURRENT_SHEET['dirname']}/current_sheet_field_lines_middle.png",
            caption="../report/current_sheet_field_lines.rst",
            category="2D Tests",
            subcategory="{fluid} / Current Sheet",
            labels={"fluid": "{fluid}", "snapshot": "middle", "diagnostic": "magnetic field lines"}
        ),
        final=report(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/{CURRENT_SHEET['dirname']}/current_sheet_field_lines_final.png",
            caption="../report/current_sheet_field_lines.rst",
            category="2D Tests",
            subcategory="{fluid} / Current Sheet",
            labels={"fluid": "{fluid}", "snapshot": "final", "diagnostic": "magnetic field lines"}
        )
    params:
        outdir=lambda wc: current_sheet_out(wc.fluid),
        phdf=lambda wc: f"{current_sheet_out(wc.fluid)}/phdf-files",
    resources:
        runtime=30
    shell:
        """
        cd {params.outdir}
        {config[plotting_python]} {CURRENT_SHEET[field_line_script]} {params.phdf} -o current_sheet
        """
