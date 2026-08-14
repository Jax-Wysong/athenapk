FIELD_LOOP = config["tests"]["field_loop"]
FIELD_LOOP_MESH = FIELD_LOOP["mesh"][config["dimension"]]
FIELD_LOOP_HYDRO_OPTIONS = hydro_cli_options(FIELD_LOOP)
FIELD_LOOP_OUTPUT_VARIABLES = output_variable_list(FIELD_LOOP)
FIELD_LOOP_PLANES = ["xy"] if config["dimension"] == "2D" else ["xy", "xz", "yz"]

def field_loop_base_out(fluid):
    return f"{config['results_root']}/{config['dimension']}/{fluid}/{FIELD_LOOP['dirname']}"

field_loop_targets = []
if FIELD_LOOP["enabled"]:
    field_loop_targets = expand(
        "{outdir}/field_loop_Bmag_{plane}.gif",
        outdir=[field_loop_base_out(fluid) for fluid in config["fluids"]],
        plane=FIELD_LOOP_PLANES,
    )

rule run_field_loop:
    input:
        exe=config["athenapk"],
        deck=FIELD_LOOP["input"]
    output:
        done=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{FIELD_LOOP['dirname']}/phdf-files/run.done"
    log:
        out=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{FIELD_LOOP['dirname']}/run.out",
        err=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{FIELD_LOOP['dirname']}/run.err"
    params:
        outdir=lambda wc: field_loop_base_out(wc.fluid),
        rundir=lambda wc: f"{field_loop_base_out(wc.fluid)}/phdf-files",
        problem_id=lambda wc: f"field_loop_{wc.fluid}_Nx{FIELD_LOOP_MESH['nx1']}x{FIELD_LOOP_MESH['nx2']}x{FIELD_LOOP_MESH['nx3']}",
        nx1=FIELD_LOOP_MESH["nx1"],
        nx2=FIELD_LOOP_MESH["nx2"],
        nx3=FIELD_LOOP_MESH["nx3"],
        mb_nx1=FIELD_LOOP_MESH["mb_nx1"],
        mb_nx2=FIELD_LOOP_MESH["mb_nx2"],
        mb_nx3=FIELD_LOOP_MESH["mb_nx3"],
        iprob=FIELD_LOOP_MESH["iprob"]
    resources:
        runtime=360,
        mem_mb=10000
        #nodes=1,
        #tasks=8,
        #mpi="srun",
        #mem_mb_per_cpu=10000
        #mem_mb=(default for now) mb means megabyte
        #slurm_partition=(default for now)
        # see https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html
        # for more options (number of tasks per job, cpu per task, mem per cpu, etc)
        #hydro/pfloor=1e-15 \
    shell:
        """
        mkdir -p {params.rundir}
        rm -f {params.rundir}/*.phdf
        rm -f {params.rundir}/*.phdf.xdmf
        rm -f {params.rundir}/*.hst
        rm -f {output.done}
        cd {params.rundir}

        {input.exe} -i {input.deck} \
          problem/field_loop/iprob={params.iprob} \
          parthenon/job/problem_id={params.problem_id} \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3={params.nx3} \
          parthenon/mesh/nghost=3 \
          parthenon/meshblock/nx1={params.mb_nx1} \
          parthenon/meshblock/nx2={params.mb_nx2} \
          parthenon/meshblock/nx3={params.mb_nx3} \
          parthenon/time/tlim=2.0 \
          parthenon/time/cfl=0.3 \
          parthenon/time/integrator={config[integrator]} \
          hydro/fluid={wildcards.fluid} \
          hydro/riemann={config[riemann]} \
          hydro/reconstruction={config[reconstruction]} \
          hydro/convergence_order={config[convergence_order]} \
          {FIELD_LOOP_HYDRO_OPTIONS} \
          hydro/gamma=1.666666666666667 \
          parthenon/output0/file_type=hdf5 \
          parthenon/output0/dt=0.02 \
          parthenon/output0/variables={FIELD_LOOP_OUTPUT_VARIABLES} \
          > {log.out} 2> {log.err}
    
        touch {output.done}
        """

if config["dimension"] == "2D":
    rule make_field_loop_videos:
        input:
            done=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{FIELD_LOOP['dirname']}/phdf-files/run.done"
        output:
            xy=report(
                f"{config['results_root']}/{config['dimension']}/{{fluid}}/{FIELD_LOOP['dirname']}/field_loop_Bmag_xy.gif",
                caption="../report/field_loop_Bmag.rst",
                category=f"{config['dimension']} Tests",
                subcategory="{fluid} / Field Loop",
                labels={"fluid": "{fluid}", "mesh": "uniform", "quantity": "magnetic field magnitude", "slice": "xy"}
            )
        params:
            outdir=lambda wc: field_loop_base_out(wc.fluid),
            phdf=lambda wc: f"{field_loop_base_out(wc.fluid)}/phdf-files",
        resources:
            runtime=60
        shell:
            """
            cd {params.outdir}
            {config[plotting_python]} {FIELD_LOOP[plotting_script]} {params.phdf} -o field_loop
            """
else:
    rule make_field_loop_videos:
        input:
            done=f"{config['results_root']}/{config['dimension']}/{{fluid}}/{FIELD_LOOP['dirname']}/phdf-files/run.done"
        output:
            xy=report(
                f"{config['results_root']}/{config['dimension']}/{{fluid}}/{FIELD_LOOP['dirname']}/field_loop_Bmag_xy.gif",
                caption="../report/field_loop_Bmag.rst",
                category=f"{config['dimension']} Tests",
                subcategory="{fluid} / Field Loop",
                labels={"fluid": "{fluid}", "mesh": "uniform", "quantity": "magnetic field magnitude", "slice": "xy"}
            ),
            xz=report(
                f"{config['results_root']}/{config['dimension']}/{{fluid}}/{FIELD_LOOP['dirname']}/field_loop_Bmag_xz.gif",
                caption="../report/field_loop_Bmag.rst",
                category=f"{config['dimension']} Tests",
                subcategory="{fluid} / Field Loop",
                labels={"fluid": "{fluid}", "mesh": "uniform", "quantity": "magnetic field magnitude", "slice": "xz"}
            ),
            yz=report(
                f"{config['results_root']}/{config['dimension']}/{{fluid}}/{FIELD_LOOP['dirname']}/field_loop_Bmag_yz.gif",
                caption="../report/field_loop_Bmag.rst",
                category=f"{config['dimension']} Tests",
                subcategory="{fluid} / Field Loop",
                labels={"fluid": "{fluid}", "mesh": "uniform", "quantity": "magnetic field magnitude", "slice": "yz"}
            )
        params:
            outdir=lambda wc: field_loop_base_out(wc.fluid),
            phdf=lambda wc: f"{field_loop_base_out(wc.fluid)}/phdf-files",
        resources:
            runtime=60
        shell:
            """
            cd {params.outdir}
            {config[plotting_python]} {FIELD_LOOP[plotting_script]} {params.phdf} -o field_loop
            """
