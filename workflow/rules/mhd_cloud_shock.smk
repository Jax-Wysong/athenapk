MHD_CLOUD_SHOCK_DEFAULTS = {
    "enabled": False,
    "input": (
        "/mnt/home/wysongj2/athenapk-fourth-Berta24/inputs/"
        "ct_glm_compatible/mhd_cloud_shock.in"
    ),
    "dirname": "mhd_cloud_shock",
    "plotting_script": (
        "/mnt/home/wysongj2/athenapk-fourth-Berta24/workflow/"
        "diagnostics_scripts/mhd_cloud_shock_density.py"
    ),
    "resolution": [128, 128, 128],
    "meshblock": [32, 32, 32],
    "output_dt": 0.002,
    "fps": 8,
    "detector_epsilon": 1.0e-12,
    "mpi_tasks": 8,
    "mem_mb_per_cpu": 8000,
}
MHD_CLOUD_SHOCK = {
    **MHD_CLOUD_SHOCK_DEFAULTS,
    **config["tests"].get("mhd_cloud_shock", {}),
}
MHD_CLOUD_SHOCK_RESOLUTION = MHD_CLOUD_SHOCK["resolution"]
MHD_CLOUD_SHOCK_MB = MHD_CLOUD_SHOCK["meshblock"]
MHD_CLOUD_SHOCK_FLUIDS = MHD_CLOUD_SHOCK.get("fluids", config["fluids"])
MHD_CLOUD_SHOCK_NUM_BLOCKS = (
    ((MHD_CLOUD_SHOCK_RESOLUTION[0] + MHD_CLOUD_SHOCK_MB[0] - 1) // MHD_CLOUD_SHOCK_MB[0])
    * ((MHD_CLOUD_SHOCK_RESOLUTION[1] + MHD_CLOUD_SHOCK_MB[1] - 1) // MHD_CLOUD_SHOCK_MB[1])
    * ((MHD_CLOUD_SHOCK_RESOLUTION[2] + MHD_CLOUD_SHOCK_MB[2] - 1) // MHD_CLOUD_SHOCK_MB[2])
)
MHD_CLOUD_SHOCK_MPI_TASKS = min(
    MHD_CLOUD_SHOCK["mpi_tasks"], MHD_CLOUD_SHOCK_NUM_BLOCKS
)

if MHD_CLOUD_SHOCK["enabled"] and config["dimension"] != "3D":
    raise ValueError("mhd_cloud_shock is a three-dimensional test")


def mhd_cloud_shock_out(fluid):
    return (
        f"{config['results_root']}/{config['dimension']}/{fluid}/"
        f"{MHD_CLOUD_SHOCK['dirname']}"
    )


mhd_cloud_shock_targets = []
if MHD_CLOUD_SHOCK["enabled"]:
    mhd_cloud_shock_targets = expand(
        "{outdir}/{name}",
        outdir=[mhd_cloud_shock_out(fluid) for fluid in MHD_CLOUD_SHOCK_FLUIDS],
        name=[
            "mhd_cloud_shock_density_xy.mp4",
            "mhd_cloud_shock_density_xy.gif",
            "mhd_cloud_shock_density_xy_final.png",
        ],
    )


rule run_mhd_cloud_shock:
    input:
        exe=config.get(
            "athenapk_mpi_hdf5",
            "/mnt/home/wysongj2/athenapk-fourth-Berta24/"
            "build-mpi-hdf5/bin/athenaPK",
        ),
        deck=MHD_CLOUD_SHOCK["input"]
    output:
        done=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{MHD_CLOUD_SHOCK['dirname']}/phdf-files/run.done"
        )
    log:
        out=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{MHD_CLOUD_SHOCK['dirname']}/run.out"
        ),
        err=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{MHD_CLOUD_SHOCK['dirname']}/run.err"
        )
    envmodules:
        "foss/2023a",
        "HDF5/1.14.0-gompi-2023a"
    params:
        rundir=lambda wc: f"{mhd_cloud_shock_out(wc.fluid)}/phdf-files",
        problem_id=lambda wc: (
            f"mhd_cloud_shock_{wc.fluid}_"
            f"Nx{MHD_CLOUD_SHOCK_RESOLUTION[0]}x"
            f"{MHD_CLOUD_SHOCK_RESOLUTION[1]}x"
            f"{MHD_CLOUD_SHOCK_RESOLUTION[2]}"
        ),
        nx1=MHD_CLOUD_SHOCK_RESOLUTION[0],
        nx2=MHD_CLOUD_SHOCK_RESOLUTION[1],
        nx3=MHD_CLOUD_SHOCK_RESOLUTION[2],
        mb_nx1=MHD_CLOUD_SHOCK_MB[0],
        mb_nx2=MHD_CLOUD_SHOCK_MB[1],
        mb_nx3=MHD_CLOUD_SHOCK_MB[2],
        output_dt=MHD_CLOUD_SHOCK["output_dt"],
        detector_epsilon=MHD_CLOUD_SHOCK.get("detector_epsilon", 1.0e-12)
    resources:
        runtime=720,
        nodes=1,
        tasks=MHD_CLOUD_SHOCK_MPI_TASKS,
        mpi="srun",
        mem_mb_per_cpu=MHD_CLOUD_SHOCK["mem_mb_per_cpu"]
    shell:
        """
        mkdir -p {params.rundir}
        rm -f {params.rundir}/*.phdf
        rm -f {params.rundir}/*.phdf.xdmf
        rm -f {params.rundir}/*.hst
        rm -f {output.done}
        cd {params.rundir}

        {resources.mpi} -n {resources.tasks} {input.exe} -i {input.deck} \
          parthenon/job/problem_id={params.problem_id} \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3={params.nx3} \
          parthenon/meshblock/nx1={params.mb_nx1} \
          parthenon/meshblock/nx2={params.mb_nx2} \
          parthenon/meshblock/nx3={params.mb_nx3} \
          parthenon/mesh/nghost=3 \
          parthenon/time/tlim=0.06 \
          parthenon/time/cfl=0.3 \
          parthenon/time/integrator={config[integrator]} \
          hydro/fluid={wildcards.fluid} \
          hydro/riemann={config[riemann]} \
          hydro/reconstruction={config[reconstruction]} \
          hydro/convergence_order={config[convergence_order]} \
          hydro/discontinuity_detector={config[discontinuity_detector]} \
          hydro/discontinuity_detector_threshold={config[discontinuity_detector_threshold]} \
          hydro/discontinuity_detector_epsilon={params.detector_epsilon} \
          hydro/gamma=1.666666666666667 \
          hydro/scratch_level=1 \
          hydro/pfloor=1e-15 \
          parthenon/output0/file_type=hdf5 \
          parthenon/output0/dt={params.output_dt} \
          parthenon/output0/variables=prim,berta24_troubled \
          parthenon/output0/include_in_final=true \
          parthenon/output0/use_final_label=true \
          > {log.out} 2> {log.err}

        touch {output.done}
        """


rule make_mhd_cloud_shock_density:
    input:
        done=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{MHD_CLOUD_SHOCK['dirname']}/phdf-files/run.done"
        )
    output:
        movie=report(
            (
                f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
                f"{MHD_CLOUD_SHOCK['dirname']}/mhd_cloud_shock_density_xy.mp4"
            ),
            caption="../report/mhd_cloud_shock_density.rst",
            category="3D Tests",
            subcategory="{fluid} / Cloud-Shock Interaction",
            labels={
                "fluid": "{fluid}",
                "quantity": "density",
                "slice": "z = 0.5",
                "scale": "logarithmic",
                "snapshot": "evolution",
            },
        ),
        gif=report(
            (
                f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
                f"{MHD_CLOUD_SHOCK['dirname']}/mhd_cloud_shock_density_xy.gif"
            ),
            caption="../report/mhd_cloud_shock_density.rst",
            category="3D Tests",
            subcategory="{fluid} / Cloud-Shock Interaction",
            labels={
                "fluid": "{fluid}",
                "quantity": "density",
                "slice": "z = 0.5",
                "scale": "logarithmic",
                "snapshot": "evolution (GIF)",
            },
        ),
        final=report(
            (
                f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
                f"{MHD_CLOUD_SHOCK['dirname']}/"
                "mhd_cloud_shock_density_xy_final.png"
            ),
            caption="../report/mhd_cloud_shock_density.rst",
            category="3D Tests",
            subcategory="{fluid} / Cloud-Shock Interaction",
            labels={
                "fluid": "{fluid}",
                "quantity": "density",
                "slice": "z = 0.5",
                "scale": "logarithmic",
                "snapshot": "t = 0.06",
            },
        )
    params:
        phdf=lambda wc: f"{mhd_cloud_shock_out(wc.fluid)}/phdf-files",
        fps=MHD_CLOUD_SHOCK.get("fps", 8)
    resources:
        runtime=120,
        mem_mb=16000
    shell:
        """
        {config[plotting_python]} {MHD_CLOUD_SHOCK[plotting_script]} \
          {params.phdf} \
          --movie {output.movie} \
          --gif {output.gif} \
          --final {output.final} \
          --fps {params.fps} \
          --expected-final-time 0.06
        """
