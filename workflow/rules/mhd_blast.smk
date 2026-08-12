MHD_BLAST_DEFAULTS = {
    "enabled": False,
    "input": (
        "/mnt/home/wysongj2/athenapk-fourth-Berta24/inputs/"
        "ct_glm_compatible/mhd_blast.in"
    ),
    "dirname": "mhd_blast",
    "plotting_script": (
        "/mnt/home/wysongj2/athenapk-fourth-Berta24/workflow/"
        "diagnostics_scripts/mhd_blast_density.py"
    ),
    "resolution": [192, 192, 192],
    "meshblock": [48, 48, 48],
    "tlim": 0.2,
    "output_dt": 0.01,
    "fps": 8,
    "detector_epsilon": 1.0e-12,
    "mpi_tasks": 16,
    "mem_mb_per_cpu": 2000,
}
MHD_BLAST = {
    **MHD_BLAST_DEFAULTS,
    **config["tests"].get("mhd_blast", {}),
}
MHD_BLAST_RESOLUTION = MHD_BLAST["resolution"]
MHD_BLAST_MB = MHD_BLAST["meshblock"]
MHD_BLAST_FLUIDS = MHD_BLAST.get("fluids", config["fluids"])
if len(MHD_BLAST_RESOLUTION) != 3 or len(MHD_BLAST_MB) != 3:
    raise ValueError("mhd_blast resolution and meshblock must have three entries")
MHD_BLAST_NUM_BLOCKS = (
    ((MHD_BLAST_RESOLUTION[0] + MHD_BLAST_MB[0] - 1) // MHD_BLAST_MB[0])
    * ((MHD_BLAST_RESOLUTION[1] + MHD_BLAST_MB[1] - 1) // MHD_BLAST_MB[1])
    * ((MHD_BLAST_RESOLUTION[2] + MHD_BLAST_MB[2] - 1) // MHD_BLAST_MB[2])
)
MHD_BLAST_MPI_TASKS = min(MHD_BLAST["mpi_tasks"], MHD_BLAST_NUM_BLOCKS)

if MHD_BLAST["enabled"] and config["dimension"] != "3D":
    raise ValueError("mhd_blast is a three-dimensional test")


def mhd_blast_out(fluid):
    return (
        f"{config['results_root']}/{config['dimension']}/{fluid}/"
        f"{MHD_BLAST['dirname']}"
    )


mhd_blast_targets = []
if MHD_BLAST["enabled"]:
    mhd_blast_targets = expand(
        "{outdir}/{name}",
        outdir=[mhd_blast_out(fluid) for fluid in MHD_BLAST_FLUIDS],
        name=[
            "mhd_blast_density_xy.mp4",
            "mhd_blast_density_xy.gif",
            "mhd_blast_density_xy_final.png",
            "mhd_blast_pressure_xy_final.png",
            "mhd_blast_specific_kinetic_energy_xy_final.png",
            "mhd_blast_magnetic_energy_xy_final.png",
        ],
    )


rule run_mhd_blast:
    input:
        exe=config.get(
            "athenapk_mpi_hdf5",
            "/mnt/home/wysongj2/athenapk-fourth-Berta24/"
            "build-mpi-site-hdf5/bin/athenaPK",
        ),
        deck=MHD_BLAST["input"]
    output:
        done=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{MHD_BLAST['dirname']}/phdf-files/run.done"
        )
    log:
        out=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{MHD_BLAST['dirname']}/run.out"
        ),
        err=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{MHD_BLAST['dirname']}/run.err"
        )
    envmodules:
        "foss/2023a",
        "HDF5/1.14.0-gompi-2023a"
    params:
        rundir=lambda wc: f"{mhd_blast_out(wc.fluid)}/phdf-files",
        problem_id=lambda wc: (
            f"mhd_blast_{wc.fluid}_"
            f"Nx{MHD_BLAST_RESOLUTION[0]}x{MHD_BLAST_RESOLUTION[1]}"
            f"x{MHD_BLAST_RESOLUTION[2]}"
        ),
        nx1=MHD_BLAST_RESOLUTION[0],
        nx2=MHD_BLAST_RESOLUTION[1],
        nx3=MHD_BLAST_RESOLUTION[2],
        mb_nx1=MHD_BLAST_MB[0],
        mb_nx2=MHD_BLAST_MB[1],
        mb_nx3=MHD_BLAST_MB[2],
        tlim=MHD_BLAST["tlim"],
        output_dt=MHD_BLAST["output_dt"],
        detector_epsilon=MHD_BLAST.get("detector_epsilon", 1.0e-12)
    resources:
        runtime=720,
        nodes=1,
        tasks=MHD_BLAST_MPI_TASKS,
        mpi="srun",
        mem_mb_per_cpu=MHD_BLAST["mem_mb_per_cpu"]
    shell:
        """
        mkdir -p {params.rundir}
        rm -f {params.rundir}/*.phdf
        rm -f {params.rundir}/*.phdf.xdmf
        rm -f {params.rundir}/*.hst
        rm -f {output.done}
        cd {params.rundir}

        {resources.mpi} -n {resources.tasks} {input.exe} -i {input.deck} \
          job/problem_id=mhd_blast \
          parthenon/job/problem_id={params.problem_id} \
          parthenon/mesh/nx1={params.nx1} \
          parthenon/mesh/nx2={params.nx2} \
          parthenon/mesh/nx3={params.nx3} \
          parthenon/meshblock/nx1={params.mb_nx1} \
          parthenon/meshblock/nx2={params.mb_nx2} \
          parthenon/meshblock/nx3={params.mb_nx3} \
          parthenon/mesh/nghost=3 \
          parthenon/time/tlim={params.tlim} \
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
          parthenon/output0/file_type=hdf5 \
          parthenon/output0/dt={params.output_dt} \
          parthenon/output0/variables=prim,berta24_troubled \
          parthenon/output0/include_in_final=true \
          parthenon/output0/use_final_label=true \
          > {log.out} 2> {log.err}

        touch {output.done}
        """


rule make_mhd_blast_density:
    input:
        done=(
            f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
            f"{MHD_BLAST['dirname']}/phdf-files/run.done"
        )
    output:
        movie=report(
            (
                f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
                f"{MHD_BLAST['dirname']}/mhd_blast_density_xy.mp4"
            ),
            caption="../report/mhd_blast_density.rst",
            category="3D Tests",
            subcategory="{fluid} / Berta C1 MHD Blast Wave",
            labels={
                "fluid": "{fluid}",
                "quantity": "density",
                "scale": "linear",
                "snapshot": "evolution",
            },
        ),
        gif=report(
            (
                f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
                f"{MHD_BLAST['dirname']}/mhd_blast_density_xy.gif"
            ),
            caption="../report/mhd_blast_density.rst",
            category="3D Tests",
            subcategory="{fluid} / Berta C1 MHD Blast Wave",
            labels={
                "fluid": "{fluid}",
                "quantity": "density",
                "scale": "linear",
                "snapshot": "evolution (GIF)",
            },
        ),
        final=report(
            (
                f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
                f"{MHD_BLAST['dirname']}/mhd_blast_density_xy_final.png"
            ),
            caption="../report/mhd_blast_density.rst",
            category="3D Tests",
            subcategory="{fluid} / Berta C1 MHD Blast Wave",
            labels={
                "fluid": "{fluid}",
                "quantity": "density",
                "scale": "linear",
                "snapshot": "z = 0, t = 0.2",
            },
        ),
        pressure_final=report(
            (
                f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
                f"{MHD_BLAST['dirname']}/mhd_blast_pressure_xy_final.png"
            ),
            caption="../report/mhd_blast_density.rst",
            category="3D Tests",
            subcategory="{fluid} / Berta C1 MHD Blast Wave",
            labels={
                "fluid": "{fluid}",
                "quantity": "thermal pressure",
                "scale": "logarithmic",
                "snapshot": "z = 0, t = 0.2",
            },
        ),
        kinetic_final=report(
            (
                f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
                f"{MHD_BLAST['dirname']}/"
                "mhd_blast_specific_kinetic_energy_xy_final.png"
            ),
            caption="../report/mhd_blast_density.rst",
            category="3D Tests",
            subcategory="{fluid} / Berta C1 MHD Blast Wave",
            labels={
                "fluid": "{fluid}",
                "quantity": "specific kinetic energy",
                "scale": "linear",
                "snapshot": "z = 0, t = 0.2",
            },
        ),
        magnetic_final=report(
            (
                f"{config['results_root']}/{config['dimension']}/{{fluid}}/"
                f"{MHD_BLAST['dirname']}/mhd_blast_magnetic_energy_xy_final.png"
            ),
            caption="../report/mhd_blast_density.rst",
            category="3D Tests",
            subcategory="{fluid} / Berta C1 MHD Blast Wave",
            labels={
                "fluid": "{fluid}",
                "quantity": "magnetic energy density",
                "scale": "linear",
                "snapshot": "z = 0, t = 0.2",
            },
        )
    params:
        phdf=lambda wc: f"{mhd_blast_out(wc.fluid)}/phdf-files",
        fps=MHD_BLAST.get("fps", 8),
        expected_final_time=MHD_BLAST["tlim"]
    resources:
        runtime=120,
        mem_mb=16000
    shell:
        """
        {config[plotting_python]} {MHD_BLAST[plotting_script]} \
          {params.phdf} \
          --movie {output.movie} \
          --gif {output.gif} \
          --final {output.final} \
          --pressure-final {output.pressure_final} \
          --kinetic-final {output.kinetic_final} \
          --magnetic-final {output.magnetic_final} \
          --fps {params.fps} \
          --expected-final-time {params.expected_final_time}
        """
