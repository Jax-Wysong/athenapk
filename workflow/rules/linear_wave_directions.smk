LINEAR_WAVE_DIRECTIONS = config["tests"]["linear_wave_directions"]
LINEAR_WAVE_DIRECTION_VECTORS = LINEAR_WAVE_DIRECTIONS["directions"]
LINEAR_WAVE_DIRECTION_NAMES = list(LINEAR_WAVE_DIRECTION_VECTORS)
LINEAR_WAVE_DIRECTION_RESOLUTIONS = LINEAR_WAVE_DIRECTIONS["resolutions"]
LINEAR_WAVE_DIRECTION_WAVE = LINEAR_WAVE_DIRECTIONS["wave"]
LINEAR_WAVE_DIRECTION_VIZ_N = LINEAR_WAVE_DIRECTIONS["visualization_resolution"]


if LINEAR_WAVE_DIRECTIONS["enabled"] and config["dimension"] != "2D":
    raise ValueError("linear_wave_directions is currently a 2D-only test")


def linear_wave_direction_out(fluid, direction):
    return (
        f"{config['results_root']}/2D/{fluid}/"
        f"{LINEAR_WAVE_DIRECTIONS['dirname']}/{direction}/"
        f"wave_{LINEAR_WAVE_DIRECTION_WAVE}"
    )


linear_wave_direction_targets = []
if LINEAR_WAVE_DIRECTIONS["enabled"]:
    linear_wave_direction_order_targets = expand(
        "{outdir}/linear_wave_orders-{wave}.txt",
        outdir=[
            linear_wave_direction_out(fluid, direction)
            for fluid in config["fluids"]
            for direction in LINEAR_WAVE_DIRECTION_NAMES
        ],
        wave=[LINEAR_WAVE_DIRECTION_WAVE],
    )
    linear_wave_direction_movie_targets = expand(
        "{outdir}/linear_wave_density.gif",
        outdir=[
            linear_wave_direction_out(fluid, direction)
            for fluid in config["fluids"]
            for direction in LINEAR_WAVE_DIRECTION_NAMES
        ],
    )
    linear_wave_direction_targets = (
        linear_wave_direction_order_targets + linear_wave_direction_movie_targets
    )


rule run_linear_wave_direction:
    input:
        executable=config["athenapk"],
        deck=LINEAR_WAVE_DIRECTIONS["input"]
    output:
        dat=(
            f"{config['results_root']}/2D/{{fluid}}/"
            f"{LINEAR_WAVE_DIRECTIONS['dirname']}/{{direction}}/"
            f"wave_{LINEAR_WAVE_DIRECTION_WAVE}/runs/N{{N}}/"
            f"linearwave-errors-{LINEAR_WAVE_DIRECTION_WAVE}.dat"
        )
    log:
        out=(
            f"{config['results_root']}/2D/{{fluid}}/"
            f"{LINEAR_WAVE_DIRECTIONS['dirname']}/{{direction}}/"
            f"wave_{LINEAR_WAVE_DIRECTION_WAVE}/runs/N{{N}}/run.out"
        ),
        err=(
            f"{config['results_root']}/2D/{{fluid}}/"
            f"{LINEAR_WAVE_DIRECTIONS['dirname']}/{{direction}}/"
            f"wave_{LINEAR_WAVE_DIRECTION_WAVE}/runs/N{{N}}/run.err"
        )
    params:
        rundir=lambda wc: (
            f"{linear_wave_direction_out(wc.fluid, wc.direction)}/runs/N{wc.N}"
        ),
        problem_id=lambda wc: (
            f"linear_wave_direction_{wc.direction.replace('-', '_')}_"
            f"{wc.fluid}_w{LINEAR_WAVE_DIRECTION_WAVE}_N{wc.N}"
        ),
        wave_n1=lambda wc: LINEAR_WAVE_DIRECTION_VECTORS[wc.direction][0],
        wave_n2=lambda wc: LINEAR_WAVE_DIRECTION_VECTORS[wc.direction][1],
        wave_n3=lambda wc: LINEAR_WAVE_DIRECTION_VECTORS[wc.direction][2],
        nx=lambda wc: int(wc.N)
    resources:
        runtime=120,
        mem_mb=4000
    wildcard_constraints:
        direction="|".join(LINEAR_WAVE_DIRECTION_NAMES)
    shell:
        """
        mkdir -p {params.rundir}
        rm -f {output.dat}
        cd {params.rundir}

        {input.executable} -i {input.deck} \
          parthenon/job/problem_id={params.problem_id} \
          problem/linear_wave_mhd/wave_flag={LINEAR_WAVE_DIRECTION_WAVE} \
          problem/linear_wave_mhd/wave_n1={params.wave_n1} \
          problem/linear_wave_mhd/wave_n2={params.wave_n2} \
          problem/linear_wave_mhd/wave_n3={params.wave_n3} \
          problem/linear_wave_mhd/compute_error=true \
          problem/linear_wave_mhd/test=true \
          parthenon/mesh/x1min=0.0 \
          parthenon/mesh/x1max=1.0 \
          parthenon/mesh/x2min=0.0 \
          parthenon/mesh/x2max=1.0 \
          parthenon/mesh/nx1={params.nx} \
          parthenon/mesh/nx2={params.nx} \
          parthenon/mesh/nx3=1 \
          parthenon/mesh/nghost=3 \
          parthenon/meshblock/nx1={params.nx} \
          parthenon/meshblock/nx2={params.nx} \
          parthenon/meshblock/nx3=1 \
          parthenon/time/tlim=1.0 \
          parthenon/time/cfl=0.3 \
          parthenon/time/integrator={config[integrator]} \
          hydro/fluid={wildcards.fluid} \
          hydro/riemann={config[riemann]} \
          hydro/reconstruction={config[reconstruction]} \
          hydro/convergence_order={config[convergence_order]} \
          hydro/discontinuity_detector={config[discontinuity_detector]} \
          hydro/discontinuity_detector_threshold={config[discontinuity_detector_threshold]} \
          hydro/gamma=1.666666666666667 \
          parthenon/output0/file_type=hdf5 \
          parthenon/output0/dt=-0.01 \
          parthenon/output0/variables=prim \
          > {log.out} 2> {log.err}
        """


rule combine_linear_wave_direction_errors:
    input:
        lambda wc: expand(
            (
                f"{linear_wave_direction_out(wc.fluid, wc.direction)}/runs/"
                f"N{{N}}/linearwave-errors-{LINEAR_WAVE_DIRECTION_WAVE}.dat"
            ),
            N=LINEAR_WAVE_DIRECTION_RESOLUTIONS,
        )
    output:
        dat=(
            f"{config['results_root']}/2D/{{fluid}}/"
            f"{LINEAR_WAVE_DIRECTIONS['dirname']}/{{direction}}/"
            f"wave_{LINEAR_WAVE_DIRECTION_WAVE}/"
            f"linearwave-errors-{LINEAR_WAVE_DIRECTION_WAVE}.dat"
        )
    wildcard_constraints:
        direction="|".join(LINEAR_WAVE_DIRECTION_NAMES)
    shell:
        """
        cat {input} > {output.dat}
        """


rule compute_linear_wave_direction_orders:
    input:
        dat=(
            f"{config['results_root']}/2D/{{fluid}}/"
            f"{LINEAR_WAVE_DIRECTIONS['dirname']}/{{direction}}/"
            f"wave_{LINEAR_WAVE_DIRECTION_WAVE}/"
            f"linearwave-errors-{LINEAR_WAVE_DIRECTION_WAVE}.dat"
        )
    output:
        txt=report(
            (
                f"{config['results_root']}/2D/{{fluid}}/"
                f"{LINEAR_WAVE_DIRECTIONS['dirname']}/{{direction}}/"
                f"wave_{LINEAR_WAVE_DIRECTION_WAVE}/"
                f"linear_wave_orders-{LINEAR_WAVE_DIRECTION_WAVE}.txt"
            ),
            caption="../report/linear_wave_direction_orders.rst",
            category="2D Tests",
            subcategory="{fluid} / Linear MHD Wave / Direction Sweep",
            labels={
                "fluid": "{fluid}",
                "mesh": "uniform",
                "direction": "{direction}",
                "wave": str(LINEAR_WAVE_DIRECTION_WAVE),
                "diagnostic": "convergence orders",
            },
        )
    wildcard_constraints:
        direction="|".join(LINEAR_WAVE_DIRECTION_NAMES)
    shell:
        """
        {config[plotting_python]} {LINEAR_WAVE_DIRECTIONS[order_script]} \
          {input.dat} > {output.txt}
        """


rule run_linear_wave_direction_viz:
    input:
        executable=config["athenapk"],
        deck=LINEAR_WAVE_DIRECTIONS["input"]
    output:
        done=(
            f"{config['results_root']}/2D/{{fluid}}/"
            f"{LINEAR_WAVE_DIRECTIONS['dirname']}/{{direction}}/"
            f"wave_{LINEAR_WAVE_DIRECTION_WAVE}/movie/phdf-files/run.done"
        )
    log:
        out=(
            f"{config['results_root']}/2D/{{fluid}}/"
            f"{LINEAR_WAVE_DIRECTIONS['dirname']}/{{direction}}/"
            f"wave_{LINEAR_WAVE_DIRECTION_WAVE}/movie/run.out"
        ),
        err=(
            f"{config['results_root']}/2D/{{fluid}}/"
            f"{LINEAR_WAVE_DIRECTIONS['dirname']}/{{direction}}/"
            f"wave_{LINEAR_WAVE_DIRECTION_WAVE}/movie/run.err"
        )
    params:
        rundir=lambda wc: (
            f"{linear_wave_direction_out(wc.fluid, wc.direction)}/movie/phdf-files"
        ),
        problem_id=lambda wc: (
            f"linear_wave_direction_movie_{wc.direction.replace('-', '_')}_"
            f"{wc.fluid}_w{LINEAR_WAVE_DIRECTION_WAVE}_N{LINEAR_WAVE_DIRECTION_VIZ_N}"
        ),
        wave_n1=lambda wc: LINEAR_WAVE_DIRECTION_VECTORS[wc.direction][0],
        wave_n2=lambda wc: LINEAR_WAVE_DIRECTION_VECTORS[wc.direction][1],
        wave_n3=lambda wc: LINEAR_WAVE_DIRECTION_VECTORS[wc.direction][2]
    resources:
        runtime=120,
        mem_mb=4000
    wildcard_constraints:
        direction="|".join(LINEAR_WAVE_DIRECTION_NAMES)
    shell:
        """
        mkdir -p {params.rundir}
        rm -f {params.rundir}/*.phdf
        rm -f {params.rundir}/*.phdf.xdmf
        rm -f {params.rundir}/*.hst
        rm -f {output.done}
        cd {params.rundir}

        {input.executable} -i {input.deck} \
          parthenon/job/problem_id={params.problem_id} \
          problem/linear_wave_mhd/wave_flag={LINEAR_WAVE_DIRECTION_WAVE} \
          problem/linear_wave_mhd/wave_n1={params.wave_n1} \
          problem/linear_wave_mhd/wave_n2={params.wave_n2} \
          problem/linear_wave_mhd/wave_n3={params.wave_n3} \
          problem/linear_wave_mhd/compute_error=false \
          problem/linear_wave_mhd/test=true \
          parthenon/mesh/x1min=0.0 \
          parthenon/mesh/x1max=1.0 \
          parthenon/mesh/x2min=0.0 \
          parthenon/mesh/x2max=1.0 \
          parthenon/mesh/nx1={LINEAR_WAVE_DIRECTION_VIZ_N} \
          parthenon/mesh/nx2={LINEAR_WAVE_DIRECTION_VIZ_N} \
          parthenon/mesh/nx3=1 \
          parthenon/mesh/nghost=3 \
          parthenon/meshblock/nx1={LINEAR_WAVE_DIRECTION_VIZ_N} \
          parthenon/meshblock/nx2={LINEAR_WAVE_DIRECTION_VIZ_N} \
          parthenon/meshblock/nx3=1 \
          parthenon/time/tlim=1.0 \
          parthenon/time/cfl=0.3 \
          parthenon/time/integrator={config[integrator]} \
          hydro/fluid={wildcards.fluid} \
          hydro/riemann={config[riemann]} \
          hydro/reconstruction={config[reconstruction]} \
          hydro/convergence_order={config[convergence_order]} \
          hydro/discontinuity_detector={config[discontinuity_detector]} \
          hydro/discontinuity_detector_threshold={config[discontinuity_detector_threshold]} \
          hydro/gamma=1.666666666666667 \
          parthenon/output0/file_type=hdf5 \
          parthenon/output0/dt={LINEAR_WAVE_DIRECTIONS[output_dt]} \
          parthenon/output0/variables=prim \
          > {log.out} 2> {log.err}

        touch {output.done}
        """


rule make_linear_wave_direction_movie:
    input:
        done=(
            f"{config['results_root']}/2D/{{fluid}}/"
            f"{LINEAR_WAVE_DIRECTIONS['dirname']}/{{direction}}/"
            f"wave_{LINEAR_WAVE_DIRECTION_WAVE}/movie/phdf-files/run.done"
        )
    output:
        gif=report(
            (
                f"{config['results_root']}/2D/{{fluid}}/"
                f"{LINEAR_WAVE_DIRECTIONS['dirname']}/{{direction}}/"
                f"wave_{LINEAR_WAVE_DIRECTION_WAVE}/linear_wave_density.gif"
            ),
            caption="../report/linear_wave_direction_movie.rst",
            category="2D Tests",
            subcategory="{fluid} / Linear MHD Wave / Direction Sweep",
            labels={
                "fluid": "{fluid}",
                "mesh": "uniform",
                "direction": "{direction}",
                "wave": str(LINEAR_WAVE_DIRECTION_WAVE),
                "quantity": "density perturbation",
            },
        )
    params:
        outdir=lambda wc: linear_wave_direction_out(wc.fluid, wc.direction),
        phdf=lambda wc: (
            f"{linear_wave_direction_out(wc.fluid, wc.direction)}/movie/phdf-files"
        )
    resources:
        runtime=60,
        mem_mb=4000
    wildcard_constraints:
        direction="|".join(LINEAR_WAVE_DIRECTION_NAMES)
    shell:
        """
        cd {params.outdir}
        {config[plotting_python]} {LINEAR_WAVE_DIRECTIONS[plotting_script]} \
          {params.phdf} -o {output.gif} \
          --direction {wildcards.direction} \
          --wave {LINEAR_WAVE_DIRECTION_WAVE} \
          --fps {LINEAR_WAVE_DIRECTIONS[fps]}
        """
