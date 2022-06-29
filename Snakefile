rule sgrA_star:
    input:
        "src/scripts/Fit-SgrAstar-PM.py"
    output:
        "src/data/Reid2020_refit.ecsv",
        "src/data/Reid2020_jointNorth.netcdf",
        "src/data/Reid2020_jointEast.netcdf"
    conda:
        "environment.yml"
    shell:
        "python {input[0]}"

rule sgrA_star_combine:
    input:
        "src/scripts/Combine-Reid-GRAVITY-SgrA.py",
        "src/data/Reid2020_refit.ecsv"
    output:
        "src/data/sgrA_star.ecsv"
    conda:
        "environment.yml"
    shell:
        "python {input[0]}"

rule basis_funcs:
    input:
        "src/scripts/Generate-sech2-basis-functions.py"
    output:
        "src/data/basis-funcs.pkl"
    conda:
        "environment.yml"
    shell:
        "python {input[0]}"