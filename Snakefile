# User config
configfile: "showyourwork.yml"


# Import the showyourwork module
module showyourwork:
    snakefile:
        "showyourwork/workflow/Snakefile"
    config:
        config

# Use all default rules
use rule * from showyourwork

rule sgrA_star:
    input:
        "src/data/Fit-SgrAstar-PM.py"
    output:
        "src/data/Reid2020_refit.ecsv",
        "src/data/Reid2020_jointNorth.netcdf",
        "src/data/Reid2020_jointEast.netcdf"
    shell:
        "python {input[0]}"
