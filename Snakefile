# source activate <conda environment where snakemake has been installed>
# snakemake <name of the file you want to create>
all_output_cols = ['X3CLGANDWORK', 'S3CLASSES', 'S3WORK', 'S3APPRENTICE', 'X4EVRAPPCLG', 'S3CLGAPPNUM']

# extract output columns
# assign missing values
rule output_extract:
    input: "data/HSLS_2016_v1_0_CSV_Datasets/hsls_16_student_v1_0.csv"
    output:
        "data/processed/output_columns.csv",
        "figures/output_extract_missingness_per_output.png"
    shell: "python ./output_extract.py"

# identify missingness and imputation for baseline variables
rule missingness_imputation:
    input: "data/HSLS_2016_v1_0_CSV_Datasets/hsls_16_student_v1_0.csv"
    output:
        "data/processed/baseline_features.csv",
        "figures/missingness_imputation_missingness_per_predictor.csv"
    shell: "python ./missingness_imputation.py"

# train neural network for each of the output columns
rule nnet_train:
    input:
        "data/processed/baseline_features.csv",
        "data/processed/output_columns.csv"
    output: "data/trained_models/nnet_{output_col}.pickle"
    shell: "python ./nnet_train.py --output_col {wildcards.output_col}"

rule nnet_train_all:
    input: expand("data/trained_models/nnet_{output_col}.pickle", output_col=all_output_cols)
    output: "nnet_train_all_done.txt"
    shell: "echo done > nnet_train_all_done.txt"

