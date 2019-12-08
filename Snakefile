localrules: meanimputation
rule meanimputation:
    input: "data/"
    output: "data/"
    shell: "python ./meanimputation.py"
