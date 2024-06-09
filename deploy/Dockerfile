FROM julia:1.9.4-bookworm

RUN apt-get update && apt-get install g++ --yes

WORKDIR /app/NowcastingHuascaranInference

COPY ./NowcastingHuascaranInference/Project.toml Project.toml
COPY ./NowcastingHuascaranInference/Manifest.toml Manifest.toml

RUN julia --project=. -e 'import Pkg; Pkg.instantiate()'

COPY ./NowcastingHuascaranInference/src src
COPY ./NowcastingHuascaranInference/scripts scripts

RUN julia --project=. -e 'using PackageCompiler; PackageCompiler.create_sysimage(cpu_target="generic", sysimage_path="inference-sysimage.so", precompile_statements_file="scripts/inference.jl")'

ENTRYPOINT ["julia", "-Jinference-sysimage.so" , "--project=.", "scripts/inference.jl", "--model=/app/model.bson", "--input_dir=/app/input", "--output_dir=/app/output"]

