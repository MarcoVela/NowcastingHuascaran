using LiveServer
import Pkg
Pkg.activate(".")

servedocs(include_dirs=["./src/"])
