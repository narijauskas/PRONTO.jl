# use this version for local preview builds
import Pkg
Pkg.activate("./docs") # activate the docs-specific Project.toml

using LiveServer
servedocs(; foldername = "./docs")
# press CTRL+C (CMD+C) to stop the preview
     