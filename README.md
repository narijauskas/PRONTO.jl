# PRONTO.jl

A julia implementation of PRONTO.

## Before Installation
Install [julia](https://julialang.org/), [git](https://git-scm.com/), [VSCode](https://code.visualstudio.com/), and the terminal of your choice. These instructions assume you can correctly install and configure these tools.

Make a [GitHub](https://github.com/) account if you don't already have one. I strongly recommend you set up an SSH key on your device and connect it to GitHub as well.

## Installation
1. change into your julia dev directory (or wherever you keep git repositories)
```
cd ~/.julia/dev
```
2. clone this repository
```
git clone git@github.com:mantasnaris/PRONTO.jl
```
or, if you don't have ssh set up:
```
https://github.com/mantasnaris/PRONTO.jl
```
3. start julia
```
julia -t auto
```
4. activate the local PRONTO environment (in julia pkg mode)
```
] activate .
```
5. download dependencies (in julia pkg mode)
```
] instantiate
```


## Usage
This code is still very much not done, stay tuned for usage instructions.

## Organization
`src` - package source files
`test` - package test files
`scripts` - examples, notebooks, etc.
`dev` - experimental code
