#!/bin/bash
#PBS -P np01
#PBS -q normal 
#PBS -l walltime=01:00:00
#PBS -l ncpus=4
#PBS -l mem=16gb
#PBS -N compile
#PBS -l wd
#PBS -o /scratch/np01/jm3247/GridapSolvers.jl/joss_paper/scalability/compile/compile.o
#PBS -e /scratch/np01/jm3247/GridapSolvers.jl/joss_paper/scalability/compile/compile.e 

source /scratch/np01/jm3247/GridapSolvers.jl/joss_paper/scalability/modules.sh

julia --project=/scratch/np01/jm3247/GridapSolvers.jl/joss_paper/scalability -O3 /scratch/np01/jm3247/GridapSolvers.jl/joss_paper/scalability/compile/compile.jl
