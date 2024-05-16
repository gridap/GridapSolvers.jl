
using Mustache
using DrWatson

function jobname(args...)
  s = savename(args...;connector="_")
  s = replace(s,"="=>"_")
  s = "stokes_$s"
  return s
end
driverdir(args...) = normpath(projectdir("../../",args...))

function clean_params(d)
  o = Dict()
  for k in keys(d)
    if isa(d[k],Real)
      o[k] = Int(d[k])
    elseif isa(d[k],Symbol) || isa(d[k],String) || isa(d[k],Number)
      o[k] = d[k]
    elseif isa(d[k],Tuple)
      o[k] = prod(d[k])
    end
  end
  return o
end

function jobdict(params)
  np = params[:np]
  nc = params[:nc]
  fparams = clean_params(params)
  Dict(
    "project" => haskey(ENV,"PROJECT") ? ENV["PROJECT"] : "",
    "q" => "normal",
    "o" => datadir(jobname(fparams,"o.txt")),
    "e" => datadir(jobname(fparams,"e.txt")),
    "walltime" => "01:00:00",
    "ncpus" => prod(np),
    "nnodes" => prod(np)÷48,
    "mem" => "$(prod(np)*4)gb",
    "name" => jobname(fparams),
    "np" => prod(np),
    "nc" => nc,
    "np_per_level" => Int[prod(np),prod(np)],
    "driver" => projectdir("driver.jl"),
    "projectdir" => driverdir(),
    "datadir"    => datadir(),
    "modules"    => projectdir("modules.sh"),
    "driverdir"  => projectdir(),
    "title"      => datadir(jobname(fparams)),
    "sysimage"   => projectdir("GridapSolvers.so")
  )
end

function generate_dictionaries(num_cells,num_procs)
  dicts = Dict[]
  for (nc,np) in zip(num_cells,num_procs)
    aux = Dict(
      :np => np,
      :nc => nc
    )
    push!(dicts,aux)
  end
  return dicts
end

###########################################

num_cells = [(6,6,3), (10,10,5),(20,20,5)]
num_procs = [(2,2)  , (4,4)    ,(6,8)]
dicts = generate_dictionaries(num_cells,num_procs)

template = read(projectdir("template.sh"),String)
for params in dicts
   fparams = clean_params(params)
   jobfile = projectdir("jobs/",jobname(fparams,"sh"))
   open(jobfile,"w") do io
    render(io,template,jobdict(params))
   end
end
