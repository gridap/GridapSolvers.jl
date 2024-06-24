
using Mustache
using DrWatson

function jobname(args...)
  s = savename(args...;connector="_")
  s = replace(s,"="=>"_")
  s = "stokes_$s"
  return s
end

function clean_params(d)
  o = Dict()
  for k in keys(d)
    if k == :nc
      #o[k] = "$(prod(d[k])÷1000)k"
    elseif isa(d[k],Real)
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
  nl = params[:nl]
  fparams = clean_params(params)
  Dict(
    "project" => haskey(ENV,"PROJECT") ? ENV["PROJECT"] : "",
    "q" => "normal",
    "o" => datadir(jobname(fparams,"o")),
    "e" => datadir(jobname(fparams,"e")),
    "walltime" => "01:00:00",
    "ncpus" => prod(np),
    "nnodes" => prod(np)÷48,
    "mem" => "$(prod(np)*4)gb",
    "name" => jobname(fparams),
    "nr" => 5,
    "np" => prod(np),
    "nc" => nc,
    "np_per_level" => fill(prod(np),nl),
    "projectdir" => projectdir(),
    "datadir"    => datadir(),
    "modules"    => projectdir("modules.sh"),
    "driverdir"  => projectdir(),
    "title"      => datadir(jobname(fparams)),
    "sysimage"   => projectdir("Scalability.so")
  )
end

function generate_dictionaries(n_procs,n_cells,n_levels)
  dicts = Dict[]
  for (np,nc,nl) in zip(n_procs,n_cells,n_levels)
    aux = Dict(
      :np => np,
      :nc => nc,
      :nl => nl
    )
    push!(dicts,aux)
  end
  return dicts
end

###########################################

n = 4
n_nodes  = [4^i for i in 0:n]
n_procs  = [1,12,48 .* n_nodes...]
n_levels = [2,2,[i+2 for i in 0:n]...]

c = (60,60)
n_cells_global = [c.*(2,2),[c.*(3,4).*(2^(i+1),2^(i+1)) for i in 0:n+1]...]
n_cells_coarse = [c,c.*(3,4),[c.*(6,8) for i in 0:n]...]
@assert all(r -> r == prod(n_cells_global[1])/n_procs[1],map(prod,n_cells_global)./n_procs)
@assert all(map((N,n,nl) -> prod(N) == prod(n)*4^(nl-1), n_cells_global,n_cells_coarse,n_levels))

dicts = generate_dictionaries(n_procs,n_cells_coarse,n_levels)

template = read(projectdir("template.sh"),String)
for params in dicts
   fparams = clean_params(params)
   jobfile = projectdir("jobs/",jobname(fparams,"sh"))
   open(jobfile,"w") do io
    render(io,template,jobdict(params))
   end
end
