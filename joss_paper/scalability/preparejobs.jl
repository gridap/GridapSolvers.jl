
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

function get_np_per_level(np,nl)
  a = fill(np,nl)
  if prod(np) > 768
    a[end] = (24,32) # 768
  end
  return a
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
    "np" => np,
    "nc" => nc,
    "np_per_level" => get_np_per_level(np,nl),
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

# Even number of nodes

n = 4
n_nodes  = [(2,2).^i for i in 0:n]
n_procs  = [(1,1),(3,4),[(6,8).*nn for nn in n_nodes]...]
n_levels = [2,2,[i+2 for i in 0:n]...]

c = (6,8) .* 8
n_cells_coarse = [c,c.*(3,4),[c.*(6,8) for i in 0:n]...]
n_cells_global = map((nc,nl) -> nc .* (2^(nl-1),2^(nl-1)),n_cells_coarse,n_levels)
@assert all(r -> r == prod(n_cells_global[1])/prod(n_procs[1]),map(prod,n_cells_global)./map(prod,n_procs))
@assert all(map((N,n,nl) -> prod(N) == prod(n)*4^(nl-1), n_cells_global,n_cells_coarse,n_levels))

dicts_even_gmg = generate_dictionaries(n_procs,n_cells_coarse,n_levels)
dicts_even_petsc = generate_dictionaries(n_procs,n_cells_global,n_levels)

# Odd number of nodes

n = 3
n_nodes  = [(2,1).*(2,2).^i for i in 0:n]
n_procs  = [(6,8).*nn for nn in n_nodes]
n_levels = [i+2 for i in 0:n]

n_cells_coarse = [c.*(12,8) for i in 0:n]
n_cells_global = map((nc,nl) -> nc .* (2^(nl-1),2^(nl-1)),n_cells_coarse,n_levels)

dicts_odd_gmg = generate_dictionaries(n_procs,n_cells_coarse,n_levels)
dicts_odd_petsc = generate_dictionaries(n_procs,n_cells_global,n_levels)

gmg = true
if gmg 
  template_file = "template_gmg.sh"
  dicts = vcat(dicts_odd_gmg,dicts_even_gmg)
else
  template_file = "template.sh"
  dicts = vcat(dicts_odd_petsc,dicts_even_petsc)
end

template = read(projectdir(template_file),String)
for params in dicts
   fparams = clean_params(params)
   jobfile = projectdir("jobs/",jobname(fparams,"sh"))
   open(jobfile,"w") do io
    render(io,template,jobdict(params))
   end
end
