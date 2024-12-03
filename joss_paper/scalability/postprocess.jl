### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ f0aae79e-13d9-11ef-263f-b720d8f10878
begin
	using Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()
	
	using Plots, DrWatson
	using DataFrames, BSON, CSV
	using Printf
end

# ╔═╡ 4725bb5d-8973-436b-af96-74cc90e7f354
begin 
	raw = collect_results(datadir("run_amg"))

	dfgr = groupby(raw,[:np])
	df = combine(dfgr) do df
		return (
			t_solver = minimum(map(x->x[:max],df.Solver)),
			t_setup  = minimum(map(x->x[:max],df.SolverSetup)),
			n_iter   = df.niter[1],
			n_levels = length(df.np_per_level[1]),
			n_dofs_u = df.ndofs_u[1],
			n_dofs_p = df.ndofs_p[1],
			n_dofs   = df.ndofs_u[1] + df.ndofs_p[1],
			n_cells  = df.ncells[1],
			n_procs  = prod(df.np[1])
		)
	end
	sort!(df,:n_procs)

	f(row) = row.n_procs ∉ []
	filter!(f,df)
end

# ╔═╡ a10880e9-680b-46f0-9bda-44e53a7196ce
begin
	plt1 = plot(xlabel="",ylabel="Time (s)",legend=false)
	plot!(plt1,df[!,:n_procs],df[!,:t_solver]./df[!,:n_iter],marker=:circ)
	#savefig(plt,projectdir("../weakScalability"))
end

# ╔═╡ 8d1b090e-2548-48fe-afb9-92f044442a80
begin
	plt2 = plot(xlabel="",ylabel="Num Levels",legend=false)
	plot!(plt2,df[!,:n_procs],df[!,:n_levels],marker=:circ)
end

# ╔═╡ f63c1451-e3ad-41bb-851b-0366eb71c0cc
begin
	plt3 = plot(xlabel="Number of processors",ylabel="Num Iter",legend=false)
	plot!(plt3,df[!,:n_procs],df[!,:n_iter],marker=:circ)
end

# ╔═╡ 7ed39f44-3648-4af6-9e7a-dc16d1f66a12
begin
	plt_agg = plot(plt1,plt2,plt3,layout=(3,1),link=:x)
	savefig(plt_agg,projectdir("../weakScalability"))
	plt_agg
end

# ╔═╡ 15178e9a-4da9-488c-be03-74abba5824a6
begin
	hline(n,w) = "|:"*repeat('-',12)*":|" * repeat(":"*repeat('-',w)*":|",n) * "\n"
	format(x::Real) = @sprintf("%4.1f",x)
	format(x::Integer) = @sprintf("%4i",x)
	row(name,data) = "|" * name * "| " * join(format.(data)," | ") * " |\n"
	
	ncols = length(df[!,:n_procs])
	table = ""
	table = table * row("Num Procs     ",df[!,:n_procs])
	table = table * hline(ncols,4)
	table = table * row("Time/Iter (s) ",df[!,:t_solver]./df[!,:n_iter])
	table = table * row("Num Levels    ",df[!,:n_levels])
	table = table * row("Num Iters     ",df[!,:n_iter])
	print(table)
end

# ╔═╡ Cell order:
# ╠═f0aae79e-13d9-11ef-263f-b720d8f10878
# ╠═4725bb5d-8973-436b-af96-74cc90e7f354
# ╠═a10880e9-680b-46f0-9bda-44e53a7196ce
# ╠═8d1b090e-2548-48fe-afb9-92f044442a80
# ╠═f63c1451-e3ad-41bb-851b-0366eb71c0cc
# ╠═7ed39f44-3648-4af6-9e7a-dc16d1f66a12
# ╠═15178e9a-4da9-488c-be03-74abba5824a6
