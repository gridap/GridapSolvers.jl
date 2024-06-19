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
end

# ╔═╡ 4725bb5d-8973-436b-af96-74cc90e7f354
begin 
	raw = collect_results(datadir())

	dfgr = groupby(raw,[:np])
	df = combine(dfgr) do df
		return (
			t_solver = minimum(map(x->x[:max],df.Solver)),
			t_setup  = minimum(map(x->x[:max],df.Setup)),
			n_iter   = df.niter[1],
			n_levels = length(df.np_per_level[1]),
			n_dofs_u = df.ndofs_u[1],
			n_dofs_p = df.ndofs_p[1],
			n_dofs   = df.ndofs_u[1] + df.ndofs_p[1],
			n_cells  = df.ncells[1]
		)
	end
	sort!(df,:np)
end

# ╔═╡ a10880e9-680b-46f0-9bda-44e53a7196ce
begin
	plt = plot(xlabel="N processors",ylabel="walltime (s)",legend=false)
	plot!(df[!,:np],df[!,:t_solver])
end

# ╔═╡ Cell order:
# ╠═f0aae79e-13d9-11ef-263f-b720d8f10878
# ╠═4725bb5d-8973-436b-af96-74cc90e7f354
# ╠═a10880e9-680b-46f0-9bda-44e53a7196ce
