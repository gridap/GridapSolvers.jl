
using Literate

src_dir = string(@__DIR__,"/../test/Applications/")
out_dir = string(@__DIR__,"/src/Examples/")

mkpath(out_dir)

for file in readdir(src_dir)
  if !endswith(file,".jl")
    continue
  end
  in_file = string(src_dir,file)
  out_file = string(out_dir,file)

  Literate.markdown(
    in_file, out_dir
  )
end
