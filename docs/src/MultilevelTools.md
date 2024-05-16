
```@meta
CurrentModule = GridapSolvers.MultilevelTools
```

# GridapSolvers.MultilevelTools

## Nested subpartitions

One of the main difficulties of multilevel algorithms is dealing with the complexity of having multiple subcommunicators. We provide some tools to deal with it. In particular we introduce `HierarchicalArray`s.

```@docs
generate_level_parts
HierarchicalArray
Base.map
with_level
```

## ModelHierarchies and FESpaceHierarchies

This objects are the multilevel counterparts of Gridap's `DiscreteModel` and `FESpace`.

```@docs
ModelHierarchy
ModelHierarchyLevel
CartesianModelHierarchy
FESpaceHierarchy
```

## Grid transfer operators

To move information between different levels, we will require grid transfer operators. Although any custom-made operator can be used, we provide some options.

```@docs
DistributedGridTransferOperator
RestrictionOperator
ProlongationOperator
MultiFieldTransferOperator
```

## Misc

```@docs
LocalProjectionMap
```
