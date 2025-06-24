using LinearAlgebra
using SparseArrays
import Graphs:
    AbstractGraph,
    edges,
    edgetype,
    has_edge,
    has_vertex,
    inneighbors,
    ne,
    nv,
    outneighbors,
    vertices,
    is_directed,
    weights,
    neighbors
import Graphs.LinAlg: laplacian_matrix

# Long term goal being to analyse turbulence over a (sparse) graph structure -> avoid convolutions and perform estimation over time and frequency
struct MyGraph{Tw} <: AbstractGraph{Tw}
    vertices::SparseVector{Int64}
    adjmat::SparseMatrixCSC{Int64}
    weights::SparseMatrixCSC{Tw}
    weights_vertex::Vector{Tw}
end

function MyGraph(adjmat, weights)
    weights_vertex = dropdims(sum(weights, dims = 2), dims = 2)
    return MyGraph(SparseVector(axes(adjmat, 1)), adjmat, weights, weights_vertex)
end
Base.show(io::IO, g::MyGraph{T}) where {T} = print(io, "MyGraph{$T}")

is_directed(G::MyGraph) = false

vertices(x::MyGraph) = findnz(x.vertices)[1]
nv(x::MyGraph) = size(x.adjmat, 1)

has_vertex(x::MyGraph, i::Int64) = x.vertices[i] !== 0
function add_vertex!(x::MyGraph, i::Int64)
    has_vertex(x, i) && throw(error("Vertex $i already present"))
    x.vertices[i] = 1
end
function rem_vertex!(x::MyGraph, i::Int64)
    x.vertices[i] = 0
    dropzeros!(x.vertices)
    rem_vertex!(x.adjmat, i)
    nothing
end
function rem_vertex!(x::SparseMatrixCSC, i::Int64)
    x[i, :] .= 0
    x[:, i] .= 0
    dropzeros!(x)
    nothing
end

# Edges
edgetype(::MyGraph) = Int64
edges(x::MyGraph) = _edges(x.adjmat)
function _edges(adjmat::SparseMatrixCSC)
    (I, J, _) = findnz(adjmat)
    (i => j for (i, j) in zip(I, J))
end

ne(x::MyGraph) = nnz(x.adjmat)

has_edge(x::MyGraph, i::Int64, j::Int64) = _has_edge(x.adjmat, i, j)
_has_edge(x::SparseMatrixCSC, i::Int64, j::Int64) = x[i, j] != 0
function add_edge!(x::MyGraph, i::Int64, j::Int64)
    add_edge!(x.adjmat, i, j)
    nothing
end
function add_edge!(x::SparseMatrixCSC, i::Int64, j::Int64)
    x[i, j] = 1
    x[j, i] = 1
    nothing
end
function rem_edge!(x::MyGraph, i::Int64, j::Int64)
    rem_edge!(x.adjmat, i, j)
    nothing
end
function rem_edge!(x::SparseMatrixCSC, i::Int64, j::Int64)
    x[i, j] = 0
    x[j, i] = 0
    dropzeros!(x)
    nothing
end

# Neighbors
function neighbors(x::MyGraph, i::Int64)
    has_vertex(x, i) || throw(error("No vertex $i"))
    _neighbors(x.adjmat, i)
end
_neighbors(x::SparseMatrixCSC, i::Int64) = findnz(x[i, :])[1]

inneighbors(x::MyGraph, i) = neighbors(x, i)
outneighbors(x::MyGraph, i) = neighbors(x, i)

# Weights
total_weight(g::MyGraph) = sum(g.weights)
function edge_weight(g::MyGraph, i, j)
    return g.weights[i, j]
end

function vertex_weight(g::MyGraph, i)
    return g.weights_vertex[i]
end

function update_weight!(g::MyGraph, i::Int64, j::Int64, value::Float64)
    has_edge(g, i, j) || throw(error("No edges between $i and $j"))
    update_weight!(g.weights, i, j, value)
    return nothing
end
function update_weight!(x::SparseMatrixCSC, i, j, value)
    x[i, j] = value
    x[j, i] = value
end

function update_vertex_weight!(g::MyGraph, i)
    g.weights_vertex[i] = sum(g.weights[i, :])
    nothing
end

function add_weight!(g::MyGraph, i, j, value)
    w = edge_weight(g, i, j) + value
    update_weight!(g, i, j, w)
end

weights(g::MyGraph) = g.weights


function extend(g::MyGraph, n = size(g.weights, 1))
    S = (n, n)
    weights = blockdiag(g.weights, spzeros(eltype(g.weights), S))
    adjmat = blockdiag(g.adjmat, spzeros(eltype(g.adjmat), S))
    return MyGraph(adjmat, weights)
end

function laplacian_matrix(g::MyGraph)
    return g.weights - spdiagm(g.weights_vertex)
end

loc_grid(i, j, n, m) = (max(i - 1, 1):min(i + 1, n), max(j - 1, 1):min(j + 1, m))
function count_edges(m, n) # 9-point stencil grid, non-periodic
    n_corner = 4
    n_side = (m + n - n_corner) * 2
    n_rem = n * m - n_corner - n_side
    e_corner = n_corner * 3
    e_side = n_side * 5
    e_rem = n_rem * 8
    return e_corner + e_side + e_rem
end

# Generate 9-point stencil grid adjacency matrix
function grid_adj_mat(size_mat, mask = trues(size_mat))
    l_idx = LinearIndices(size_mat)
    m, n = size_mat
    #n_edges = count_edges(m,n)
    I = Int64[]#(undef,n_edges)
    J = Int64[]#(undef,n_edges)
    k = 0
    for i = 1:m, j = 1:n
        if !mask[i, j]
            continue
        end
        l_grid = loc_grid(i, j, m, n)
        for i_g in l_grid[1], j_g in l_grid[2]
            if i_g == i && j_g == j || !mask[i_g, j_g]
                continue
            else
                push!(I, l_idx[i, j])
                push!(J, l_idx[i_g, j_g])
                k += 1
            end
        end
    end
    V = ones(Int64, length(I))
    adj_mat = sparse(I, J, V, m * n, n * m)
    return adj_mat
end

# Generate a weight matrix given an adjacency matrix and a weight function. The weight function takes linearindices, i.e. If the vertices reference the entry of an AbstractArray, the weight between two vertice i and j is computed using the weight function with the linear index of both vertices.
function generate_weight_mat(
    adj_mat::SparseMatrixCSC,
    weight_func::Function;
    normalize = false,
)
    I, J, A = findnz(adj_mat)
    W = Array{Float64}(undef, length(A))
    for (k, (i, j)) in enumerate(zip(I, J))
        W[k] = weight_func(i, j)
    end
    if normalize
        weight_mat = sparse(I, J, W, size(adj_mat)...)
        weight_vertex = sum(weight_mat, dims = 2)
        for k = 1:length(W)
            W[k] = W[k] / weight_vertex[I[k]]
        end
    end
    return sparse(I, J, W, size(adj_mat)...)
end

function make_gaussian_kernel(size_Z, vertex_mapping::Function, Sigma::AbstractMatrix)
    C = CartesianIndices(size_Z)
    function weight_func(i::Int, j::Int)
        c_i = C[i]
        c_j = C[j]
        v_i = vertex_mapping(c_i)
        v_j = vertex_mapping(c_j)
        v = v_i - v_j
        d = dot(v, v' / Sigma)
        return exp(-0.5 * d)
    end
end

function apply_func_on_edges(adj_mat::SparseMatrixCSC, func::Function)
    E = _edges(adj_mat)
    func_type = Base.return_types(func)[1]
    out = Vector{func_type}(undef, length(E))
    for (k, (i, j)) in enumerate(E)
        out[k] = func(i, j)
    end
    return out
end

apply_func_on_edges(g::MyGraph, func::Function) = apply_func_on_edges(g.adjmat, func)
