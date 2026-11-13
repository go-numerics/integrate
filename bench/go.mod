module github.com/go-numerics/integrate/bench

go 1.27.3

require (
	github.com/go-ndarray/ndarray v0.41.0
	github.com/go-numerics/integrate v0.2.2
	github.com/go-scikit/refbench v0.4.0
)

require (
	github.com/go-fleettools/fleettools v0.81.0 // indirect
	github.com/go-simd/floats v0.17.0 // indirect
	golang.org/x/sys v0.49.0 // indirect
)

replace github.com/go-numerics/integrate => ../
