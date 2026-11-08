//go:build race

package integrate

// raceOn: sync.Pool drops items at random under the race detector, so an
// allocation count that relies on a pool is not exact there.
const raceOn = true
