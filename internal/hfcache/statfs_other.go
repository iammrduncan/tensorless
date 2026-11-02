//go:build !linux && !darwin

package hfcache

import "errors"

// FreeSpace is not implemented on this platform.
func FreeSpace(dir string) (uint64, error) {
	return 0, errors.ErrUnsupported
}
