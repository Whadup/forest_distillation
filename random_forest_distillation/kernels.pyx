#cython: language_level=3
import cython
import numpy as np
# cimport numpy as np
from libc.stdlib cimport abort, malloc, free
import cython.parallel
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def inner_loop(double b_l,
                double b_r,
                double base_c,
                double [::1] split_values,
                long [::1] split_partitions,
                double last_split_value,
                double [:] values,
                double [:] starts,
                double theta_l, double theta_r, double best_split_gain):
    cdef double best_split_value = -1.0
    cdef double best_split_output_l = -1.0
    cdef double best_split_output_r = -1.0
    cdef double best_split_b_l = -1.0
    cdef double best_split_b_r = -1.0
    cdef double active_partitions = 0.0
    cdef long j
    for j in range(len(split_values)):
        split_value = split_values[j]
        i = split_partitions[j]
        delta = split_value - last_split_value
        v = values[i]
        b_l += delta * active_partitions
        b_r -= delta * active_partitions
        if split_value > starts[i]: # partition ends+
            active_partitions -= v
        else: #partition begins
            active_partitions += v
        last_split_value = split_value
        c_l = base_c * (split_value - theta_l)
        c_r = base_c * (theta_r - split_value)
        if c_r == 0:
            gain = b_l * b_l / c_l
        elif c_l == 0:
            gain = b_r * b_r / c_r
        else:
            gain = b_l * b_l / c_l + b_r * b_r / c_r
        if gain > best_split_gain:
            best_split_gain = gain
            best_split_value = split_value
            if c_l > 0:
                best_split_output_l = b_l / (2 * c_l)
                best_split_b_l = b_l
            if c_r > 0:
                best_split_output_r = b_r / (2 * c_r)
                best_split_b_r = b_r
    
    return (best_split_gain, best_split_value, (best_split_output_l, best_split_output_r), (best_split_b_l, best_split_b_r))

def intervals_intersect(float l1, float r1, float l2, float r2):
    return max(l1, l2) <= min(r1, r2)

def volume_overlap(theta1, theta2):
    l, r = np.maximum(theta1[..., 0], theta2[..., 0]), np.minimum(theta1[..., 1], theta2[..., 1])
    return (np.maximum(l, r) - l).prod(-1)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double inner_bias(double [:] values, double [:, :, ::1] thetas,  Py_ssize_t n, Py_ssize_t m) nogil:
    cdef double overlap = 1.0
    cdef double tmp = 1.0
    cdef double agg2 = 0.0
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t k = 0
    for i in range(n):
        overlap = 1.0
        for k in range(m):
            overlap *= thetas[i, k, 1] - thetas[i, k, 0]
        agg2 += values[i] * overlap #volume_overlap(thetas[i], thetas[j])
    return agg2

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double inner(double [:] values, double [:, :, ::1] thetas, long i, Py_ssize_t n, Py_ssize_t m) nogil:
    cdef double overlap = 1.0
    cdef double tmp = 1.0
    cdef double agg2 = 0.0
    cdef Py_ssize_t j = 0
    cdef Py_ssize_t k = 0
    for j in range(n):
        overlap = 1.0
        for k in range(m):
            tmp = max(thetas[i, k, 0], thetas[j, k, 0])
            overlap *= max(tmp, min(thetas[i, k, 1], thetas[j, k, 1])) - tmp
        agg2 += values[i] * values[j] * overlap #volume_overlap(thetas[i], thetas[j])
    return agg2

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double inner_skewness(double [:] values, double [:, :, ::1] thetas, double [:,::1] theta, Py_ssize_t n, Py_ssize_t m) nogil:
    cdef double overlap = 1.0
    cdef double tmp = 1.0
    cdef double agg2 = 0.0
    cdef Py_ssize_t i,j,k = 0
    cdef Py_ssize_t f = 0
    for i in range(n):
        # theta[:, 0] = thetas[i,:,0]
        # theta[:, 1] = thetas[i,:,1]
        for j in range(n):
            stop = False
            for f in range(m):
                theta[f, 0] = max(thetas[i, f, 0], thetas[j, f, 0])
                theta[f, 1] = min(thetas[i, f, 1], thetas[j, f, 1])
                # theta[f, 0] = min(theta[f, 0], theta[f, 1])
                if theta[f, 1] <= theta[f, 0]:
                    stop = True
                    break
            if stop:
                continue
            for k in range(n):
                overlap = 1.0
                for f in range(m):
                    tmp = max(theta[f, 0], thetas[k, f, 0])
                    overlap *= max(tmp, min(theta[f, 1], thetas[k, f, 1])) - tmp
                    if overlap <= 0:
                        break
                agg2 += values[i] * values[j] * values[k] * overlap #volume_overlap(thetas[i], thetas[j])
    return 0.125 * agg2

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double inner_kurtosis(double [:] values, double [:, :, ::1] thetas, double [:,::1] theta, double [:,::1] theta2, Py_ssize_t n, Py_ssize_t m) nogil:
    cdef double overlap = 1.0
    cdef double tmp = 1.0
    cdef double agg2 = 0.0
    cdef Py_ssize_t i,j,k = 0
    cdef Py_ssize_t f = 0
    for i in range(n):
        # theta[:, 0] = thetas[i,:,0]
        # theta[:, 1] = thetas[i,:,1]
        for j in range(n):
            stop = False
            for f in range(m):
                theta[f, 0] = max(thetas[i, f, 0], thetas[j, f, 0])
                theta[f, 1] = min(thetas[i, f, 1], thetas[j, f, 1])
                # theta[f, 0] = min(theta[f, 0], theta[f, 1])
                if theta[f, 1] <= theta[f, 0]:
                    stop = True
                    break
            if stop:
                continue
            #TODO CRAP THIS SUCKS WE HAVE TO REMEMBER SHIT
            for l in range(n):
                stop = False
                for f in range(m):
                    theta2[f, 0] = max(theta[f, 0], thetas[l, f, 0])
                    theta2[f, 1] = min(theta[f, 1], thetas[l, f, 1])
                    # theta[f, 0] = min(theta[f, 0], theta[f, 1])
                    if theta2[f, 1] <= theta2[f, 0]:
                        stop = True
                        break
                if stop:
                    continue
                for k in range(n):
                    overlap = 1.0
                    for f in range(m):
                        tmp = max(theta2[f, 0], thetas[k, f, 0])
                        overlap *= max(tmp, min(theta2[f, 1], thetas[k, f, 1])) - tmp
                        if overlap <= 0:
                            break
                    agg2 += values[i] * values[j] * values[k] * values[l] * overlap #volume_overlap(thetas[i], thetas[j])
    return 0.0625 * agg2

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def variance(node):
    cdef double [:] values = node.partitions_value
    cdef double [:, :, ::1] thetas = node.partitions_theta
    cdef Py_ssize_t i = 0
    # cdef long j = 0
    # cdef long k = 0
    cdef Py_ssize_t n = values.shape[0]
    cdef Py_ssize_t m = thetas.shape[1]
    
    # cdef double overlap = 1
    # cdef double* overlap
    # print(n)
    cdef double agg = 0.0
    # for i in cython.parallel.prange(n, nogil=True, num_threads=1):
    with nogil:
        for i in range(n):
            agg += 0.25 * inner(values, thetas, i, n, m)
        # overlap = <double *> malloc(sizeof(double) * 2)
        # for j in range(n):
        #     overlap[0] = 1.0
        #     for k in range(m):
        #         overlap[1] = max(thetas[i, k, 0], thetas[j, k, 0])
        #         overlap[0] *= max(overlap[1], min(thetas[i, k, 1], thetas[j, k, 1])) - overlap[1]
        #     agg += values[i] * values[j] * overlap[0] #volume_overlap(thetas[i], thetas[j])
        # free(overlap)
    return agg

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def skewness(node):
    cdef double [:] values = node.partitions_value
    cdef double [:, :, ::1] thetas = node.partitions_theta
    cdef Py_ssize_t i = 0
    # cdef long j = 0
    # cdef long k = 0
    cdef Py_ssize_t n = values.shape[0]
    cdef Py_ssize_t m = thetas.shape[1]
    cdef double [:, ::1] theta = np.zeros((m, 2))
    # cdef double overlap = 1
    # cdef double* overlap
    # print(n)
    cdef double agg = 0.0
    # for i in cython.parallel.prange(n, nogil=True, num_threads=1):
    with nogil:
        agg = inner_skewness(values, thetas, theta, n, m)
    return agg


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def bias(node):
    cdef double [:] values = node.partitions_value
    cdef double [:, :, ::1] thetas = node.partitions_theta
    cdef Py_ssize_t i = 0
    # cdef long j = 0
    # cdef long k = 0
    cdef Py_ssize_t n = values.shape[0]
    cdef Py_ssize_t m = thetas.shape[1]
    
    # cdef double overlap = 1
    # cdef double* overlap
    # print(n)
    cdef double agg = 0.0
    with nogil:
        agg = 0.5 * inner_bias(values, thetas, n, m)
    return agg

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def kurtosis(node):
    cdef double [:] values = node.partitions_value
    cdef double [:, :, ::1] thetas = node.partitions_theta
    cdef Py_ssize_t i = 0
    # cdef long j = 0
    # cdef long k = 0
    cdef Py_ssize_t n = values.shape[0]
    cdef Py_ssize_t m = thetas.shape[1]
    cdef double [:, ::1] theta = np.zeros((m, 2))
    cdef double [:, ::1] theta2 = np.zeros((m, 2))
    # cdef double overlap = 1
    # cdef double* overlap
    # print(n)
    cdef double agg = 0.0
    # for i in cython.parallel.prange(n, nogil=True, num_threads=1):
    with nogil:
        agg = inner_kurtosis(values, thetas, theta, theta2, n, m)
    return agg
