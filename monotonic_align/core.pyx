cimport cython
from cython.parallel import prange
from libc.stdio cimport printf


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void maximum_path_each(int[:,::1] path, float[:,::1] value, int t_y, int t_x, float max_neg_val=-1e9) nogil:
  cdef int x
  cdef int y
  cdef int direction
  cdef float v_prev
  cdef float v_left
  cdef float v_up
  cdef float tmp
  cdef int index = t_x - 1

  for y in range(t_y):
    for x in range(t_x):
      if x == 0:
        v_up = max_neg_val
        if y == 0:
          v_left = max_neg_val
          v_prev = 0.
        else:
          v_left = value[y-1, x]
          v_prev = max_neg_val
      else:
        if y == 0:
          v_up = value[y, x-1]
          v_left = max_neg_val
          v_prev = max_neg_val
        else:
          v_prev = value[y-1, x-1]
          v_up = value[y, x-1]
          v_left = value[y-1, x]

      tmp = max(value[y, x] + v_prev, value[y, x] + v_up, v_left)
      if value[y, x] + v_prev == tmp:
        path[y, x] = 1
      else:
        if value[y, x] + v_up == tmp:
          path[y, x] = 2
        else:
          path[y, x] = 3

      value[y, x] = tmp

  # """
  for y in range(t_y - 1, -1, -1):
    # printf(b"------------------------\n")
    # printf(b"y: %d\n", y)
    # printf(b"i: %d\n", index)

    if index == -1:
      break

    direction = path[y, index]
    # printf(b"d: %d\n", direction)

    while direction == 2:
      path[y, index] = 0
      index = index - 1
      if index == -1:
        break
      direction = path[y, index]
      # printf(b"i: %d\n", index)
      # printf(b"d: %d\n", direction)

    if direction == 1:
      path[y, index] = 0
      index = index - 1
      continue
    if direction == 3:
      continue
    
  # """

  


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void maximum_path_c(int[:,:,::1] paths, float[:,:,::1] values, int[::1] t_ys, int[::1] t_xs) nogil:
  cdef int b = paths.shape[0]
  cdef int i
  for i in prange(b, nogil=True):
    maximum_path_each(paths[i], values[i], t_ys[i], t_xs[i])
