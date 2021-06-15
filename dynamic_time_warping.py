import numpy as np
import itertools as it
import sys

sys.setrecursionlimit(5000)


class DynamicTimeWarping:

    def __init__(self):
        self.z = None
        self.D = None
        self.TS_1 = None
        self.TS_2 = None
        self.bucket_size = None
        self.dist_func = None

    def _dist(self, t1, t2, x, y):
        dict_ = {
            'Abs': abs(t1[x] - t2[y]),
            'Eucl': np.sqrt((t1[x] ** 2) + (t2[y] ** 2))
        }
        return dict_.get(self.dist_func)

    def _method_forward(self, T1, T2, i, j):
        i_max = len(self.z[:, 0]) - 1
        j_max = len(self.z[0, :]) - 1
        distance = self._dist(T1, T2, i, j)
        if j >= j_max and i < i_max:
            if i <= 0:
                self.z[i, j, 0] = distance + self.z[i, j - 1, 0]
                self.z[i, j, 1] = self.z[i, j - 1, 1] + 1
            else:
                self._UpdateIntermediate(i, j, distance)
            self.z = self._method_forward(T1, T2, i + 1, 0)
        elif i >= i_max and j >= j_max:
            self._UpdateIntermediate(i, j, distance)
        else:
            if i <= 0 and j <= 0:
                self.z[i, j, 0] = distance
                self.z[i, j, 1] = 1
            elif i <= 0 and j > 0:
                self.z[i, j, 0] = distance + self.z[i, j - 1, 0]
                self.z[i, j, 1] = self.z[i, j - 1, 1] + 1
            elif j <= 0 and i > 0:
                self.z[i, j, 0] = distance + self.z[i - 1, j, 0]
                self.z[i, j, 1] = self.z[i - 1, j, 1] + 1
            else:
                self._UpdateIntermediate(i, j, distance)
            self.z = self._method_forward(T1, T2, i, j + 1)
        return self.z

    def _method_backward(self, T1, T2, i, j, z_ind):
        z_i = z_ind[0] - len(T1) + i
        z_j = z_ind[1] - len(T2) + j
        # calculate the _distance
        _distance = self._dist(T1, T2, i, j)
        if j <= 0 and i > 0:
            # Recursive throygh i ---> ROWS
            # --! remember to reset j !--#
            self.z = self._method_backward(T1, T2, i - 1, len(T2) - 1, z_ind)
            # In this point we've just stop to the column of the sub-array
            # must see now if its also the first column of the Array
            if z_j <= 0:
                self.z[z_i, z_j, 0] = _distance + self.z[z_i - 1, z_j, 0]
                self.z[z_i, z_j, 1] = self.z[z_i - 1, z_j, 1] + 1
            else:
                # Intermediate Point
                self._UpdateIntermediate(z_i, z_j, _distance)
        elif i <= 0 and j <= 0:
            # First element (0,0) of the Sub-Array
            # Here we are at the begin of each subarray and must fill with the position on the
            if z_i <= 0 and z_j <= 0:
                # if also is the first element of the Array
                self.z[z_i, z_j, 0] = _distance
                self.z[z_i, z_j, 1] = 1
            elif z_i <= 0 and z_j > 0:
                # if it is only in the first row of the Array
                self.z[z_i, z_j, 0] = _distance + self.z[z_i, z_j - 1, 0]
                self.z[z_i, z_j, 1] = self.z[z_i, z_j - 1, 1] + 1
            elif z_i > 0 and z_j <= 0:
                # if it is only in the first column of the Array
                self.z[z_i, z_j, 0] = _distance + self.z[z_i - 1, z_j, 0]
                self.z[z_i, z_j, 1] = self.z[z_i - 1, z_j, 1] + 1
            else:
                # Intermediate Point
                self._UpdateIntermediate(z_i, z_j, _distance)
        else:
            # Must call the recursive function
            # Recursive through j ---> COLUMNS
            self.z = self._method_backward(T1, T2, i, j - 1, z_ind)
            # Now must find the value
            if z_i <= 0:
                # if the point is at the first row of the Array & the Sub-Array
                # but not in the first column of any array
                self.z[z_i, z_j, 0] = _distance + self.z[z_i, z_j - 1, 0]  # must add only the left element
                self.z[z_i, z_j, 1] = self.z[z_i, z_j - 1, 1] + 1
            else:
                # Intermediate Point
                self._UpdateIntermediate(z_i, z_j, _distance)
        return self.z

    def _CalculateWithBuckets(self, TS_1, TS_2, bag_size):
        TimSer_1 = self._CreateBuckets(TS_1, bag_size)
        TimSer_2 = self._CreateBuckets(TS_2, bag_size)
        ind_1 = list(range(len(TimSer_1)))
        ind_2 = list(range(len(TimSer_2)))
        indices = list(it.product(ind_1, ind_2))
        for (k, l) in indices:
            z_indices = (k * len(TimSer_1[0]) + len(TimSer_1[k]), l * len(TimSer_2[0]) + len(TimSer_2[l]))
            self.z = self._method_backward(TimSer_1[k], TimSer_2[l], len(TimSer_1[k]) - 1, len(TimSer_2[l]) - 1,
                                           z_indices)
        return

    def _CreateBuckets(self, Serie, bucket_size):
        cnt = 0
        Aux = []
        for i in range(bucket_size, len(Serie), self.bucket_size):
            Aux.append(list(Serie[cnt:i]))
            cnt += bucket_size
        Aux.append(list(Serie[cnt:]))
        return Aux

    def _UpdateIntermediate(self, i, j, _distance):
        # find the min value of the three previous values
        min_val = min(self.z[i - 1, j, 0], self.z[i, j - 1, 0], self.z[i - 1, j - 1, 0])
        # update the value of the z Array
        self.z[i, j, 0] = _distance + min_val
        # Update the counte of the z Array(depends on the condition)
        if min_val == self.z[i - 1, j - 1, 0]:
            self.z[i, j, 1] = self.z[i - 1, j - 1, 1] + 1
        elif min_val == self.z[i, j - 1, 0]:
            self.z[i, j, 1] = self.z[i, j - 1, 1] + 1
        else:
            self.z[i, j, 1] = self.z[i - 1, j, 1] + 1
        return

    def add(self, t, serie_id=1):
        if serie_id == 2:
            z_aux = np.zeros((len(self.TS_1), len(t), 2))
            self.z = np.concatenate((self.z, z_aux), axis=1)
            start = len(self.TS_2) - 1
            self.TS_2 = np.concatenate((self.TS_2, t))
            self.forward(0, start)
        else:
            z_aux = np.zeros((len(t), len(self.TS_2), 2))
            self.z = np.concatenate((self.z, z_aux), axis=0)
            start = len(self.TS_1) - 1
            self.TS_1 = np.concatenate((self.TS_1, t))
            self.forward(start, 0, 0)

        dist = self.z[-1, -1, 0]
        steps = self.z[-1, -1, 1]
        self.D = dist / steps

    def delete(self, num, serie_id=1, dist_func=None, method='forward', bucket_size=1):
        ind = np.arange(num)
        if serie_id == 'both':
            self.TS_1 = np.delete(self.TS_1, ind, axis=0)
            self.TS_2 = np.delete(self.TS_1, ind, axis=0)
        elif serie_id == 1:
            self.TS_1 = np.delete(self.TS_1, ind, axis=0)
        elif serie_id == 2:
            self.TS_2 = np.delete(self.TS_2, ind, axis=0)
        self.get(self.TS_1, self.TS_2, bucket_size, method, dist_func)

    def backward(self, i, j, k):
        self._CalculateWithBuckets(self.TS_1, self.TS_2, k)

    def forward(self, i, j, k):
        self.z = self._method_forward(self.TS_1, self.TS_2, i, j)

    def get(self, T1, T2, bucket_size=10, method='backward', dist_func='Abs'):
        if dist_func is not None:
            self.dist_func = dist_func
        self.TS_1 = T1
        self.TS_2 = T2
        self.bucket_size = bucket_size
        self.z = np.zeros((len(self.TS_1), len(self.TS_2), 2))
        getattr(self, method)(0, 0, bucket_size)
        dist = self.z[-1, -1, 0]
        steps = self.z[-1, -1, 1]
        self.D = dist / steps
