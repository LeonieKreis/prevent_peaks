import numpy as np
import torch

m_in = 4 # 64,128,256
m_out = 1 # 64,128,256
n = 4 # 32,16,8
input_shape = (n, n)
kernel = torch.randn(3, 3, m_in, m_out)

def SingularValues(kernel, input_shape):
    transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    #u, s, v = np.linalg.svd(transforms, compute_uv=True)
    s = np.linalg.svd(transforms, compute_uv=False)
    return s


sv = SingularValues(kernel, input_shape)
#print(sv)
sv_f = np.ndarray.flatten(sv)
#print(sv_f)
max_sv = max(sv_f)
k_2 = torch.norm(kernel, p=2)
k_1 = torch.norm(kernel, p=1)

print(f'Kernel fro (2) norm: {k_2}')
print(f'Max Singular value: {max_sv}')
print(f'kernel 1 norm: {k_1}')
print(f'ratio 1 norm to max sing value: {k_1 / max_sv}')
print(f'ratio fro norm to max sing value: {k_2 / max_sv}')

