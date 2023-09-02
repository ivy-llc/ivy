import torch
F = torch.nn.functional
import numpy as np
import ivy
from ivy.functional.frontends.torch.nn.functional.vision_functions import grid_sample
from copy import deepcopy

if __name__ == "__main__":
    # 2D
    # n, h, w, c = 12, 12, 7, 3
    # to_h, to_w = 7, 11
    # image = np.random.randn(n, c, h, w)
    # grid = np.random.randint(0, 4, size=(n, to_h, to_w, 2)) * 1.0
    # grid = np.zeros((n, to_h, to_w, 2))
    # grid = np.random.randn(n, to_h, to_w, 2)

    # 3D
    # n, h, w, d, c = 3, 12, 5, 7, 4
    # to_h, to_w, to_d = 6, 3, 4
    # image = np.random.randn(n, c, d, h, w)
    # grid = np.random.randn(n, to_d, to_h, to_w, 3)
    # grid = np.zeros((n, to_d, to_h, to_w, 3))
    # grid = np.random.randint(0, 4, size=(n, to_d, to_h, to_w, 3)) * 1.0

    # Custom
    image = np.array([[[[-0.5 , -0.5 , -0.5 , -0.5 , -0.5 ],
                     [-0.5 , -0.5 , -0.5 , -0.5 , -0.5 ],
                     [-0.5 , -0.5 , -0.5 , -0.5 , -0.5 ],
                     [-0.5 , -0.5 , -0.5 , -0.5 , -0.5 ],
                     [-0.5 , -0.5 , -0.5 , -0.5 , -0.75]]]])
    grid = np.array([[[[-0.5, -0.5],
                     [-0.5, -0.5]],
                    [[-0.5, -0.5],
                     [-0.5, -0.5]]]])
    # image = np.array([[[[-5.00000000e-01, -5.00000000e-01, -5.00000000e-01,
    #                   -5.00000000e-01, -5.00000000e-01],
    #                  [-5.00000000e-01, -5.00000000e-01, -1.19209290e-07,
    #                   -5.96046448e-08, -5.96046448e-08],
    #                  [-5.00000000e-01, -5.00000000e-01, -5.00000000e-01,
    #                  -5.00000000e-01, -9.09090909e-01],
    #                  [-5.00000000e-01, -5.00000000e-01, -5.00000000e-01,
    #                   -5.00000000e-01, -5.00000000e-01],
    #                  [-5.00000000e-01, -5.00000000e-01, -5.00000000e-01,
    #                   -5.00000000e-01, -1.00000000e-05]]]])
    # grid = np.array([[[[-1.19209290e-07, -5.00000000e-01],
    #                  [-6.66133813e-16, -6.66133813e-16]],
    #
    #                 [[-6.66133813e-16, -5.00000000e-01],
    #                  [-5.00000000e-01, -5.00000000e-01]]]])


    mode = 'nearest'
    padding_mode = 'border'
    align_corner = False


    output = F.grid_sample(deepcopy(torch.Tensor(image)), deepcopy(torch.Tensor(grid)), mode=mode, align_corners=align_corner, padding_mode=padding_mode)
    print(f"\nOutput({mode}, align_corner={align_corner}): {output.shape} {padding_mode}")
    print(output)

    image_i, grid_i = ivy.array(image), ivy.array(grid)
    output_ivy = grid_sample(deepcopy(image_i), deepcopy(grid_i), mode=mode, align_corners=align_corner, padding_mode=padding_mode)
    print(f"ivy Output({mode}, align_corner={align_corner} (shape:{output_ivy.shape})")
    print(output_ivy)
    print(np.allclose(output, output_ivy.ivy_array.to_numpy(), atol=1e-3))



# import ivy.functional.frontends.torch as torch_frontend
# import torch
# import numpy as np
#
# if __name__ == '__main__':
#     inp = torch.randn((3, 4, 4, 4))
#     grid = torch.randint(0, 4, size=(3, 4, 4, 2), dtype=torch.float32)
#
#     mode = 'bilinear'
#     padding = 'reflection'
#
#     result_orig = torch.nn.functional.grid_sample(inp.clone(), grid.clone(), mode=mode, padding_mode=padding,
#                                                   align_corners=False)
#     print('native fn result shape', result_orig.shape)
#
#     result_new = torch_frontend.nn.functional.grid_sample(inp.clone(), grid.clone(), mode=mode, padding_mode=padding,
#                                                           align_corners=False)
#     print('frontend fn result shape', result_new.shape)
#
#     print(np.allclose(result_orig, result_new.ivy_array.to_numpy(), atol=1e-4))


# import ivy.functional.frontends.torch as torch_frontend
# import torch
# import numpy as np
# import ivy
#
# if __name__ == '__main__':
#     ivy.set_backend('numpy')
#
#     inp = torch.randn((3, 3, 4, 2))
#     # grid = torch.randint(0, 4, size=(3, 4, 4, 2), dtype=torch.float32)
#     grid = torch.randn(size=(3, 4, 4, 2))
#
#     mode = 'nearest'
#     padding = 'zeros'
#     align_corners = False
#
#     result_orig = torch.nn.functional.grid_sample(inp.clone(), grid.clone(), mode=mode, padding_mode=padding,
#                                                   align_corners=align_corners)
#     print('native fn result shape', result_orig.shape)
#
#     result_new = torch_frontend.nn.functional.grid_sample(ivy.array(inp.clone().detach().numpy()),
#                                                           ivy.array(grid.clone().detach().numpy()), mode=mode,
#                                                           padding_mode=padding, align_corners=align_corners)
#     print('frontend fn result shape', result_new.shape)
#
#     print(np.allclose(result_orig, result_new.ivy_array.to_numpy(), atol=1e-4))
    # print(result_orig[0])
    # print(result_new.ivy_array.to_numpy()[0])

