import torch
import torch.nn as nn
import torch.nn.functional as F

#focal = 1000

def createDepthHistograms(depths, sizes):

    hists = []
    # Three sizes for three yolo levels
    for size in sizes:
        # Patch to compute depth histogram from
        patch_size = depths.shape[2]//size[0]
        # As many bins as the width (seems reasonable)
        bin_size = size[1]
        # Reshape depth images into batch of patches
        patches = depths.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        bsize, _, hP, wP, _, _ = patches.shape
        # Reshape so all patches are in the batch dimension
        patches = patches.reshape(-1, patch_size, patch_size)
        # Flatten all pixels in patches
        patches = patches.view(-1, patch_size*patch_size)
        # Compute histogram
        hists.append(torch.stack([F.normalize(torch.histc(x, bins=20, min=0, max=40),dim=0) for x in patches]).view(bsize, hP, wP, -1))

    return hists

def generateRays(poses, sizes, imSize, device = 'cuda:0'):

    intr, extr = poses

    focal = intr[0, 0]
    #px = intr[0, 2]
    #py = intr[1, 2]

    baseRays = []
    # Three sizes for three yolo levels
    for size in sizes:
        patch_size = imSize//size[1]
        # Image coordinate grid
        y = torch.arange(0, size[0])
        x = torch.arange(0, size[1])
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        # Offset with actual sizes
        grid_x = grid_x*patch_size + patch_size//2
        grid_y = grid_y*patch_size + patch_size//2
        # Principal points in image center
        px = imSize//2
        py = imSize//4
        # Compute rays
        rays = torch.tensor([[(u-px)/focal, -(v-py)/focal, 1] for u,v in zip(grid_x.flatten(), grid_y.flatten())]).float().to(device)
        baseRays.append(rays)

    # Transform rays
    rays = []
    for bRay in baseRays:
        trRays = []
        for pose in extr:
            # Use just rotation, translation will be accounted for later on
            trRay = torch.matmul(pose[:3, :3], bRay.T).T
            trRays.append(trRay)
        rays.append(torch.stack(trRays))
    return rays

def getIntersectionWeights(rays, poses, hists, device = 'cuda:0'):

    nviews, HW, _ = rays.shape

    weights = torch.diag(torch.ones(HW).to(device))
    weights = weights.repeat(nviews, nviews)

    alpha = 50

    poses = poses[1]

    pos = poses[:, 0:3, 3]

    indices = torch.arange(HW * HW).to(device)

    for i in range(nviews-1):
        for j in range(i+1, nviews):

            dist = pos[j]-pos[i]

            query_rays = rays[i].view(-1, 3)
            key_rays = rays[j].view(-1, 3)
            num_rays = query_rays.shape[0]

            hist_q = hists[i].reshape(HW, -1).repeat(num_rays, 1)
            hist_k = torch.repeat_interleave(hists[j].reshape(HW, -1), num_rays, 0)

            Qmtx = query_rays.repeat(num_rays, 1)
            Kmtx = torch.repeat_interleave(key_rays, num_rays, 0)

            normals = torch.linalg.cross(Qmtx, Kmtx)
            norm_factor = 1.0/torch.norm(normals, 2, dim=1)

            pos_1 = torch.linalg.cross(Kmtx, normals)
            pos_2 = torch.linalg.cross(Qmtx, normals)

            intersection_dist_1 = torch.mm(pos_1, dist.unsqueeze(1)).squeeze()*norm_factor
            intersection_dist_2 = torch.mm(pos_2, dist.unsqueeze(1)).squeeze()*norm_factor

            intersection_1 = pos[j] + intersection_dist_1.unsqueeze(1)*Kmtx
            intersection_2 = pos[i] + intersection_dist_2.unsqueeze(1)*Qmtx

            transformed_1 = torch.mm(poses[i, 0:3, 0:3], intersection_1.T).T + pos[i]
            transformed_2 = torch.mm(poses[j, 0:3, 0:3], intersection_2.T).T + pos[j]

            transformed_1 = (transformed_1/(transformed_1[..., 2:3]+1e-7)).reshape(num_rays, num_rays, -1)
            transformed_2 = (transformed_2/(transformed_2[..., 2:3]+1e-7)).reshape(num_rays, num_rays, -1)

            diffs_1 = ((transformed_1[..., :2] - Qmtx.reshape(num_rays, num_rays, -1)[..., :2])**2).sum(dim=-1)
            diffs_2 = ((transformed_2[..., :2] - Kmtx.reshape(num_rays, num_rays, -1)[..., :2])**2).sum(dim=-1)

            w_1 = torch.exp(-alpha * diffs_1)
            w_1[w_1 < 0.1] = 0
            w_hist_1 = hist_k[(indices, intersection_dist_2.round().long())]
            # Normalize
            w_1 = w_1 / (w_1.sum(dim=0)+1e-7)
            weights[i*HW:(i+1)*HW,j*HW:(j+1)*HW] = w_1 * w_hist_1.reshape(HW, HW)

            w_2 = torch.exp(-alpha * diffs_2)
            w_2[w_2 < 0.1] = 0
            w_hist_2 = hist_q[(indices, intersection_dist_1.round().long())]
            # Normalize
            w_2 = w_2 / (w_2.sum(dim=0)+1e-7)
            weights[j*HW:(j+1)*HW,i*HW:(i+1)*HW] = w_2 * w_hist_2.reshape(HW, HW)

            #projection = torch.mm(A, transformed.T).T
            #projection = (projection[:, 0:2]/projection[:, 2:3]).reshape(num_rays, num_rays, -1)


    return weights

def getAllRayIntersections(rays, poses):
    distances = []
    depths = []
    nviews = rays[0].shape[0]
    for ray in rays:
        distForLevel = []
        depthForLevel = []
        for i in range(nviews-1):
            # Query view is the one "being reconstructed"
            ray1 = ray[i]
            query_pos = poses[i][:3,3]
            for j in range(i+1, nviews):
                # Key view is the one providing other features
                ray2 = ray[j]
                key_pos = poses[j][:3,3]
                # Compute normals to both rays
                normals = getAllCrossProducts(ray1, ray2)
                # Normalization factor
                factors = 1.0/torch.linalg.norm(normals, dim=-1)
                # Translation between cameras
                trans = (query_pos - key_pos).unsqueeze(1)
                # Distances between rays
                dist = torch.abs(torch.matmul(normals,trans))[..., 0] * factors

                # Distance of intersection along query rays
                d_q = torch.matmul(getAllCrossProductsMtx(ray1, normals), -trans)[..., 0] * factors
                # Distance of intersection along key rays
                d_k = torch.matmul(getAllCrossProductsMtx(ray2, normals.permute(1, 0, 2)), -trans)[..., 0] * factors

                distForLevel.append(dist)
                depthForLevel.append(torch.stack([d_q, d_k], -1))

        distances.append(torch.stack(distForLevel))
        depths.append(torch.stack(depthForLevel))
    return distances, depths

def getAllCrossProducts(vec1, vec2):
    numrays = vec1.shape[0]
    vec1mtx = vec1.repeat(numrays, 1)
    vec2mtx = torch.repeat_interleave(vec2, numrays, 0)
    return torch.linalg.cross(vec1mtx, vec2mtx).view(numrays, numrays, 3)

def getAllCrossProductsMtx(vec1, mtx):
    numrays = vec1.shape[0]
    vecmtx = torch.repeat_interleave(vec1, numrays, 0)
    vec2 = mtx.reshape(-1, 3)
    return torch.linalg.cross(vecmtx, vec2).view(numrays, numrays, 3)

def getElements(distances, depths, query, keys):
    dist_arr = []
    q_depth = []
    k_depth = []
    nviews = depths.shape[0]
    for key in keys:
        dist = distances[fromMatrixToVector(query, key, nviews-1)]
        depth = depths[fromMatrixToVector(query, key, nviews-1)]
        dist_arr.append(dist)
        q_depth.append(depth[..., 0 if query > key else 1])
        k_depth.append(depth[..., 1 if query > key else 0])

    return dist_arr, q_depth, k_depth

def fromMatrixToVector(i, j, N):
   if i <= j:
      return i * N - (i - 1) * i // 2 + j - i - 1
   else:
      return j * N - (j - 1) * j // 2 + i - j - 1

def sumFeatures(features, distances, query, keys, device = 'cuda:0'):
    # Initialize features with features of the query view
    newFeat = torch.zeros(features[query].shape).to(device)
    newFeat += features[query]
    fCount = newFeat.shape[0]
    alpha = 100.0
    for i, key in enumerate(keys):
        # Weights are RBF from ray distance
        weights = torch.exp(-alpha * distances[i])
        # Remove small weights
        weights[weights < 0.1] = 0
        # Normalize
        weights = weights / weights.sum(dim=0)
        # Add key features
        currFeat = features[key].view(fCount, -1)
        newFeat += torch.matmul(weights, currFeat.T).T.view(newFeat.shape)
    return newFeat

def sumFeaturesDepthHist(features, distances, query, keys, hist, depths_key, device = 'cuda:0'):
    # Initialize features with features of the query view
    newFeat = torch.zeros(features[query].shape).to(device)
    newFeat += features[query]
    fCount = newFeat.shape[0]
    alpha = 50.0
    for i, key in enumerate(keys):
        # Get key depth histogram and depths
        d_hist = hist[key].view(-1, hist.shape[-1])
        ray_depth = depths_key[i].long()
        # Get weights from histogram
        hist_w = d_hist[(torch.arange(d_hist.shape[0]).cuda(), ray_depth)]
        # Weights are RBF from ray distance
        weights = torch.exp(-alpha * distances[i])
        weights[weights < 0.1] = 0
        # Remove small weights
        weights *= hist_w
        # Normalize
        weights = weights / (weights.sum(dim=0)+1e-7)
        # Add key features
        currFeat = features[key].view(fCount, -1)
        newFeat += torch.matmul(weights, currFeat.T).T.view(newFeat.shape)
    return newFeat

def sumFeaturesDepthAttention(features, distances, hist, depths_query, depths_key):
    newFeat = []
    return newFeat