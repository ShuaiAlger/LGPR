import numpy as np

import cv2


def get_rotmat(angle, as_3d=False, scale=1.0, center_x=0.0, center_y=0.0):
    cos_angle, sine_angle = np.cos(angle) * scale, np.sin(angle) * scale
    rotation = [cos_angle, -sine_angle, sine_angle, cos_angle]
    rotation = np.reshape(rotation, (2, 2)).T
    if as_3d:
        matrix_3d = np.eye(3)
        matrix_3d[:2, :2] = rotation
        matrix_3d[0, 2] = ((1 - cos_angle)*center_x) - (sine_angle*center_y)
        matrix_3d[1, 2] = (sine_angle*center_x) + ((1-cos_angle)*center_y)
        return matrix_3d
    return rotation

def get_translation_mat(image_height, image_width, trans, transformed_corners):
    left_top_min = np.min(transformed_corners, axis=0)
    right_bottom_min = np.min(np.array([image_width, image_height]) - transformed_corners, axis=0)
    trans_x_value = int(np.random.uniform(0, trans) * image_width)
    trans_y_value = int(np.random.uniform(0, trans) * image_height)
    if np.random.uniform() > 0.5: #translate x with respect to left axis
        trans_x = trans_x_value if left_top_min[0] < 0 else -trans_x_value
    else: #translate x with respect to right axis
        trans_x = trans_x_value if right_bottom_min[0] > 0 else -trans_x_value
    if np.random.uniform() > 0.5: #translate y with respect to top axis
        trans_y = trans_y_value if left_top_min[1] < 0 else -trans_y_value
    else: #translate y with respect to bottom axis
        trans_y = trans_y_value if right_bottom_min[1] > 0 else -trans_y_value
    translate_mat = np.eye(3)
    translate_mat[0, 2] = trans_x
    translate_mat[1, 2] = trans_y
    return translate_mat

def get_perspective_mat(patch_ratio, center_x, center_y, pers_x, pers_y, shear_ratio, shear_angle, rotation_angle, scale, trans):
    shear_angle, rotation_angle = np.deg2rad(shear_angle), np.deg2rad(rotation_angle)
    image_height, image_width = center_y * 2, center_x * 2
    patch_bound_w, patch_bound_h = int(patch_ratio * image_width), int(patch_ratio * image_height)
    patch_corners = np.array([[0,0], [0, patch_bound_h], [patch_bound_w, patch_bound_h], [patch_bound_w, 0]]).astype(np.float32)
    pers_value_x = np.random.normal(0, pers_x/2)
    pers_value_y = np.random.normal(0, pers_y/2)
    pers_matrix = np.array([[1, 0, 0], [0, 1, 0], [pers_value_x, pers_value_y, 1]])
    #shear_ratio is given by shear_x/shear_y
    if np.random.uniform() > 0.5:
        shear_ratio_value = np.random.uniform(1, 1+shear_ratio)
        shear_x, shear_y = 1, 1 / shear_ratio_value
    else:
        shear_ratio_value = np.random.uniform(1-shear_ratio, 1)
        shear_x, shear_y = shear_ratio_value, 1
    shear_angle_value = np.random.uniform(-shear_angle, shear_angle)
    shear_matrix = get_rotmat(-shear_angle_value, as_3d=True, center_x=center_x, center_y=center_y) @ np.diag([shear_x, shear_y, 1]) @ get_rotmat(shear_angle_value, as_3d=True, center_x=center_x, center_y=center_y)
    shear_perspective = shear_matrix @ pers_matrix
    rotation_angle_value = np.random.uniform(-rotation_angle, rotation_angle)
    scale_value = np.random.uniform(1, 1+(2*scale))
    #priotrising scaling up compared to scaling down
    scaled_rotation_matrix = get_rotmat(rotation_angle_value, as_3d=True, scale=scale_value, center_x=center_x, center_y=center_y)
    homography_matrix = scaled_rotation_matrix @ shear_perspective
    trans_patch_corners = cv2.perspectiveTransform(np.reshape(patch_corners, (-1, 1, 2)), homography_matrix).squeeze(1)
    translation_matrix = get_translation_mat(image_height, image_width, trans, trans_patch_corners)
    homography_matrix = translation_matrix @ homography_matrix
    return homography_matrix


import torch
import numpy as np
import pdb

debug_cnt = -1

def make_batch(augmentor, difficulty = 0.3, train = True):
    Hs = []
    img_list = augmentor.train if train else augmentor.test
    dev = augmentor.device
    batch_images = []

    with torch.no_grad(): # we dont require grads in the augmentation
        for b in range(augmentor.batch_size):
            rdidx = np.random.randint(len(img_list))
            img = torch.tensor(img_list[rdidx], dtype=torch.float32).permute(2,0,1).to(augmentor.device).unsqueeze(0)
            batch_images.append(img)

        batch_images = torch.cat(batch_images)

        p1, H1 = augmentor(batch_images, difficulty)
        p2, H2 = augmentor(batch_images, difficulty, TPS = True, prob_deformation = 0.7)

    return p1, p2, H1, H2


def plot_corrs(p1, p2, src_pts, tgt_pts):
    import matplotlib.pyplot as plt
    p1 = p1.cpu()
    p2 = p2.cpu()
    src_pts = src_pts.cpu() ; tgt_pts = tgt_pts.cpu()
    rnd_idx = np.random.randint(len(src_pts), size=200)
    src_pts = src_pts[rnd_idx, ...]
    tgt_pts = tgt_pts[rnd_idx, ...]

    #Plot ground-truth correspondences
    fig, ax = plt.subplots(1,2,figsize=(18, 12))
    colors = np.random.uniform(size=(len(tgt_pts),3))
    #Src image
    img = p1
    for i, p in enumerate(src_pts):
        ax[0].scatter(p[0],p[1],color=colors[i])
    ax[0].imshow(img.permute(1,2,0).numpy()[...,::-1])

    #Target img
    img2 = p2
    for i, p in enumerate(tgt_pts):
        ax[1].scatter(p[0],p[1],color=colors[i])
    ax[1].imshow(img2.permute(1,2,0).numpy()[...,::-1])
    plt.show()


def get_corresponding_pts(p1, p2, H, H2, augmentor, h, w, crop = None):
    '''
        Get dense corresponding points
    '''
    global debug_cnt
    negatives, positives = [], []

    with torch.no_grad():
        #real input res of samples
        rh, rw = p1.shape[-2:]
        ratio = torch.tensor([rw/w, rh/h], device = p1.device)

        (H, mask1) = H
        (H2, src, W, A, mask2) = H2

        #Generate meshgrid of target pts
        x, y = torch.meshgrid(torch.arange(w, device=p1.device), torch.arange(h, device=p1.device), indexing ='xy')
        mesh = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        target_pts = mesh.view(-1, 2) * ratio

        #Pack all transformations into T
        for batch_idx in range(len(p1)):
            with torch.no_grad():
                T = (H[batch_idx], H2[batch_idx], 
                    src[batch_idx].unsqueeze(0), W[batch_idx].unsqueeze(0), A[batch_idx].unsqueeze(0))
                #We now warp the target points to src image
                src_pts = (augmentor.get_correspondences(target_pts, T) ) #target to src 
                tgt_pts = (target_pts)
            
                #Check out of bounds points
                mask_valid = (src_pts[:, 0] >=0) & (src_pts[:, 1] >=0) & \
                            (src_pts[:, 0] < rw) & (src_pts[:, 1] < rh)

                negatives.append( tgt_pts[~mask_valid] )            
                tgt_pts = tgt_pts[mask_valid]
                src_pts = src_pts[mask_valid]


                #Remove invalid pixels
                mask_valid =    mask1[batch_idx, src_pts[:,1].long(), src_pts[:,0].long()]  & \
                                mask2[batch_idx, tgt_pts[:,1].long(), tgt_pts[:,0].long()]
                tgt_pts = tgt_pts[mask_valid]
                src_pts = src_pts[mask_valid]

                # limit nb of matches if desired
                if crop is not None:
                    rnd_idx = torch.randperm(len(src_pts), device=src_pts.device)[:crop]
                    src_pts = src_pts[rnd_idx]
                    tgt_pts = tgt_pts[rnd_idx]

                if debug_cnt >=0 and debug_cnt < 4:
                    plot_corrs(p1[batch_idx], p2[batch_idx], src_pts , tgt_pts )
                    debug_cnt +=1

                src_pts = (src_pts / ratio)
                tgt_pts = (tgt_pts / ratio)

                #Check out of bounds points
                padto = 10 if crop is not None else 2
                mask_valid1 = (src_pts[:, 0] >= (0 + padto)) & (src_pts[:, 1] >= (0 + padto)) & \
                             (src_pts[:, 0] < (w - padto)) & (src_pts[:, 1] < (h - padto))
                mask_valid2 = (tgt_pts[:, 0] >= (0 + padto)) & (tgt_pts[:, 1] >= (0 + padto)) & \
                             (tgt_pts[:, 0] < (w - padto)) & (tgt_pts[:, 1] < (h - padto))
                mask_valid = mask_valid1 & mask_valid2
                tgt_pts = tgt_pts[mask_valid]
                src_pts = src_pts[mask_valid]         

                #Remove repeated correspondences
                lut_mat = torch.ones((h, w, 4), device = src_pts.device, dtype = src_pts.dtype) * -1
                # src_pts_np = src_pts.cpu().numpy()
                # tgt_pts_np = tgt_pts.cpu().numpy()
                try:
                    lut_mat[src_pts[:,1].long(), src_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)
                    mask_valid = torch.all(lut_mat >= 0, dim=-1)
                    points = lut_mat[mask_valid]
                    positives.append(points)
                except:
                    pdb.set_trace()
                    print('..')

    return negatives, positives


def crop_patches(tensor, coords, size = 7):
    '''
        Crop [size x size] patches around 2D coordinates from a tensor.
    '''
    B, C, H, W = tensor.shape

    x, y = coords[:, 0], coords[:, 1]
    y = y.view(-1, 1, 1)
    x = x.view(-1, 1, 1)
    halfsize = size // 2
    # Create meshgrid for indexing
    x_offset, y_offset = torch.meshgrid(torch.arange(-halfsize, halfsize+1), torch.arange(-halfsize, halfsize+1), indexing='xy')
    y_offset = y_offset.to(tensor.device)
    x_offset = x_offset.to(tensor.device)

    # Compute indices around each coordinate
    y_indices = (y + y_offset.view(1, size, size)).squeeze(0) + halfsize
    x_indices = (x + x_offset.view(1, size, size)).squeeze(0) + halfsize

    # Handle out-of-boundary indices with padding
    tensor_padded = torch.nn.functional.pad(tensor, (halfsize, halfsize, halfsize, halfsize), mode='constant')

    # Index tensor to get patches
    patches = tensor_padded[:, :, y_indices, x_indices] # [B, C, N, H, W]
    return patches

def subpix_softmax2d(heatmaps, temp = 0.25):
    N, H, W = heatmaps.shape
    heatmaps = torch.softmax(temp * heatmaps.view(-1, H*W), -1).view(-1, H, W)
    x, y = torch.meshgrid(torch.arange(W, device =  heatmaps.device ), torch.arange(H, device =  heatmaps.device ), indexing = 'xy')
    x = x - (W//2)
    y = y - (H//2)
    #pdb.set_trace()
    coords_x = (x[None, ...] * heatmaps)
    coords_y = (y[None, ...] * heatmaps)
    coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H*W, 2)
    coords = coords.sum(1)

    return coords


def check_accuracy(X, Y, pts1 = None, pts2 = None, plot=False):
    with torch.no_grad():
        #dist_mat = torch.cdist(X,Y)
        dist_mat = X @ Y.t()
        nn = torch.argmax(dist_mat, dim=1)
        #nn = torch.argmin(dist_mat, dim=1)
        correct = nn == torch.arange(len(X), device = X.device)

        if pts1 is not None and plot:
            import matplotlib.pyplot as plt
            canvas = torch.zeros((60, 80),device=X.device)
            pts1 = pts1[~correct]
            canvas[pts1[:,1].long(), pts1[:,0].long()] = 1
            canvas = canvas.cpu().numpy()
            plt.imshow(canvas), plt.show()

        acc = correct.sum().item() / len(X)
        return acc

def get_nb_trainable_params(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	nb_params = sum([np.prod(p.size()) for p in model_parameters])
 
	print('Number of trainable parameters: {:d}'.format(nb_params))


import torch.nn.functional as F


def dual_softmax_loss(X, Y, temp = 0.2):
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = (X @ Y.t()) * temp
    conf_matrix12 = F.log_softmax(dist_mat, dim=1)
    conf_matrix21 = F.log_softmax(dist_mat.t(), dim=1)

    with torch.no_grad():
        conf12 = torch.exp( conf_matrix12 ).max(dim=-1)[0]
        conf21 = torch.exp( conf_matrix21 ).max(dim=-1)[0]
        conf = conf12 * conf21

    target = torch.arange(len(X), device = X.device)

    loss = F.nll_loss(conf_matrix12, target) + \
           F.nll_loss(conf_matrix21, target)

    return loss, conf

def smooth_l1_loss(input, target, beta=2.0, size_average=True):
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean() if size_average else loss.sum()

def fine_loss(f1, f2, pts1, pts2, fine_module, ws=7):
    '''
        Compute Fine features and spatial loss
    '''
    C, H, W = f1.shape
    N = len(pts1)

    #Sort random offsets
    with torch.no_grad():
        a = -(ws//2)
        b = (ws//2)
        offset_gt = (a - b) * torch.rand(N, 2, device = f1.device) + b
        pts2_random = pts2 + offset_gt

    #pdb.set_trace()
    patches1 = crop_patches(f1.unsqueeze(0), (pts1+0.5).long(), size=ws).view(C, N, ws * ws).permute(1, 2, 0) #[N, ws*ws, C]
    patches2 = crop_patches(f2.unsqueeze(0), (pts2_random+0.5).long(), size=ws).view(C, N, ws * ws).permute(1, 2, 0)  #[N, ws*ws, C]

    #Apply transformer
    patches1, patches2 = fine_module(patches1, patches2)

    features = patches1.view(N, ws, ws, C)[:, ws//2, ws//2, :].view(N, 1, 1, C) # [N, 1, 1, C]
    patches2 = patches2.view(N, ws, ws, C) # [N, w, w, C]

    #Dot Product
    heatmap_match = (features * patches2).sum(-1)
    offset_coords = subpix_softmax2d(heatmap_match)

    #Invert offset because center crop inverts it
    offset_gt = -offset_gt 

    #MSE
    error = ((offset_coords - offset_gt)**2).sum(-1).mean()

    #error = smooth_l1_loss(offset_coords, offset_gt)

    return error





def keypoint_position_loss(kpts1, kpts2, pts1, pts2, softmax_temp = 1.0):
    '''
        Computes coordinate classification loss, by re-interpreting the 64 bins to 8x8 grid and optimizing
        for correct offsets
    '''
    C, H, W = kpts1.shape
    kpts1 = kpts1.permute(1,2,0) * softmax_temp
    kpts2 = kpts2.permute(1,2,0) * softmax_temp

    with torch.no_grad():
        #Generate meshgrid
        x, y = torch.meshgrid(torch.arange(W, device=kpts1.device), torch.arange(H, device=kpts1.device), indexing ='xy')
        xy = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        xy*=8

        #Generate collision map
        hashmap = torch.ones((H*8, W*8, 2), dtype = torch.long, device = kpts1.device) * -1
        hashmap[(pts1[:,1]).long(), (pts1[:,0]).long(), :] = (pts2).long()

        #Estimate offset of src kpts 
        _, kpts1_offsets = kpts1.max(dim=-1)
        kpts1_offsets_x = kpts1_offsets  % 8
        kpts1_offsets_y = kpts1_offsets // 8
        kpts1_offsets_xy = torch.cat([kpts1_offsets_x.unsqueeze(-1), 
                                      kpts1_offsets_y.unsqueeze(-1)], dim=-1)
        #pdb.set_trace()
        kpts1_coords = xy + kpts1_offsets_xy

        #find src -> tgt pts
        kpts1_coords = kpts1_coords.view(-1,2)
        gt_12 = hashmap[kpts1_coords[:,1], kpts1_coords[:,0]]
        mask_valid = torch.all(gt_12 >= 0, dim=-1)
        gt_12 = gt_12[mask_valid]

        #find offset labels
        labels2 = (gt_12/8) - (gt_12/8).long()
        labels2 = (labels2 * 8).long()
        labels2 = labels2[:, 0] + 8*labels2[:, 1] #linear index
        
    kpts2_selected = kpts2[(gt_12[:, 1]/8).long(), (gt_12[:, 0]/8).long()]        

    kpts1_selected = F.log_softmax(kpts1.view(-1,C)[mask_valid], dim=-1)
    kpts2_selected = F.log_softmax(kpts2_selected, dim=-1)

    #Here we enforce softmax to keep current max on src kps
    with torch.no_grad():
        _, labels1 =  kpts1_selected.max(dim=-1)

    predicted2 = kpts2_selected.max(dim=-1)[1]
    acc =  (labels2 == predicted2)
    acc = acc.sum() / len(acc)

    loss = F.nll_loss(kpts1_selected, labels1, reduction = 'mean') + \
           F.nll_loss(kpts2_selected, labels2, reduction = 'mean')
    
    #pdb.set_trace()

    return loss, acc

def coordinate_classification_loss(coords1, pts1, pts2, conf):
    '''
        Computes the fine coordinate classification loss, by re-interpreting the 64 bins to 8x8 grid and optimizing
        for correct offsets after warp
    '''
    #Do not backprop coordinate warps
    with torch.no_grad():

        coords1_detached = pts1 * 8 

        #find offset
        offsets1_detached = (coords1_detached/8) - (coords1_detached/8).long()
        offsets1_detached = (offsets1_detached * 8).long()
        labels1 = offsets1_detached[:, 0] + 8*offsets1_detached[:, 1]

    #pdb.set_trace()
    coords1_log = F.log_softmax(coords1, dim=-1)

    predicted = coords1.max(dim=-1)[1]
    acc =  (labels1 == predicted)
    acc = acc[conf > 0.1]
    acc = acc.sum() / len(acc)

    loss = F.nll_loss(coords1_log, labels1, reduction = 'none')
    
    #Weight loss by confidence, giving more emphasis on reliable matches
    conf = conf / conf.sum()
    loss = (loss * conf).sum()

    return loss * 2., acc

def keypoint_loss(heatmap, target):
    # Compute L1 loss
    L1_loss = F.l1_loss(heatmap, target)
    return L1_loss * 3.0

def hard_triplet_loss(X,Y, margin = 0.5):

    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = torch.cdist(X, Y, p=2.0)
    dist_pos = torch.diag(dist_mat)
    dist_neg = dist_mat + 100.*torch.eye(*dist_mat.size(), dtype = dist_mat.dtype, 
            device = dist_mat.get_device() if dist_mat.is_cuda else torch.device("cpu"))

    #filter repeated patches on negative distances to avoid weird stuff on gradients
    dist_neg = dist_neg + dist_neg.le(0.01).float()*100.

    #Margin Ranking Loss
    hard_neg = torch.min(dist_neg, 1)[0]

    loss = torch.clamp(margin + dist_pos - hard_neg, min=0.)

    return loss.mean()






