"here, we use the LIIF method, modified from the LIIF algorithm: https://github.com/yinboc/liif"
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from models import register
from utils import make_coord


@register('liif')
class LIIF(nn.Module):

    def __init__(self, encoder_spec_BF, encoder_spec_DF, encoder_DPC, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True, coordencode=False, radialencode=False, encoord_dim=48, angle_num=12):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.coordencode = coordencode
        self.encoord_dim = encoord_dim
        self.radialencode = radialencode
        self.angle_num = angle_num

        self.encoder_BF = models.make(encoder_spec_BF)   # here, we make the model for extract the latent code from bright field illumination
        self.encoder_DF = models.make(encoder_spec_DF)   # here, we make the model for extract the latent code from dark field illumination
        self.encoder_DPC = models.make(encoder_DPC)      # here, we add another channel for the DPC init reconstruction
        self.DF_dim = self.encoder_DF.out_dim
        self.DPC_dim = self.encoder_DPC.out_dim

        if imnet_spec is not None:
            imnet_in_dim = self.encoder_BF.out_dim + self.encoder_DF.out_dim + self.encoder_DPC.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
                self.DF_dim *= 9
            imnet_in_dim += 2     # attach coordinate
            if self.cell_decode:
                imnet_in_dim += 2
            if self.coordencode:  # add spatial encoding, need to modify code more, now it has some problem, don't use it
                if radialencode:
                    imnet_in_dim += 2 * self.encoord_dim * (2 * self.angle_num)
                    coord_dim = 2 + 2 * self.encoord_dim * (2 * self.angle_num)
                else:
                    imnet_in_dim += self.encoord_dim * 4   # 4 is for the x, y and cos(), sin()
                    coord_dim = 2 + self.encoord_dim * 4   # record dimension of coordinates and encoded coordinates
                self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim, 'coord_dim': coord_dim})
                # self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim, 'coord_dim': coord_dim, 'dark_dim': self.DF_dim})
            else:
                self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})

        else:
            self.imnet = None

    def gen_feat(self, inp):   # generate latent code for the training
        self.feat_BF = self.encoder_BF(inp[:, 0:2, :])   # bright field code
        self.feat_DF = self.encoder_DF(inp[:, 2:5, :])   # dark field code
        self.feat_DPC = self.encoder_DPC(inp[:, 5:6, :])   # DPC code
        self.feat = torch.cat((self.feat_BF, self.feat_DF, self.feat_DPC), axis=1)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat   # generated latent code

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])   # feature unfolding the neighbor 3*3 latent code

        if self.local_ensemble:   # used for the local ensemble
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1]), divide by 2 for radius
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # set coordinates for latent codes
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []     # collect predictions for queried point based on different latent code
        areas = []     # collect areas as weights
        for vx in vx_lst:   # here, the [-1, 1] is used for local ensemble
            for vy in vy_lst:
                coord_ = coord.clone()        # this is coordinates of the queried points
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(       # pick the latent code related to the high-resolution images from the low-res latent space
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(      # pick the corresponding coordinates
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord   # calculate the relative coordinates
                rel_coord[:, :, 0] *= feat.shape[-2]   # times a scale factor
                rel_coord[:, :, 1] *= feat.shape[-1]

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([q_feat, rel_cell], dim=-1)    # here, it's the input feature + cell size
                else:
                    inp = [q_feat]

                inp = torch.cat([inp, rel_coord], dim=-1)    # here, it's the input feature + cell size + coordinates to MLP

                if self.coordencode:   # there is some problem for encoding, don't use by now
                    if self.radialencode:
                        coord_encode = utils.radial_encode(rel_coord, self.encoord_dim, self.angle_num).cuda()
                    else:
                        coord_encode = utils.encode_coordnate(rel_coord, self.encoord_dim).cuda()   # here, add the encoded coordinates as the last dimenstion
                    inp = torch.cat([inp, coord_encode], dim=-1)
                    self.coord_dim = rel_coord.shape[-1] + coord_encode.shape[-1]   # keep the dimensions of coordinates

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:   # diagonal area since larger area -> far from datapoint
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)   # output with weighted sum
        return ret

    def forward(self, inp, coord, cell):   # output of prediction
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)