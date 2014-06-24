'''
Created on Nov 9, 2012

@author: jiayq
'''

from iceberk import pipeline
import numpy as np

class SpatialPinkTrainer(pipeline.ZcaTrainer):
    """My strange definition - instead of whiten the data, make the data pink.
    The "pinkness" is defined assuming that the incoming patches can be reshaped
    as a width*height shape, and the standard deviation of the transformed 
    feature at each spatial location will be the correlation it has with the
    center patch. if the patch is multi-channeled, the correlation is averaged
    over the multiple channels. if the width or height is an even number, the
    correlation is also averaged over the 2 (or 4, if both are even) locations,
    and again, over the channels.
    """
    def __init__(self, specs):
        pipeline.ZcaTrainer.__init__(self, specs)
        if 'size' not in specs:
            raise KeyError, "The trainer should be provided with the size of the patches."
    
    def train(self, incoming_patches):
        (W, b), (_, _, covmat) = pipeline.ZcaTrainer.train(
                self, incoming_patches)
        # figure out the correlation
        height, width = self.specs['size']
        num_channels = int(covmat.shape[0] / height / width)
        if height * width * num_channels != covmat.shape[0]:
            raise ValueError, "The incoming patches does not fit my specs."
        # compute the correlation
        std = np.sqrt(np.diag(covmat) + np.finfo(np.float64).eps)
        # convert the covmat to correlation in-place
        covmat /= std
        covmat /= std[:, np.newaxis]
        corr = covmat.reshape(height, width, num_channels, height, width, num_channels)
        # get the center
        corr_center = corr[int(np.ceil(height/2.-1)):int(np.ceil(height/2.+0.5)),
                           int(np.ceil(width/2.-1)):int(np.ceil(width/2.+0.5)),]
        # over the last channels
        num_centers = np.prod(corr_center.shape[:2])
        weights = corr_center.reshape((num_centers,) + corr_center.shape[2:])\
                .mean(axis=0)
        # now, if we have multiple channels, we compute the per-channel average
        # but no cross-channel ones
        weights = np.sum(weights[i,:,:,i] for i in range(num_channels)).flatten()
        weights /= weights.max()
        np.abs(weights, out=weights)
        np.sqrt(weights, out=weights)
        # now, duplicate the weights so it has multiple channels
        weights = np.tile(weights[:,np.newaxis], (1,num_channels)).flatten()
        W *= weights
        # currently I don't return any metadata...
        return (W, b), (weights)