import torch
import torch.nn as nn
import torch.nn.functional as F


from basicsr.utils.registry import LOSS_REGISTRY

# ---------------------------------
# 2024-04-23
# anchor：当前图像
# negative：同batch的其他图像中，距离最远的那个
# postive：ground truth
# ---------------------------------

@LOSS_REGISTRY.register()
class BatchTripletLoss(nn.Module):
    """
    Triplet loss function that calculates loss over a batch of embeddings.
    Assumes that the positive sample is given and fixed for all anchors in the batch.
    """
    def __init__(self, margin=0.5):
        super(BatchTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, batch, positive):
        """
        Compute the triplet loss for each anchor in the batch with the given positive
        and the farthest negative sample in the batch.
        
        :param batch: Tensor, the batch of embeddings.
        :param positive: Tensor, the positive embedding.
        :return: Tensor, the mean triplet loss for the batch.
        """
        losses = []
        for i, anchor in enumerate(batch):

            # Skip if the current sample is the positive
            if torch.equal(anchor, positive):
                continue

            # Compute distances from the anchor to all negatives in the batch
            # Negatives are all the samples in the batch except the anchor and the given positive
            negatives = torch.cat([batch[:i], batch[i+1:]])
            negative_distances = F.pairwise_distance(anchor.unsqueeze(0), negatives, 2)
            
            # Find the farthest negative in the batch for the anchor
            max_negative_distance, _ = negative_distances.max(dim=0)
            
            # Compute distance between the anchor and the given positive
            positive_distance = F.pairwise_distance(anchor.unsqueeze(0), positive.unsqueeze(0), 2)

            # Compute triplet loss for the current anchor
            loss = F.relu(positive_distance - max_negative_distance + self.margin)
            losses.append(loss)

        # Calculate the mean loss over all the valid anchors
        losses = torch.cat(losses)
        return losses.mean()

# # Example usage:
# batch_size = 5
# feature_dim = 128
# margin = 1.0

# # Create a random batch of embeddings with shape (batch_size, feature_dim)
# batch = torch.randn(batch_size, feature_dim)

# # Select a sample to be the fixed positive embedding for all anchors
# # In a real scenario, this would be given/known in advance.
# fixed_positive = batch[1].unsqueeze(0) # Just as an example, the second sample is the fixed positive

# # Compute the batch triplet loss
# batch_triplet_loss = BatchTripletLoss(margin)
# loss = batch_triplet_loss(batch, fixed_positive)
# print(loss)
