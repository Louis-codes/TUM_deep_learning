from torch import nn
import torch
from ..network import SCORE_SAVER

class ScaledDotAttention(nn.Module):

    def __init__(self,
                 d_k):
        """

        Args:
            d_k: Dimension of Keys and Queries
            dropout: Dropout probability
        """
        super().__init__()
        self.d_k = d_k

        self.softmax = None

        ########################################################################
        # TODO:                                                                #
        #   Task 2: Initialize the softmax layer (torch.nn implementation)     #
        #                                                                      #
        ########################################################################

        self.softmax = nn.Softmax(dim=-1)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor) -> torch.Tensor:
        """
        Computes the scaled dot attention given query, key and value inputs. Stores the scores in SCORE_SAVER for
        visualization

        Args:
            q: Query Inputs
            k: Key Inputs
            v: Value Inputs

        Shape:
            - q: (*, sequence_length_queries, d_model)
            - k: (*, sequence_length_keys, d_model)
            - v: (*, sequence_length_keys, d_model)
            - outputs: (*, sequence_length_queries, d_v)
        """
        scores = None
        outputs = None

        ########################################################################
        # TODO:                                                                #
        #   Task 2:                                                            #
        #       - Calculate the scores using the queries and keys              #
        #       - Normalise the scores using the softmax function              #
        #       - Compute the updated embeddings and return the output         #
        #                                                                      #
        # Hint 2:                                                              #
        #       - torch.transpose(x, dim_1, dim_2) swaps the dimensions dim_1  #
        #         and dim_2 of the tensor x!                                   #
        #       - Later we will insert more dimensions into *, so how could    #
        #         index these dimensions to always get the right ones?         #
        #       - Also dont forget to scale the scores as discussed!           #
        ########################################################################

        scores = torch.matmul(q, torch.transpose(k, -2, -1)) / (self.d_k ** 0.5)
        scores = self.softmax(scores)
        outputs = torch.matmul(scores, v)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        SCORE_SAVER.save(scores)

        return outputs
