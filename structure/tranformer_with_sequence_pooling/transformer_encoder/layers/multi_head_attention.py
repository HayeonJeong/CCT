import torch
import math
import torch.nn.functional as F
from torch import nn
#from easydict import EasyDict as edict

'''
torch.transpose(input, dim0, dim1)

[Example]
    # 2D 텐서 생성
    x = torch.tensor([[1, 2, 3],
                    [4, 5, 6]])

    # 첫 번째 차원과 두 번째 차원의 순서를 바꿈
    transposed_x = torch.transpose(x, 0, 1)
    print(transposed_x)
    
    # tensor([[1, 4],
        [2, 5],
        [3, 6]])
'''

def scaled_dot_product_attention(query, key, value, mask=None):
    # query, key, value: (n_batch, seq_len, d_k)
    # mask: (n_batch, seq_len, seq_len)

    d_k = query.size(-1) # d_k = d_model / num_heads
    attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, (n_batch, seq_len, seq_len)
    attention_score = attention_score / math.sqrt(d_k)

    # Masking (optional)
    if mask is not None:
        attention_score = attention_score.masked_fill(mask==0, float('-inf'))
        # mask tensor는 0 or 1로 이루어져 있으며, 0인 위치는 어텐션 스코어에 음의 무한대 값을 채워주어 해당 위치의 가중치를 0으로 만듦
        # softmax 함수 적용 시 exp^(-inf) = 0, 따라서 attention weights가 모두 0이 됨
        # 이렇게 함으로써 마스크가 적용된 위치의 정보를 attention weights에 반영하지 않도록 할 수 있음
    
    weights = F.softmax(attention_score, dim=-1) # (n_batch, seq_len, seq_len)
    
    # bmm : 배치 차원을 고려한 행렬-행렬 곱셈
    # Q. matmul vs. mm vs. bmm (https://velog.io/@regista/torch.dot-torch.matmul-torch.mm-torch.bmm)
    result = torch.bmm(weights, value) # (n_batch, seq_len, d_k)

    return result

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        '''
        nn.Linear(input_feature_dimension, output_feature_dimension)
        # Linear Transformation (선형 변환: 차원 변환)
        '''
        super(AttentionHead, self).__init__()
        self.Q = nn.Linear(embed_dim, head_dim)
        self.K = nn.Linear(embed_dim, head_dim)
        self.V = nn.Linear(embed_dim, head_dim)
        
    # Mask
    def forward(self, query, key, value, mask=None):
        # query = torch.transpose(query, 0, 1)
        # value = torch.transpose(value, 0, 1)
        # key = torch.transpose(key, 0, 1)
        
        # print("query size : ", query.size())
        
        attention_output = scaled_dot_product_attention(self.Q(query), self.K(key), self.V(value), mask=mask)
        return attention_output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = cfg.encoder.d_model # CCT-7/3x1 (256), embed_dim = d_model
        self.num_heads = cfg.encoder.num_heads # CCT-7/3x1 (4)
        self.head_dim = self.embed_dim // self.num_heads # d_k = d_model // num_heads

        self.heads = nn.ModuleList(
            [AttentionHead(self.embed_dim, self.head_dim) for _ in range(self.num_heads)]
        )
        self.fc_layer = nn.Linear(self.embed_dim, self.embed_dim) # modified

    # Mask
    def forward(self, hidden_state, mask=None):

        outs = torch.cat([h(hidden_state, hidden_state, hidden_state, mask=mask)
													for h in self.heads], dim=-1)
        outs = self.fc_layer(outs)
        return outs