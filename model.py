import torch
from torch import Tensor, nn
import torch.nn.functional as F
import fairseq

class SSLModel(nn.Module):
    """
    Args:
        device: Device (cuda/cpu)
        freeze_xlsr: Whether to freeze XLSR parameters
            - True: Freeze all parameters, only extract features (no XLSR update)
            - False: Unfreeze parameters, allow fine-tuning (will update XLSR)
    """
    def __init__(self, device, freeze_xlsr=True, finetuned_model_path=None):
        super(SSLModel, self).__init__()
        if freeze_xlsr:
            print("XLSR: Loading base model structure")
        # Replace this path with the actual path where you have the frozen XLSR-53-300m model checkpoint "xlsr2_300m.pt" 
        cp_path = ''
        
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0].to(device)
        self.device = device
        self.out_dim = 1024 #XLSR_DIM_INPUT
        self.freeze_xlsr = freeze_xlsr

        # Load finetuned weights if path provided
        if finetuned_model_path is not None:
            checkpoint = torch.load(finetuned_model_path, map_location=device, weights_only=False)
            # Load the XLSR model weights
            if 'ssl_model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['ssl_model_state_dict'])
            else:
                raise KeyError("Checkpoint do not contain 'ssl_model_state_dict' key")
        else:
            print("XLSR: Using original pretrained model")

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        

    def extract_feat(self, input_data):
        """
        Extract XLSR features
        
        Args:
            input_data: Audio input
        
        Returns:
            embedding: Output features from the last layer
            layerresult: Outputs from all layers
        """
        if next(self.model.parameters()).device != input_data.device:
            self.model.to(device=input_data.device)
        if next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(dtype=input_data.dtype)
        
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]

        else:
            input_tmp = input_data

        # Extract features
        model_output = self.model(input_tmp, mask=False, features_only=True)
        embedding = model_output['x']  # Features from the last layer
        layerresult = model_output['layer_results']  # Features from all layers
        return embedding, layerresult

class PoolAttFF(nn.Module):
    """
    Attention pooling network

    Args:
        dim_hidden: Hidden dimension of input features
        dropout: Dropout rate for attention network
    """
    def __init__(self, dim_hidden, dropout):
        super().__init__()
        self.dim_hidden = dim_hidden

        self.linear1 = nn.Linear(self.dim_hidden, 2 * self.dim_hidden)
        self.linear2 = nn.Linear(2 * self.dim_hidden, 1)

        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, seq_len) - 1 for real data, 0 for padding (optional)

        Returns:
            x_pooled: (batch_size, hidden_dim) - pooled features
        """
        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2, 1)

        # Apply mask if provided
        if mask is not None:
            expanded_mask = mask.unsqueeze(1)
            mask_positions = (expanded_mask == 0)
            att = att.masked_fill(mask_positions, float('-inf'))

        # softmax(-inf) = 0
        att = F.softmax(att, dim=2)
        x_pooled = torch.bmm(att, x).squeeze(1)
        return x_pooled

class AD_XLSR_Model(nn.Module):
    """
    AD detection model for XLSR features

    Input:
        - x: (batch_size, seq_len, 1024) - XLSR features
        - mask: (batch_size, seq_len) - attention mask

    Output:
        - logits: (batch_size, 2) - Control and Dementia logits
    """

    def __init__(self, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.norm = nn.BatchNorm1d(1024)

        self.linear_layer1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.linear_layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.linear_layer3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.linear_layer4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)

        self.linear_layer5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)

        self.dropout = nn.Dropout(dropout)

        self.pool_ad = PoolAttFF(
            dim_hidden=32,
            dropout=dropout)

        self.output_layer = nn.Linear(32, 2)
    
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, 1024) - XLSR features
            mask: (batch_size, seq_len) - attention mask (1=real, 0=padding), optional

        Returns:
            out: (batch_size, 2) - AD classification logits
        """
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.linear_layer1(x)
        x = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)

        x = self.linear_layer2(x)
        x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)

        x = self.linear_layer3(x)
        x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)

        x = self.linear_layer4(x)
        x = self.bn4(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.linear_layer5(x)
        x = self.bn5(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout(x)

        x_pooled = self.pool_ad(x, mask)
        out = self.output_layer(x_pooled)
        return out


class AD_EGE_Model(nn.Module):
    """
    AD detection model for eGeMAPS features

    Input:
        - x: (batch_size, 10, 25) - eGeMAPS features

    Output:
        - logits: (batch_size, 2) - Control and Dementia logits
    """

    def __init__(self, dim_input=25, dim_hidden=14, dropout=0.3):
        super().__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dropout = nn.Dropout(dropout)
        
        self.linear_layer1 = nn.Linear(25,64)
        self.norm1 = nn.BatchNorm1d(64)
        
        self.linear_layer2 = nn.Linear(64, 32)
        self.norm2 = nn.BatchNorm1d(32)

        self.pool_ad = PoolAttFF(dim_hidden=32, dropout=dropout)
        self.output_layer = nn.Linear(32, 2)
    
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, 25) - eGeMAPS features
            mask: Not used for eGeMAPS (no padding needed)

        Returns:
            out: (batch_size, 2) - AD classification logits
        """
        x = self.linear_layer1(x)
        x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.linear_layer2(x)
        x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout(x)
        
        x_pooled = self.pool_ad(x, mask)
        out = self.output_layer(x_pooled)
        return out