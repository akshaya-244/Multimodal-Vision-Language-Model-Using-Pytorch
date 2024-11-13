from typing import Optional, Tuple
import torch
import torch.nn as nn

# wHY CONFIG? PAligemma comes in different sizes
class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size=768, #size of embedding vector
        intermediate_size=3072, #linear layer
        num_hidden_layers=12, #no of layers of the vision transformer
        num_attention_heads=12,
        num_channels=3, #RGB
        #Any image that is input is resized into 224x224
        image_size=224,# 224. 414, 896 different sizes 224-default
        patch_size=16, #each patch of size 16
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int=None, # no of output embeddings for each IMAGE
        **kwargs
        ):
        super().__init__()
        self.hidden_size=hidden_size
        self.intermediate_size=intermediate_size
        self.num_hidden_layers=num_hidden_layers
        self.num_attention_heads=num_attention_heads
        self.num_channels=num_channels
        self.image_size=image_size
        self.patch_size=patch_size
        self.layer_norm_eps=layer_norm_eps
        self.attention_dropout=attention_dropout
        self.num_image_tokens=num_image_tokens

class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config=config
        self.embed_dim=config.hidden_size
        self.image_size=config.image_size
        self.patch_size=config.patch_size

        self.patch_embedding=nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size, #3x3 of patches
            stride=self.patch_size, #check nb 
            padding="valid", #no padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions= self.num_patches
        
        self.position_embedding=nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1,-1)),
            persistent=False
        )
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height,width=pixel_values.shape #Batch_Size, Channels, Height, Width]
        #Convolve the patch_Size kernel over the image, with no overlapping patches since the 
        patch_embeds=self.patch_embedding(pixel_values)

        embeddings=patch_embeds.flatten(2)
        embeddings=embeddings.transpose(1,2)
        embeddings=embeddings+self.position_embedding(self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.embed_dim=config.hidden_size
        self.num_heads=config.num_attention_heads
        self.head_dim=self.embed_dim // self.num_heads
        self.scale=self.head_dim**-0.5
        self.dropout=config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj= nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj= nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj= nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        #hidden_states: [Batch_size, Num_Patches, Embed_dim]- Batch_size: batch of images. num_patches: each image is divided into num_patches patches and each patch is represented by embed_dim size of vector
        #seq_len is also num_patches
        batch_size, seq_len, _=hidden_states.size()
        #query_states: [Batch_size, Num_patches, Embed_Dim]
        #This QKV are parameters of the self attention
        query_states=self.q_proj(hidden_states)
        #key_states: [Batch_size, Num_patches, Embed_Dim]
        key_states=self.k_proj(hidden_states)
        #value_states: [Batch_size, Num_patches, Embed_Dim]
        value_states=self.v_proj(hidden_states)
        #Embed_Dim is divided into Head_dim and num_heads: 8*128=1024
        #query_states: [Batch_size, , Num_patches, Num_Heads, Head_Dim]. The transpose here makes it [Batch_size,  Num_Heads, Num_patches, Head_Dim].
        query_states=query_states.view(batch_size, seq_len, self.num_heads,self.head_dim).transpose(1,2)
        key_states=key_states.view(batch_size,seq_len,self.num_heads, self.head_dim).transpose(1,2)
        value_states=value_states.view(batch_size,seq_len, self.num_heads,self.head_dim).transpose(1,2)
        #Calculate the attention using the formula Q * K^T / sqrt (d_k).attn_weights: [Batch_size, Num_heads, num_patches, num_patches]
        attn_weights=(torch.matmul(query_states, key_states.transpose(2,3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is" f" {attn_weights.size()}")

        #Apply the softmax rowise  
        attn_weights=nn.functional.softmax(attn_weights, dim=1, dtype=torch.float32).to(query_states.dtype)
        #Apply dropout only during training
        attn_weights=nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        #Multiply the attention weights by the value states attn_output: [Batch_size, Num_Heads, Num_patches, Head_Dim]
        attn_output=torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size,self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is" f"{attn_output.size()}"
            )

        # [Batch_size, Num_Heads, Num_patches, Head_Dim] -> [Batch_size, , Num_patches, Num_Heads, Head_Dim].
        #contiguous to reshape
        attn_output = attn_output.transpose(1,2).contiguous()
        #[Batch_size, Num_patches, Num_Heads, Head_Dim] -> [Batch_size, Num_patches, Embed_dim]
        attn_output=attn_output.reshape(batch_size, seq_len,self.embed_dim)
        attn_output=self.out_proj(attn_output)

        return attn_output, attn_weights





class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.fc1=nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2=nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        #[Batch_size, Num_patches, Embed_Dim] -> [Batch_size, Num_patches, Intermediate_Size]
        hidden_states=self.fc1(hidden_states)
        #hidden_states: [Batch_size, Num_Patches, Intermediate Size]-non linearity fucntion gelu-probably gelu works better than relu-dont know the exact reason
        hidden_states=nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states=self.fc2(hidden_states)
        return hidden_states



class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()      
        self.embed_dim=config.hidden_size
        #Attention
        self.self_attn=SiglipAttention(config)
        self.layer_norm1=nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)
        self.mlp=SiglipMLP(config)
        self.layer_norm2=nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        #residual: [Batch_size num_patches, Embed_Dim]
        residual=hidden_states
        hidden_states=self.layer_norm1(hidden_states)
        hidden_states, _=self.self_attn(hidden_states=hidden_states)
        #skip +
        hidden_states=residual+hidden_states
        residual=hidden_states
        hidden_states=self.mlp(hidden_states)
        #+ skip
        hidden_states=residual+hidden_states

        return hidden_states
    
class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config=config
        self.layers=nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        #inputs_embeds: [Batch_size, Num_patches, Embed_dim]
        hidden_states=inputs_embeds
        for encoder_layer in self.layers:
            #[Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Embed_dim]
            hidden_states=encoder_layer(hidden_states)
        
        return hidden_states

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config=config
        embed_dim=config.hidden_size

        #What this does is convert image grid to convolutioned flattened embedding and add that to positional encoding
        #Get patches from embeddings
        self.embeddings=SiglipVisionEmbeddings(config)
        #now this embedding is sent to the encoder.Siglip Encoder
        self.encoder=SiglipEncoder(config) 
        self.post_layernorm=nn.LayerNorm(embed_dim, eps=config.layer_norm_eps) #TODO

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        #Batch_size, Channels, Height, Width -> [Batch_size, num_patches, Embed_dim]
        #Convert patches to embeddings
        hidden_states=self.embeddings(pixel_values)

        last_hidden_state=self.encoder(inputs_embeds=hidden_states)
        last_hidden_state=self.post_layernorm(last_hidden_state)
        return last_hidden_state

class SiglipVisionModel(nn.Module):

    def __init__(self,config: SiglipVisionConfig):

        super().__init__()
        self.config=config
        self.vision_model=SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        #Batch_size, Channels, Height, Width -> [Batch_size, num_patches, Embed_dim]
        return self.vision_model(pixel_values=pixel_values)