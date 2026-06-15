import numpy as np
from scipy.special import softmax, erf

class VisionTransformer:
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 num_classes: int = 1000, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0,
                 W_patch=None, cls_token=None, pos_embed=None,
                 encoder_weights=None, W_head=None):
        """
        Initialize Vision Transformer. If weight arrays are provided, use them;
        otherwise initialize randomly.
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        self.W_patch = W_patch
        self.cls_token = cls_token
        self.pos_embed = pos_embed
        self.W1 = encoder_weights[0]["W1"]
        self.W2 = encoder_weights[0]["W2"]
        self.Wq = encoder_weights[0]["Wq"]
        self.Wk = encoder_weights[0]["Wk"]
        self.Wv = encoder_weights[0]["Wv"]
        self.Wo = encoder_weights[0]["Wo"]
        self.Whead = W_head
        
        # Initialize weights here

        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        """
        # YOUR CODE HERE
        def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
            """
            Convert image to patch embeddings.
            W_proj: projection matrix of shape (patch_dim, embed_dim). If None, initialize randomly.
            """
        # YOUR CODE HERE
            b, h, w, c = image.shape
            p = patch_size
            n = h//p * w//p
            patches = image.reshape(b, h//p,p,w//p,p,c)
            patches = patches.transpose(0,1,3,2,4,5).reshape(b,n,p*p*c)
            
            patch_dim = p**2 *c
            if W_proj is None:
                W_proj = np.random.randn(patch_dim, embed_dim)*0.02
            return patches@W_proj
        def prepend_class_token(patches, embed_dim, cls_token=None):
            b, n, d = patches.shape
            if cls_token is None:
                cls_token = np.random.randn(1, 1, embed_dim) * 0.02
            cls_token = np.asarray(cls_token).reshape(1, 1, d)   # normalize any incoming shape
            cls = np.broadcast_to(cls_token, (b, 1, d))          # one CLS per image in batch
            return np.concatenate([cls, patches], axis=1)        # (b, n+1, d)
          

        def add_position_embedding(patches: np.ndarray, num_patches: int, embed_dim: int, pos_embed: np.ndarray = None) -> np.ndarray:
                """
                Add position embeddings to patch embeddings.
                pos_embed: position embedding of shape (1, N, D). If None, initialize randomly.
                """
                # YOUR CODE HERE
                if pos_embed is None:
                    pos_embed = np.random.randn(1, num_patches+1, embed_dim) * 0.02
                return patches+pos_embed

    
        def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                            Wq: np.ndarray = None, Wk: np.ndarray = None, Wv: np.ndarray = None,
                            Wo: np.ndarray = None, W1: np.ndarray = None, W2: np.ndarray = None) -> np.ndarray:
                """
                ViT Transformer encoder block with Pre-LayerNorm.
                Weight matrices are provided as inputs for deterministic testing.
                """
                # YOUR CODE HERE
                b, seq_len,embed = x.shape
                mean =x.mean(axis=-1).reshape(b,-1,1)
                var = x.var(axis=-1).reshape(b,-1,1)
                layer_normed_x = (x-mean)/(var**0.5+1e-6)
                q = layer_normed_x @ Wq #(b, seq, em) * (em, em)
                k = layer_normed_x @ Wk
                v = layer_normed_x @ Wv
                b, seq_len,embed  = q.shape
                dk = embed//num_heads
                q=q.reshape(b,seq_len,num_heads, dk).transpose(0,2,1,3) 
                k=k.reshape(b,seq_len,num_heads, dk).transpose(0,2,1,3) 
                v=v.reshape(b,seq_len,num_heads, dk).transpose(0,2,1,3) 
                y = (q@k.transpose(0,1,3,2))/(dk**0.5)
                msa = ((softmax(y, axis=-1) @ v).transpose(0,2,1,3).reshape(b,seq_len,embed))@Wo #(b,num_heads,seq_len,dk) -> (b, seq_len, embed)
                
                x_f = x+msa
                mean =x_f.mean(axis=-1).reshape(b,-1,1)
                var = x_f.var(axis=-1).reshape(b,-1,1)
                layer_normed_x_f = (x_f-mean)/(var**0.5+1e-6)
                def gelu_exact(x):
                    """Exact GELU activation function using the error function."""
                    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
            
                gelu_inner = layer_normed_x_f@W1
                mlp = gelu_exact(gelu_inner)@W2
            
                return x_f+mlp
        def classification_head(encoder_output: np.ndarray, num_classes: int, W_head: np.ndarray = None) -> np.ndarray:
                """
                Classification head for ViT. Extract [CLS], LayerNorm, linear projection.
                W_head: projection matrix (D, num_classes). If None, initialize randomly.
                """
                # YOUR CODE HERE
                h_cls = encoder_output[:,0,:]
                b, d = h_cls.shape
                mean = np.mean(h_cls, axis=-1, keepdims=True)
                var = np.var(h_cls, axis=-1, keepdims=True)
                h_cls_cap = (h_cls-mean)/(var**0.5+1e-6)
                if W_head is None:
                    W_head = np.random.randn(d, num_classes)*0.02
                logits = h_cls_cap@W_head
                return logits
        z = patch_embed(x,self.patch_size,self.embed_dim,self.W_patch)
        z = prepend_class_token(z,self.embed_dim,self.cls_token)
        z = add_position_embedding(z,self.num_patches, self.embed_dim, self.pos_embed)
        z = vit_encoder_block(z, self.embed_dim,self.num_heads,self.mlp_ratio, self.Wq,self.Wk,self.Wv,self.Wo,self.W1,self.W2)
        logits = classification_head(z,self.num_classes,self.Whead)
        return logits
        