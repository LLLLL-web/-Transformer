import torch
from torch import nn
import torch.nn.functional as F

class Embedding(nn.Module):
    """词嵌入层，将输入的词索引映射为密集向量表示"""
    def __init__(self,vocab_size,d_model):
        super().__init__()#必须先调用父类的方法再初始化子类自己的属性
        self.embedding=nn.Embedding(vocab_size,d_model) # 定义词嵌入层
        self.d_model=d_model #方便查看维度		
    def forward(self,x):
        #将输入序列x映射为词嵌入向量
        return self.embedding(x) #等价于self.embedding.forward(x)

class PositionalEncoding(nn.Module):
    """位置编码层，为输入序列添加位置信息，使用正弦和余弦函数生成位置编码"""
    def __init__(self,d_model,dropout,max_len=5000):
        #有默认值的参数必须放在没有默认值的参数后面，否则会报错
        super().__init__()
        assert d_model%2==0 #必须为偶数
        self.dropout=nn.Dropout(dropout)
        self.max_len=max_len
        # 1. 创建位置编码矩阵容器
        pe=torch.zeros(max_len,d_model) # 形状：[max_len, d_model]
		# 2. 生成位置索引向量
        position=torch.arange(0,max_len).unsqueeze(1)  #形状：[max_len, 1]，
        # 3. 计算频率除数项
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(torch.log(torch.tensor(10000.0)))/d_model).unsqueeze(0)  #形状：[1,d_model/2]，也可以不扩展直接[d_model/2]利用广播机制
        # 4. 应用正弦和余弦函数生成位置编码
        pe[:,::2]=torch.sin(position*div_term)  #偶数位置使用sin
        pe[:,1::2]=torch.cos(position*div_term)  #奇数位置使用cos
        pe=pe.unsqueeze(0)  # 扩展维度，形状：[1, max_len, d_model]

        self.register_buffer('pe', pe)  #将pe注册为buffer，这样在调用model.to(device)时，pe会自动转移到对应设备上，包含在模型的状态字典 state_dict中，但不会被优化器更新
        #pe.requires_grad=False 也可以表示不需要计算梯度
    def forward(self,x):
		#位置编码相加
        x=x+self.pe[:,:x.size(1),:]  #原本pe的第1维是max_len，这里只截取实际长度，形状：[batch_size, seq_len, d_model]
        #也可写作x=x+self.pe[:,:x.size(1)]，Pytorch切片操作默认保留未指定维度的全部元素
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """多头注意力机制，将输入拆分为多个头并行计算注意力，最后拼接结果"""
    def __init__(self,d_model,num_heads,dropout):
        super().__init__()
        assert d_model%num_heads==0 #保证能拆分成整数个头

        self.key=nn.Linear(d_model,d_model) #形状都是[batch_size, seq_len, d_model]
        self.query=nn.Linear(d_model,d_model) 
        self.value=nn.Linear(d_model,d_model) 
        self.proj=nn.Linear(d_model,d_model) 

        self.d_model=d_model
        self.num_heads=num_heads
        self.head_dim=d_model//num_heads

        self.dropout=nn.Dropout(dropout)
        self.scale=torch.sqrt(torch.tensor(self.head_dim)) #缩放因子
    def forward(self,query,key,value,mask=None):
        batch_size,s_seq_len,d_model=query.shape #Source Sequence Length（源序列长度），指的是query序列的长度
        batch_size,t_seq_len,d_model=value.shape #Target Sequence Length（目标序列长度），指的是key和value序列的长度

        #1.输入线性变换
        #维度：[batch_size, num_heads, s_seq_len, head_dim]
        Q=self.query(query).view(batch_size,s_seq_len,self.num_heads,self.head_dim).permute(0,2,1,3) 
        K=self.key(key).view(batch_size,t_seq_len,self.num_heads,self.head_dim).permute(0,2,1,3) 
        V=self.value(value).view(batch_size,t_seq_len,self.num_heads,self.head_dim).permute(0,2,1,3) 

        #2.注意力分数计算（缩放点积注意力）
        #Q维度：[batch_size, num_heads, s_seq_len, head_dim]
        #K.transpose(-2, -1)：交换最后两个维度，K变为[batch_size, num_heads,head_dim, t_seq_len]
        #矩阵乘法（每个位置(i,j)表示第i个query与第j个key的相似度）
        scores=torch.matmul(Q,K.transpose(-2,-1))/self.scale #形状[batch_size,num_heads,s_seq_len,t_seq_len]
        
        #3.掩码处理
        if mask is not None: #如果存在掩码，则将掩码应用到注意力分数上
            scores=scores.masked_fill(mask==0, float('-inf')) #将掩码位置的分数设为一个很小的值，防止其在softmax中有较大权重
        
        #4.Softmax权重计算
        attention_weights=torch.softmax(scores,dim=-1)
        
        #5.Dropout正则化
        attention_weights=self.dropout(attention_weights)
        
        #6.加权求和
        #attention_weights：[batch_size, num_heads, s_seq_len, t_seq_len]
        #V：[batch_size, num_heads, t_seq_len, head_dim]
        #矩阵乘法后：[batch_size, num_heads, s_seq_len, head_dim]
        context=torch.matmul(attention_weights,V)  #形状[batch_size,num_heads,s_seq_len,head_dim]
        
        #7.多头拼接
        #重塑回原始形状: [batch_size, s_seq_len, d_model]
        context=context.permute(0,2,1,3).contiguous().view(batch_size,s_seq_len,self.d_model)

        #8.最终投影
        output=self.proj(context) #形状[batch_size,seq_len,d_model]
        return output

class LayerNorm(nn.Module):
    """层归一化，对每个样本的最后一维进行归一化，稳定训练过程"""
    def __init__(self,d_model,eps=1e-10):
        super().__init__()
        self.gamma=nn.Parameter(torch.ones(d_model)) # 缩放参数
        self.beta=nn.Parameter(torch.zeros(d_model)) # 平移参数
        self.eps=eps # 防止除零的小常数
        
    def forward(self,x):
        #1.计算均值和方差
        mean=x.mean(-1,keepdim=True)
        var=x.var(-1,unbiased=False,keepdim=True)

        #2.归一化计算
        out=(x-mean)/torch.sqrt(var+self.eps)

        #3.缩放和平移
        out=self.gamma*out+self.beta
        return out

class ResidualConnection(nn.Module):
    """残差连接，将子层的输出与原始输入相加，缓解梯度消失问题"""
    def __init__(self,d_model,drop_prob):
        super().__init__()
        self.norm=LayerNorm(d_model)  #用自己定义的LayerNorm，也可以用nn.LayerNorm
        self.dropout=nn.Dropout(drop_prob)

    def forward(self,x,sublayer):
        # Pre-LN: 先进行LayerNorm，再传入子层，然后dropout和残差连接
        # sublayer 是一个可调用对象（如MultiHeadAttention实例）
        return x+self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络，两层全连接层，中间使用GELU激活函数"""
    def __init__(self,d_model,hidden,dropout=0.1):
        super().__init__()
        self.fc1=nn.Linear(d_model,hidden)
        self.fc2=nn.Linear(hidden,d_model)
        self.dropout=nn.Dropout(dropout)
    # 输入 → Linear(d_model→hidden) → ReLU → Dropout → Linear(hidden→d_model) → 输出
    def forward(self,x):
        x=self.fc1(x)    # 扩展维度
        x=F.gelu(x)      # 替换ReLU为GELU
        x=self.dropout(x) # 随机失活
        x=self.fc2(x)    # 恢复维度
        return x

class EncoderLayer(nn.Module):
    """编码器层，包含自注意力机制和前馈网络两个子层"""
    def __init__(self,d_model,num_heads,hidden,dropout):
        super().__init__()
        self.self_attention=MultiHeadAttention(d_model,num_heads,dropout)
        self.feed_forward=PositionwiseFeedForward(d_model,hidden,dropout)
        self.residual1=ResidualConnection(d_model,dropout)
        self.residual2=ResidualConnection(d_model,dropout)
    def forward(self,x,mask=None):
        # 第一个子层：自注意力 + 残差连接&归一化
        x=self.residual1(x,lambda x: self.self_attention(x,x,x,mask))
        # 第二个子层：前馈网络 + 残差连接&归一化
        x=self.residual2(x,self.feed_forward)
        return x

class DecoderLayer(nn.Module):
    """解码器层，包含掩码自注意力、交叉注意力和前馈网络三个子层"""
    def __init__(self,d_model,num_heads,hidden,dropout):
        super().__init__()
        self.self_attention=MultiHeadAttention(d_model,num_heads,dropout)
        self.cross_attention=MultiHeadAttention(d_model,num_heads,dropout)
        self.feed_forward=PositionwiseFeedForward(d_model,hidden,dropout)
        self.residual1=ResidualConnection(d_model,dropout)
        self.residual2=ResidualConnection(d_model,dropout)
        self.residual3=ResidualConnection(d_model,dropout)
    def forward(self,x,encoder_output,src_mask=None,tgt_mask=None):
        # 第一个子层：掩码自注意力 + 残差连接&归一化
        x=self.residual1(x,lambda x: self.self_attention(x,x,x,tgt_mask))
        # 第二个子层：交叉注意力 + 残差连接&归一化
        x=self.residual2(x,lambda x: self.cross_attention(x,encoder_output,encoder_output,src_mask))
        # 第三个子层：前馈网络 + 残差连接&归一化
        x=self.residual3(x,self.feed_forward)
        return x

class Encoder(nn.Module):
    """编码器，由词嵌入、位置编码和多个编码器层堆叠而成"""
    def __init__(self,vocab_size,d_model,num_heads,hidden,num_layers,dropout,max_len=5000):
        super().__init__()
        self.embedding=Embedding(vocab_size,d_model)
        self.positional_encoding=PositionalEncoding(d_model,dropout,max_len)
        self.layers=nn.ModuleList([
            EncoderLayer(d_model,num_heads,hidden,dropout)
            for _ in range(num_layers)
        ])
        self.norm=LayerNorm(d_model)
    def forward(self,x,mask=None):
        # 词嵌入 + 位置编码
        x=self.embedding(x)
        x=self.positional_encoding(x)
        
        # 通过所有编码器层
        for layer in self.layers:
            x=layer(x,mask)

        return x

class Decoder(nn.Module):
    """解码器，由词嵌入、位置编码和多个解码器层堆叠而成，接收编码器输出作为上下文"""
    def __init__(self,vocab_size,d_model,num_heads,hidden,num_layers,dropout,max_len=5000):
        super().__init__()
        self.embedding=Embedding(vocab_size,d_model)
        self.positional_encoding=PositionalEncoding(d_model,dropout,max_len)
        self.layers=nn.ModuleList([
            DecoderLayer(d_model,num_heads,hidden,dropout)
            for _ in range(num_layers)
        ])
        self.norm=LayerNorm(d_model)
    def forward(self,x,encoder_output,src_mask=None,tgt_mask=None):
        # 词嵌入 + 位置编码
        x=self.embedding(x)
        x=self.positional_encoding(x)
        
        # 通过所有解码器层
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)

        return x

class Transformer(nn.Module):
    """完整的Transformer模型，包含编码器、解码器和输出投影层"""
    def __init__(self,src_vocab_size,tgt_vocab_size,d_model,num_heads,hidden,
                 num_layers,dropout,max_len=5000):
        super().__init__()
        self.encoder=Encoder(src_vocab_size,d_model,num_heads,hidden,num_layers,dropout,max_len)
        self.decoder=Decoder(tgt_vocab_size,d_model,num_heads,hidden,num_layers,dropout,max_len)
        self.output_projection=nn.Linear(d_model,tgt_vocab_size)
        
    def forward(self,src,tgt,src_mask=None,tgt_mask=None):
        # 编码器前向传播
        encoder_output=self.encoder(src,src_mask)
        # 解码器前向传播
        decoder_output=self.decoder(tgt,encoder_output,src_mask,tgt_mask)
        # 输出logits（原始分数）
        output=self.output_projection(decoder_output)
        return output
# 测试代码
def test_transformer_components():
    print("测试Transformer各个组件...")
    
    # 设置参数
    batch_size = 2
    seq_len = 10
    src_vocab_size = 1000
    tgt_vocab_size = 800
    d_model = 512
    num_heads = 8
    hidden = 2048
    num_layers = 6
    dropout = 0.1
    
    # 创建模型实例
    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        hidden=hidden,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # 创建测试数据
    src_seq = torch.randint(0, src_vocab_size, (batch_size, seq_len))
    tgt_seq = torch.randint(0, tgt_vocab_size, (batch_size, seq_len))
    
    print(f"输入形状: src={src_seq.shape}, tgt={tgt_seq.shape}")
    
    # 测试前向传播
    with torch.no_grad():
        output = transformer(src_seq, tgt_seq)
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    # 测试单个编码器层
    print("\n测试编码器层...")
    encoder_layer = EncoderLayer(d_model, num_heads, hidden, dropout)
    encoder_input = torch.randn(batch_size, seq_len, d_model)
    encoder_output = encoder_layer(encoder_input)
    print(f"编码器层输入形状: {encoder_input.shape}")
    print(f"编码器层输出形状: {encoder_output.shape}")
    
    # 测试单个解码器层
    print("\n测试解码器层...")
    decoder_layer = DecoderLayer(d_model, num_heads, hidden, dropout)
    decoder_input = torch.randn(batch_size, seq_len, d_model)
    encoder_output_for_decoder = torch.randn(batch_size, seq_len, d_model)
    decoder_output = decoder_layer(decoder_input, encoder_output_for_decoder)
    print(f"解码器层输入形状: {decoder_input.shape}")
    print(f"解码器层输出形状: {decoder_output.shape}")
    
    # 测试掩码功能
    print("\n测试掩码功能...")
    # 创建因果掩码（防止看到未来信息）
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    print(f"因果掩码形状: {causal_mask.shape}")
    
    # 测试带掩码的前向传播
    with torch.no_grad():
        masked_output = transformer(src_seq, tgt_seq, tgt_mask=causal_mask)
        print(f"带掩码的输出形状: {masked_output.shape}")
    
    # 测试参数数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_parameters(transformer)
    print(f"\n模型总参数量: {total_params:,}")
    
    # 测试各组件参数量
    encoder_params = count_parameters(transformer.encoder)
    decoder_params = count_parameters(transformer.decoder)
    print(f"编码器参数量: {encoder_params:,}")
    print(f"解码器参数量: {decoder_params:,}")
    
    print("\n所有测试完成！")

if __name__ == "__main__":
    test_transformer_components()

