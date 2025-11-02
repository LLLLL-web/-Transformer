import torch
from torch import nn
import torch.nn.functional as F

class Embedding(nn.Module):
	def __init__(self,vocab_size,d_model):
		super().__init__()#必须先调用父类的方法再初始化子类自己的属性
		self.embedding=nn.Embedding(vocab_size,d_model) # 定义词嵌入层
		self.d_model=d_model #方便查看维度		
	def forward(self,x):
		#将输入序列x映射为词嵌入向量
		return self.embedding(x) #等价于self.embedding.forward(x)

class PositionalEncoding(nn.Module):
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
    def __init__(self,d_model,eps=1e-10):
        super().__init__()
        self.gamma=nn.Parameter(torch.ones(d_model))
        self.beta=nn.Parameter(torch.zeros(d_model))
        self.eps=eps
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
    def __init__(self,d_model,drop_prob):
        super().__init__()
        self.norm=LayerNorm(d_model)
        self.dropout=nn.Dropout(drop_prob)

    def forward(self,x,sublayer_output):
        # 残差连接: x + 子层输出(经过dropout)，然后进行LayerNorm
        return self.norm(x+self.dropout(sublayer_output))


class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,hidden,dropout=0.1):
        super().__init__()
        self.fc1=nn.Linear(d_model,hidden)
        self.fc2=nn.Linear(hidden,d_model)
        self.dropout=nn.Dropout(dropout)
    # 输入 → Linear(d_model→hidden) → ReLU → Dropout → Linear(hidden→d_model) → 输出
    def forward(self,x):
        x=self.fc1(x)    # 扩展维度
        x=F.relu(x)      # 非线性激活
        x=self.dropout(x) # 随机失活
        x=self.fc2(x)    # 恢复维度
        return x



def test_components():
    """测试所有定义的 Transformer 组件。"""
    
    # 共同参数
    D_MODEL = 512
    VOCAB_SIZE = 10000
    SEQ_LEN = 20
    BATCH_SIZE = 32
    NUM_HEADS = 8
    DROPOUT_RATE = 0.1
    FFN_HIDDEN = 2048 # 通常是 d_model * 4

    print("--- 开始测试 Transformer 组件 ---")

    # 1. Embedding 测试
    print("1. 测试 Embedding...")
    embedding_layer = Embedding(VOCAB_SIZE, D_MODEL)
    # 输入：[batch_size, seq_len]
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    embedded_output = embedding_layer(input_ids)
    print(f"   输入形状: {input_ids.shape}")
    print(f"   输出形状: {embedded_output.shape}")
    assert embedded_output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
    print("   ✅ Embedding 测试通过")
    print("-" * 20)

    # 2. PositionalEncoding 测试
    print("2. 测试 PositionalEncoding...")
    pe_layer = PositionalEncoding(D_MODEL, DROPOUT_RATE)
    # 输入：[batch_size, seq_len, d_model] (即 embedded_output)
    pe_output = pe_layer(embedded_output)
    print(f"   输入形状: {embedded_output.shape}")
    print(f"   输出形状: {pe_output.shape}")
    assert pe_output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
    print("   ✅ PositionalEncoding 测试通过")
    print("-" * 20)

    # 3. MultiHeadAttention 测试
    print("3. 测试 MultiHeadAttention (Self-Attention)...")
    mha_layer = MultiHeadAttention(D_MODEL, NUM_HEADS, DROPOUT_RATE)
    # Q, K, V 都使用 pe_output (Self-Attention)
    qkv = pe_output
    # 创建一个简单的 Look-ahead Mask (用于解码器，下三角矩阵)
    # 形状 [1, 1, seq_len, seq_len]
    attn_mask = (torch.ones(SEQ_LEN, SEQ_LEN).tril() == 1).unsqueeze(0).unsqueeze(0)

    mha_output = mha_layer(qkv, qkv, qkv, mask=attn_mask)
    print(f"   输入(QKV)形状: {qkv.shape}")
    print(f"   输出形状: {mha_output.shape}")
    assert mha_output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
    print("   ✅ MultiHeadAttention 测试通过")
    print("-" * 20)
    
    # 4. LayerNorm 测试
    print("4. 测试 LayerNorm...")
    ln_layer = LayerNorm(D_MODEL)
    ln_output = ln_layer(pe_output)
    print(f"   输入形状: {pe_output.shape}")
    print(f"   输出形状: {ln_output.shape}")
    assert ln_output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
    print("   ✅ LayerNorm 测试通过")
    print("-" * 20)

    # 5. PositionwiseFeedForward 测试
    print("5. 测试 PositionwiseFeedForward...")
    ffn_layer = PositionwiseFeedForward(D_MODEL, FFN_HIDDEN, DROPOUT_RATE)
    ffn_output = ffn_layer(pe_output)
    print(f"   输入形状: {pe_output.shape}")
    print(f"   输出形状: {ffn_output.shape}")
    assert ffn_output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
    print("   ✅ PositionwiseFeedForward 测试通过")
    print("-" * 20)

    # 6. ResidualConnection 测试
    print("6. 测试 ResidualConnection...")
    res_layer = ResidualConnection(D_MODEL, DROPOUT_RATE)
    # x: pe_output (未经子层处理的输入)
    # sublayer_output: mha_output (子层的输出)
    res_output = res_layer(pe_output, mha_output)
    print(f"   输入(x)形状: {pe_output.shape}")
    print(f"   子层输出形状: {mha_output.shape}")
    print(f"   输出形状: {res_output.shape}")
    assert res_output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
    print("   ✅ ResidualConnection 测试通过")
    print("-" * 20)
    
    print("\n🎉 所有组件的形状测试均通过!")

# 执行测试
if __name__ == '__main__':
    test_components()