# MemoReader: Large-Scale Reading Comprehension through Neural MemoryController #

## Motivation

Existing approaches are still limited in understanding, up to a few paragraphs, failing to properly comprehend lengthy document. 

## Contribution

- an advanced memory-augmented architecture
- an expanded gated recurrent unit with dense connections that mitigate potential information distortion occurring in the memory. 

## Proposed Metohd

### Memory Controller

$$ o_t,i_t=Controller(d_t,M_{t-1}) $$

At time $t$, the controller generates an interface vector $i_t$ for read and write operations and an output vector $o_t$  based on the input vector $d_t$ and the external memory content from previous time step, $M_{t-1} \in \mathbb{R}^{p \times q}$, where $p$ is the memory size and $q$ is the vector dimension of each memory.

$$\{X_t\}^n_{t=1}=EncoderBlock^x(D) \in \mathbb{R}^{n\times k}$$

$$z_t=\[x_t;m^1_{t-1};\cdots;m^s_{t-1}\]\in \mathbb{R}^{k+sq}$$

$$h^m_t=BiGRU(z_t,h^m_{t-1},h^m_{t+1})\in \mathbb{R}^{2l}$$

$$v_t=W_h h_t^m + W_m\[m^1_t;\cdots;m^s_t\]\in\mathbb{R}^{2l}$$

$$o_t=ReLU(W_vv_t+d_t)\in\mathbb{R}^l$$

### Dense Encoder Block with Self Attention  

$$r_t=BiGRU(p_t,r_{t-1},r_{t+1})\in\mathbb{R}^{2l}$$

$$g_t$$=\[r_t;p_t\]\in\mathbb{R}^{3l}$$

$$s^g_{ij}=w_a\cdot g_i+w_b\cdot g_j + w_f\cdot(g_i\bigodot g_j) \in \mathbb{R}^{n \times n}$$

$$A^g=Softmax(S^g, dim=1), G=\{g_t\}^n_{t=1},Q=\{q_t\}_(t=1)^n=A^gG\in\mathbb{R}^{n\times 3l}$$


## Reading Comprehension Model with Proposed Components  

char embedding $e_w$, char embedding $e_c$, $e=\[e_w;e_c\]$\in \mathbb{R}^{400}$

$$E^q=\{e_u^q\}_{u=1}^m \in \mathbb{R}^{n \times 400}, m=document_length$$

$$E^d=\{e_t^d\}_{t=1}^n \in \mathbb{R}^{n \times 400}, n=question_length$$  

$$C^q={c^q_u}^m_{n=1}=EncoderBlock^q(E^q) \in \mathbb{R}^{m \times k}$$

$$C^d={c^d_t}^n_{t=1}=EncoderBlock^d(E^d) \in \mathbb{R}^{n \times k}$$

$$s_{ij}=w_q \cdot c^q_i + w_d \cdot c^d_j + w_c \cdot (c^q_i \bigodot c^d_j)$$

$$\widetilde{C}^q=\{\widetilde{c}^q_t\}^n_{t=1}=A^TC^q \in \mathbb{R}^{n \times k}$$

$$\widetilde{c}^d=\sum ^n_{t=1} \widetilde{a}_tc^d_t \in \mathbb{R}^k $$

$$d_t=\varphi([\[c_t^d;\widetilde{c}_t^q;c_t^d\bigodot\widetilde{c}^q_t;c^d_t \bigodot \widetilde{c}^d_t\]) \in \mathbb{R}^l $$








