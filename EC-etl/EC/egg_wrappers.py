import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical, Categorical
from .modules import hook_print_grad
import math
from typing import Optional

def gumbel_softmax_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    training: bool = True,
    straight_through: bool = False,
):

    size = logits.size()
    if not training:
        indexes = logits.argmax(dim=-1)
        one_hot = torch.zeros_like(logits).view(-1, size[-1])
        one_hot.scatter_(1, indexes.view(-1, 1), 1)
        one_hot = one_hot.view(*size)
        return one_hot

    sample = RelaxedOneHotCategorical(logits=logits, temperature=temperature).rsample()

    if straight_through:
        size = sample.size()
        indexes = sample.argmax(dim=-1)
        hard_sample = torch.zeros_like(sample).view(-1, size[-1])
        hard_sample.scatter_(1, indexes.view(-1, 1), 1)
        hard_sample = hard_sample.view(*size)

        sample = sample + (hard_sample - sample).detach()
    return sample

class RnnSenderGS(nn.Module):
    """
    Gumbel Softmax wrapper for Sender that outputs variable-length sequence of symbols.
    The user-defined `agent` takes an input and outputs an initial hidden state vector for the RNN cell;
    `RnnSenderGS` then unrolls this RNN for the `max_len` symbols. The end-of-sequence logic
    is supposed to be handled by the game implementation. Supports vanilla RNN ('rnn'), GRU ('gru'), and LSTM ('lstm')
    cells.
    >>> class Sender(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc_out = nn.Linear(10, 5) #  input size 10, the RNN's hidden size is 5
    ...     def forward(self, x, _aux_input=None):
    ...         return self.fc_out(x)
    >>> agent = Sender()
    >>> agent = RnnSenderGS(agent, vocab_size=2, embed_dim=10, hidden_size=5, max_len=3, temperature=1.0, cell='lstm')
    >>> output = agent(torch.ones((1, 10)))
    >>> output.size()  # batch size x max_len+1 x vocab_size
    torch.Size([1, 4, 2])
    """

    def __init__(
        self,
        agent,
        vocab_size,
        embed_dim,
        hidden_size,
        max_len,
        temperature,
        cell="rnn",
        trainable_temperature=False,
        straight_through=False,
    ):
        super(RnnSenderGS, self).__init__()
        self.agent = agent

        assert max_len >= 1, "Cannot have a max_len below 1"
        self.max_len = max_len

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Linear(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )

        self.straight_through = straight_through
        self.cell = None

        cell = cell.lower()

        if cell == "rnn":
            self.cell = nn.RNNCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "gru":
            self.cell = nn.GRUCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "lstm":
            self.cell = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x, aux_input=None):
        
        prev_hidden = self.agent(x, aux_input)
        # if self.training:
        #     prev_hidden.register_hook(hook_print_grad)
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))
        sequence = []

        for step in range(self.max_len):
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)

            
            step_logits = self.hidden_to_output(h_t)
            
            # x = step_logits # fp
            # if self.training:
            #     step_logits.register_hook(hook_print_grad)
            
            
            # x = F.gumbel_softmax(step_logits, tau=self.temperature, dim=-1)

            x = gumbel_softmax_sample(
                step_logits, self.temperature, self.training, self.straight_through
            )
            
            prev_hidden = h_t
            
            e_t = self.embedding(x)
            

            sequence.append(x)

        # if self.training:
        #     sequence[0].register_hook(hook_print_grad)
        sequence = torch.stack(sequence).permute(1, 0, 2)

        eos = torch.zeros_like(sequence[:, 0, :]).unsqueeze(1)
        eos[:, 0, 0] = 1
        sequence = torch.cat([sequence, eos], dim=1)
        # if self.training:
        #     sequence.register_hook(hook_print_grad)
        return sequence


class RnnReceiverGS(nn.Module):
    """
    Gumbel Softmax-based wrapper for Receiver agent in variable-length communication game. The user implemented logic
    is passed in `agent` and is responsible for mapping (RNN's hidden state + Receiver's optional input)
    into the output vector. Since, due to the relaxation, end-of-sequence symbol might have non-zero probability at
    each timestep of the message, `RnnReceiverGS` is applied for each timestep. The corresponding EOS logic
    is handled by `SenderReceiverRnnGS`.
    """

    def __init__(self, agent, vocab_size, embed_dim, hidden_size, cell="rnn"):
        super(RnnReceiverGS, self).__init__()
        self.agent = agent

        self.cell = None
        cell = cell.lower()
        if cell == "rnn":
            self.cell = nn.RNNCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "gru":
            self.cell = nn.GRUCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "lstm":
            self.cell = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.embedding = nn.Linear(vocab_size, embed_dim)

    def forward(self, message, input=None, aux_input=None):
        outputs = []
        
        emb = self.embedding(message)

        prev_hidden = None
        prev_c = None

        # to get an access to the hidden states, we have to unroll the cell ourselves
        for step in range(message.size(1)):
            e_t = emb[:, step, ...]
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = (
                    self.cell(e_t, (prev_hidden, prev_c))
                    if prev_hidden is not None
                    else self.cell(e_t)
                )
            else:
                h_t = self.cell(e_t, prev_hidden)

            outputs.append(self.agent(h_t, input, aux_input))
            prev_hidden = h_t

        outputs = torch.stack(outputs).permute(1, 0, 2)

        return outputs


def find_lengths(messages: torch.Tensor):
    """
    :param messages: A tensor of term ids, encoded as Long values, of size (batch size, max sequence length).
    :returns A tensor with lengths of the sequences, including the end-of-sequence symbol <eos> (in EGG, it is 0).
    If no <eos> is found, the full length is returned (i.e. messages.size(1)).
    >>> messages = torch.tensor([[1, 1, 0, 0, 0, 1], [1, 1, 1, 10, 100500, 5]])
    >>> lengths = find_lengths(messages)
    >>> lengths
    tensor([3, 6])
    """
    max_k = messages.size(1)
    zero_mask = messages == 0
    # a bit involved logic, but it seems to be faster for large batches than slicing batch dimension and
    # querying torch.nonzero()
    # zero_mask contains ones on positions where 0 occur in the outputs, and 1 otherwise
    # zero_mask.cumsum(dim=1) would contain non-zeros on all positions after 0 occurred
    # zero_mask.cumsum(dim=1) > 0 would contain ones on all positions after 0 occurred
    # (zero_mask.cumsum(dim=1) > 0).sum(dim=1) equates to the number of steps  happened after 0 occured (including it)
    # max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1) is the number of steps before 0 took place

    lengths = max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1)
    lengths.add_(1).clamp_(max=max_k)

    return lengths


class RnnEncoder(nn.Module):
    """Feeds a sequence into an RNN (vanilla RNN, GRU, LSTM) cell and returns a vector representation
    of it, which is found as the last hidden state of the last RNN layer.
    Assumes that the eos token has the id equal to 0.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        n_hidden,
        cell="rnn",
        num_layers=1,
    ):
        """
        Arguments:
            vocab_size {int} -- The size of the input vocabulary (including eos)
            embed_dim {int} -- Dimensionality of the embeddings
            n_hidden {int} -- Dimensionality of the cell's hidden state
        Keyword Arguments:
            cell {str} -- Type of the cell ('rnn', 'gru', or 'lstm') (default: {'rnn'})
            num_layers {int} -- Number of the stacked RNN layers (default: {1})
        """
        super(RnnEncoder, self).__init__()

        cell = cell.lower()
        cell_types = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}

        if cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.cell = cell_types[cell](
            input_size=embed_dim,
            batch_first=True,
            hidden_size=n_hidden,
            num_layers=num_layers,
        )

        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(
        self, message, lengths=None
    ):
        """Feeds a sequence into an RNN cell and returns the last hidden state of the last layer.
        Arguments:
            message {torch.Tensor} -- A sequence to be processed, a torch.Tensor of type Long, dimensions [B, T]
        Keyword Arguments:
            lengths {Optional[torch.Tensor]} -- An optional Long tensor with messages' lengths. (default: {None})
        Returns:
            torch.Tensor -- A float tensor of [B, H]
        """
        emb = self.embedding(message)

        if lengths is None:
            lengths = find_lengths(message)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, rnn_hidden = self.cell(packed)

        if isinstance(self.cell, nn.LSTM):
            rnn_hidden, _ = rnn_hidden

        return rnn_hidden[-1]


class RnnSenderReinforce(nn.Module):
    """
    Reinforce Wrapper for Sender in variable-length message game. Assumes that during the forward,
    the wrapped agent returns the initial hidden state for a RNN cell. This cell is the unrolled by the wrapper.
    During training, the wrapper samples from the cell, getting the output message. Evaluation-time, the sampling
    is replaced by argmax.
    >>> class Agent(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(10, 3)
    ...     def forward(self, x, _input=None, _aux_input=None):
    ...         return self.fc(x)
    >>> agent = Agent()
    >>> agent = RnnSenderReinforce(agent, vocab_size=5, embed_dim=5, hidden_size=3, max_len=10, cell='lstm')
    >>> input = torch.FloatTensor(16, 10).uniform_(-0.1, 0.1)
    >>> message, logprob, entropy = agent(input)
    >>> message.size()  # batch size x max_len+1
    torch.Size([16, 11])
    >>> (entropy[:, -1] > 0).all().item()  # EOS symbol will have 0 entropy
    False
    """

    def __init__(
        self,
        agent,
        vocab_size,
        embed_dim,
        hidden_size,
        max_len,
        num_layers=1,
        cell="rnn",
    ):
        """
        :param agent: the agent to be wrapped
        :param vocab_size: the communication vocabulary size
        :param embed_dim: the size of the embedding used to embed the output symbols
        :param hidden_size: the RNN cell's hidden state size
        :param max_len: maximal length of the output messages
        :param cell: type of the cell used (rnn, gru, lstm)
        """
        super(RnnSenderReinforce, self).__init__()
        self.agent = agent

        assert max_len >= 1, "Cannot have a max_len below 1"
        self.max_len = max_len

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.cells = None

        cell = cell.lower()
        cell_types = {"rnn": nn.RNNCell, "gru": nn.GRUCell, "lstm": nn.LSTMCell}

        if cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        cell_type = cell_types[cell]
        self.cells = nn.ModuleList(
            [
                cell_type(input_size=embed_dim, hidden_size=hidden_size)
                if i == 0
                else cell_type(input_size=hidden_size, hidden_size=hidden_size)
                for i in range(self.num_layers)
            ]
        )  # noqa: E502

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x, aux_input=None):
        prev_hidden = [self.agent(x, aux_input)]
        prev_hidden.extend(
            [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)]
        )

        prev_c = [
            torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)
        ]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * x.size(0))

        sequence = []
        logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_hidden[i])
                prev_hidden[i] = h_t
                input = h_t

            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(x))

            input = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

        sequence = torch.cat([sequence, zeros.long()], dim=1)
        logits = torch.cat([logits, zeros], dim=1)
        entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy


class RnnReceiverReinforce(nn.Module):
    """
    Reinforce Wrapper for Receiver in variable-length message game. The wrapper logic feeds the message into the cell
    and calls the wrapped agent on the hidden state vector for the step that either corresponds to the EOS input to the
    input that reaches the maximal length of the sequence.
    This output is assumed to be the tuple of (output, logprob, entropy).
    """

    def __init__(
        self, agent, vocab_size, embed_dim, hidden_size, cell="rnn", num_layers=1
    ):
        super(RnnReceiverReinforce, self).__init__()
        self.agent = agent
        self.encoder = RnnEncoder(vocab_size, embed_dim, hidden_size, cell, num_layers)

    def forward(self, message, input=None, aux_input=None, lengths=None):
        encoded = self.encoder(message, lengths)
        sample, logits, entropy = self.agent(encoded, input, aux_input)

        return sample, logits, entropy


class RnnReceiverDeterministic(nn.Module):
    """
    Reinforce Wrapper for a deterministic Receiver in variable-length message game. The wrapper logic feeds the message
    into the cell and calls the wrapped agent with the hidden state that either corresponds to the end-of-sequence
    term or to the end of the sequence. The wrapper extends it with zero-valued log-prob and entropy tensors so that
    the agent becomes compatible with the SenderReceiverRnnReinforce game.
    As the wrapped agent does not sample, it has to be trained via regular back-propagation. This requires that both the
    the agent's output and  loss function and the wrapped agent are differentiable.
    >>> class Agent(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input=None, _aux_input=None):
    ...         return self.fc(rnn_output)
    >>> agent = RnnReceiverDeterministic(Agent(), vocab_size=10, embed_dim=10, hidden_size=5)
    >>> message = torch.zeros((16, 10)).long().random_(0, 10)  # batch of 16, 10 symbol length
    >>> output, logits, entropy = agent(message)
    >>> (logits == 0).all().item()
    1
    >>> (entropy == 0).all().item()
    1
    >>> output.size()
    torch.Size([16, 3])
    """

    def __init__(
        self, agent, vocab_size, embed_dim, hidden_size, cell="rnn", num_layers=1
    ):
        super(RnnReceiverDeterministic, self).__init__()
        self.agent = agent
        self.encoder = RnnEncoder(vocab_size, embed_dim, hidden_size, cell, num_layers)

    def forward(self, message, input=None, aux_input=None, lengths=None):
        encoded = self.encoder(message, lengths)
        agent_output = self.agent(encoded, input, aux_input)

        logits = torch.zeros(agent_output.size(0)).to(agent_output.device)
        entropy = logits

        return agent_output, logits, entropy

class TransformerReceiverDeterministic(nn.Module):
    def __init__(
        self,
        agent,
        vocab_size,
        max_len,
        embed_dim,
        num_heads,
        hidden_size,
        num_layers,
        positional_emb=True,
        causal=True,
    ):
        super(TransformerReceiverDeterministic, self).__init__()
        self.agent = agent
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            max_len=max_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_size=hidden_size,
            positional_embedding=positional_emb,
            causal=causal,
        )

    def forward(self, message, input=None, aux_input=None, lengths=None):
        if lengths is None:
            lengths = find_lengths(message)

        transformed = self.encoder(message, lengths)
        agent_output = self.agent(transformed, input, aux_input)

        logits = torch.zeros(agent_output.size(0)).to(agent_output.device)
        entropy = logits

        return agent_output, logits, entropy


class TransformerSenderReinforce(nn.Module):
    def __init__(
        self,
        agent,
        vocab_size,
        embed_dim,
        max_len,
        num_layers,
        num_heads,
        hidden_size,
        generate_style="standard",
        causal=True,
    ):
        """
        :param agent: the agent to be wrapped, returns the "encoder" state vector, which is the unrolled into a message
        :param vocab_size: vocab size of the message
        :param embed_dim: embedding dimensions
        :param max_len: maximal length of the message (including <eos>)
        :param num_layers: number of transformer layers
        :param num_heads: number of attention heads
        :param hidden_size: size of the FFN layers
        :param causal: whether embedding of a particular symbol should only depend on the symbols to the left
        :param generate_style: Two alternatives: 'standard' and 'in-place'. Suppose we are generating 4th symbol,
            after three symbols [s1 s2 s3] were generated.
            Then,
            'standard': [s1 s2 s3] -> embeddings [[e1] [e2] [e3]] -> (s4 = argmax(linear(e3)))
            'in-place': [s1 s2 s3] -> [s1 s2 s3 <need-symbol>] \
                                   -> embeddings [[e1] [e2] [e3] [e4]] \
                                   -> (s4 = argmax(linear(e4)))
        """
        super(TransformerSenderReinforce, self).__init__()
        self.agent = agent

        assert generate_style in ["standard", "in-place"]
        self.generate_style = generate_style
        self.causal = causal

        assert max_len >= 1, "Cannot have max_len below 1"
        self.max_len = max_len

        self.transformer = TransformerDecoder(
            embed_dim=embed_dim,
            max_len=max_len,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
        )

        self.embedding_to_vocab = nn.Linear(embed_dim, vocab_size)

        self.special_symbol_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.embed_tokens = torch.nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=self.embed_dim ** -0.5)
        self.embed_scale = math.sqrt(embed_dim)

    def generate_standard(self, encoder_state):
        batch_size = encoder_state.size(0)
        device = encoder_state.device

        sequence = []
        logits = []
        entropy = []

        special_symbol = (
            self.special_symbol_embedding.expand(batch_size, -1).unsqueeze(1).to(device)
        )
        input = special_symbol

        for step in range(self.max_len):
            if self.causal:
                attn_mask = torch.triu(
                    torch.ones(step + 1, step + 1).byte(), diagonal=1
                ).to(
                    device
                )  # noqa: E226
                attn_mask = attn_mask.float().masked_fill(attn_mask == 1, float("-inf"))
            else:
                attn_mask = None
            output = self.transformer(
                embedded_input=input, encoder_out=encoder_state, attn_mask=attn_mask
            )
            step_logits = F.log_softmax(
                self.embedding_to_vocab(output[:, -1, :]), dim=1
            )

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())
            if self.training:
                symbols = distr.sample()
            else:
                symbols = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(symbols))
            sequence.append(symbols)

            new_embedding = self.embed_tokens(symbols) * self.embed_scale
            input = torch.cat([input, new_embedding.unsqueeze(dim=1)], dim=1)

        return sequence, logits, entropy

    def generate_inplace(self, encoder_state):
        batch_size = encoder_state.size(0)
        device = encoder_state.device

        sequence = []
        logits = []
        entropy = []

        special_symbol = (
            self.special_symbol_embedding.expand(batch_size, -1)
            .unsqueeze(1)
            .to(encoder_state.device)
        )
        output = []
        for step in range(self.max_len):
            input = torch.cat(output + [special_symbol], dim=1)
            if self.causal:
                attn_mask = torch.triu(
                    torch.ones(step + 1, step + 1).byte(), diagonal=1
                ).to(
                    device
                )  # noqa: E226
                attn_mask = attn_mask.float().masked_fill(attn_mask == 1, float("-inf"))
            else:
                attn_mask = None

            embedded = self.transformer(
                embedded_input=input, encoder_out=encoder_state, attn_mask=attn_mask
            )
            step_logits = F.log_softmax(
                self.embedding_to_vocab(embedded[:, -1, :]), dim=1
            )

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())
            if self.training:
                symbols = distr.sample()
            else:
                symbols = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(symbols))
            sequence.append(symbols)

            new_embedding = self.embed_tokens(symbols) * self.embed_scale
            output.append(new_embedding.unsqueeze(dim=1))

        return sequence, logits, entropy

    def forward(self, x, aux_input=None):
        encoder_state = self.agent(x, aux_input)

        if self.generate_style == "standard":
            sequence, logits, entropy = self.generate_standard(encoder_state)
        elif self.generate_style == "in-place":
            sequence, logits, entropy = self.generate_inplace(encoder_state)
        else:
            assert False, "Unknown generate style"

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

        sequence = torch.cat([sequence, zeros.long()], dim=1)
        logits = torch.cat([logits, zeros], dim=1)
        entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy
    

class SinusoidalPositionEmbedding(nn.Module):
    """Implements sinusoidal positional embeddings"""

    def __init__(self, max_len: int, model_dim: int) -> None:
        super(SinusoidalPositionEmbedding, self).__init__()
        pos = torch.arange(0.0, max_len).unsqueeze(1).repeat(1, model_dim)
        dim = torch.arange(0.0, model_dim).unsqueeze(0).repeat(max_len, 1)
        div = torch.exp(-math.log(10000) * (2 * (dim // 2) / model_dim))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])
        self.register_buffer("pe", pos.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Updates the input embedding with positional embedding
        Arguments:
            x {torch.Tensor} -- Input tensor
        Returns:
            torch.Tensor -- Input updated with positional embeddings
        """
        # fmt: off
        t = self.pe[:, :x.size(1), :]
        # fmt: on
        return x + t


class TransformerEncoder(nn.Module):
    """Implements a Transformer Encoder. The masking is done based on the positions of the <eos>
    token (with id 0).
    Two regimes are implemented:
    * 'causal' (left-to-right): the symbols are masked such that every symbol's embedding only can depend on the
        symbols to the left of it. The embedding of the <eos> symbol is taken as the representative.
    *  'non-causal': a special symbol <sos> is pre-pended to the input sequence, all symbols before <eos> are un-masked.
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        embed_dim: int,
        num_heads: int,
        hidden_size: int,
        num_layers: int = 1,
        positional_embedding=True,
        causal: bool = True,
    ) -> None:
        super().__init__()

        # in the non-causal case, we will use a special symbol prepended to the input messages which would have
        # term id of `vocab_size`. Hence we increase the vocab size and the max length
        if not causal:
            max_len += 1
            vocab_size += 1

        self.base_encoder = TransformerBaseEncoder(
            vocab_size=vocab_size,
            max_len=max_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_size=hidden_size,
            positional_embedding=positional_embedding,
        )
        self.max_len = max_len
        self.sos_id = torch.tensor([vocab_size - 1]).long()
        self.causal = causal

    def forward(
        self, message: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if lengths is None:
            lengths = find_lengths(message)

        batch_size = message.size(0)

        if not self.causal:
            prefix = self.sos_id.to(message.device).unsqueeze(0).expand((batch_size, 1))
            message = torch.cat([prefix, message], dim=1)
            lengths = lengths + 1

            max_len = message.size(1)
            len_indicators = (
                torch.arange(max_len).expand((batch_size, max_len)).to(lengths.device)
            )
            lengths_expanded = lengths.unsqueeze(1)
            padding_mask = len_indicators >= lengths_expanded

            transformed = self.base_encoder(message, padding_mask)
            # as the input to the agent, we take the embedding for the first symbol
            # which is always the special <sos> one
            transformed = transformed[:, 0, :]
        else:
            max_len = message.size(1)
            len_indicators = (
                torch.arange(max_len).expand((batch_size, max_len)).to(lengths.device)
            )
            lengths_expanded = lengths.unsqueeze(1)
            padding_mask = len_indicators >= lengths_expanded

            attn_mask = torch.triu(torch.ones(max_len, max_len).byte(), diagonal=1).to(
                lengths.device
            )
            attn_mask = attn_mask.float().masked_fill(attn_mask == 1, float("-inf"))
            transformed = self.base_encoder(
                message, key_padding_mask=padding_mask, attn_mask=attn_mask
            )

            last_embeddings = []
            for i, l in enumerate(
                lengths.clamp(max=self.max_len - 1).cpu()
            ):  # noqa: E226
                last_embeddings.append(transformed[i, l, :])
            transformed = torch.stack(last_embeddings)

        return transformed


class TransformerBaseEncoder(torch.nn.Module):
    """
    Implements a basic Transformer Encoder module with fixed Sinusoidal embeddings.
    Initializations of the parameters are adopted from fairseq.
    Does not handle the masking w.r.t. message lengths, left-to-right order, etc.
    This is supposed to be done on a higher level.
    """

    def __init__(
        self,
        vocab_size,
        max_len,
        embed_dim,
        num_heads,
        num_layers,
        hidden_size,
        p_dropout=0.0,
        positional_embedding=True,
    ):
        super().__init__()

        # NB: they use a different one
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.embed_dim = embed_dim
        self.max_source_positions = max_len
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = (
            SinusoidalPositionEmbedding(
                max_len + 1, embed_dim  # accounting for the forced EOS added by EGG
            )
            if positional_embedding
            else None
        )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerEncoderLayer(
                    embed_dim=embed_dim, num_heads=num_heads, hidden_size=hidden_size
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = p_dropout

        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=self.embed_dim ** -0.5)

    def forward(self, src_tokens, key_padding_mask=None, attn_mask=None):
        # embed tokens and positions
        x = self.embed_scale * self.embedding(src_tokens)

        if self.embed_positions is not None:
            x = self.embed_positions(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # encoder layers
        for layer in self.layers:
            x = layer(x, key_padding_mask, attn_mask)

        x = self.layer_norm(x)

        #  T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        hidden_size,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=num_heads, dropout=attention_dropout
        )
        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)

        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.normalize_before = True
        self.fc1 = torch.nn.Linear(self.embed_dim, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, self.embed_dim)
        # it seems there are two ways to apply layer norm - before (in tensor2tensor code)
        # or after (the original paper). We resort to the first as it is suggested to be more robust
        self.layer_norm = torch.nn.LayerNorm(self.embed_dim)

        self.init_parameters()

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _att = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norm(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x

    def init_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)


class TransformerDecoder(torch.nn.Module):
    """
    Does not handle the masking w.r.t. message lengths, left-to-right order, etc.
    This is supposed to be done on a higher level.
    """

    def __init__(
        self, embed_dim, max_len, num_layers, num_heads, hidden_size, dropout=0.0
    ):
        super().__init__()

        self.dropout = dropout

        self.embed_positions = SinusoidalPositionEmbedding(max_len, embed_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(num_heads, embed_dim, hidden_size)
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, embedded_input, encoder_out, key_mask=None, attn_mask=None):
        # embed positions
        embedded_input = self.embed_positions(embedded_input)

        x = F.dropout(embedded_input, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        for layer in self.layers:
            x, attn = layer(x, encoder_out, key_mask=key_mask, attn_mask=attn_mask)

        x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block. Follows an implementation in fairseq with args.decoder_normalize_before=True,
    i.e. order of operations is different from those in the original paper.
    """

    def __init__(
        self,
        num_heads,
        embed_dim,
        hidden_size,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=num_heads, dropout=attention_dropout
        )  # self-attn?

        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)

        # NB: we pass encoder state as a single vector at the moment (form the user-defined module)
        # hence this attention layer is somewhat degenerate/redundant. Nonetherless, we'll have it
        # for (a) proper compatibility (b) in case we'll decide to pass multipel states
        self.encoder_attn = torch.nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=num_heads, dropout=attention_dropout
        )

        self.encoder_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)

        self.fc1 = torch.nn.Linear(self.embed_dim, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, self.embed_dim)

        self.layer_norm = torch.nn.LayerNorm(self.embed_dim)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x, encoder_out, key_mask=None, attn_mask=None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x, key=x, value=x, key_padding_mask=key_mask, attn_mask=attn_mask
        )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.encoder_attn_layer_norm(x)
        # would be a single vector, so no point in attention at all
        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            # static_kv=True,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norm(x)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        return x, attn