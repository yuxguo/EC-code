import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import torchvision

from .egg_wrappers import RnnSenderGS, RnnReceiverGS, RnnSenderReinforce, RnnReceiverDeterministic, find_lengths, TransformerSenderReinforce, TransformerReceiverDeterministic
from .args import Args
from .modules import *
from .visual_model import VisualModel, SymbolModel

import egg.core.baselines as baselines
from collections import defaultdict


class SpeakerAnalogyModel(nn.Module):
    # input: image embedding[B, 6, image_embedding]
    # output: rule[B, rule_embedding]
    def __init__(self, args):
        super().__init__()
        self.analogy = SharedGroupMLP(**args.discri_analogy_shared_mlp_configs)

    def forward(self, x):
        b, n, e = x.size()  # n == 6
        x = x.view(2 * b, 3, e)
        x = self.analogy(x)
        rules = x.view(b, 2, -1)
        x = torch.mean(rules, dim=1)
        return x, rules # [B, 2, 400]

class ConstrativeSpeakerAnalogyModel(nn.Module):
    # input: image embedding[B, 6, image_embedding]
    # output: rule[B, rule_embedding]
    def __init__(self, args):
        super().__init__()
        self.analogy = SharedGroupMLP(**args.constrative_analogy_shared_mlp_configs)

    def forward(self, x):
        b, n, e = x.size()  # n == 6
        x = x.view(2 * b, 3, e)
        diff_21 = x[:, 1, :] - x[:, 0, :]
        diff_32 = x[:, 2, :] - x[:, 1, :]
        x = torch.stack([diff_21, diff_32], dim=1)
        x = self.analogy(x)
        rules = x.view(b, 2, -1)
        x = torch.mean(rules, dim=1)
        return x, rules # [B, 2, 400]

class DiscriListenerAnalogyModel(nn.Module):
    # contrastive method
    def __init__(self, args):
        super().__init__()
        self.analogy = SharedGroupMLP(**args.discri_analogy_shared_mlp_configs)

    def forward(self, listener_context, listener_candidates):
        # [B, 2, image_embedding], [B, 8, image_embedding], [B, message_embedding]
        # [B, 2, image_embedding] -> [B, 1, 2, image_embedding] -> [B, 8, 2, image_embedding]
        listener_context = listener_context.unsqueeze(1).repeat(1, 8, 1, 1)

        # [B, 8, image_embedding], [B, 8, 1, image_embedding]
        listener_candidates = listener_candidates.unsqueeze(2)

        # [B, 8, 2 + 1, image_embedding]
        listener_merged = torch.cat([listener_context, listener_candidates], dim=2)

        b, n, c, e = listener_merged.size()
        listener_merged = listener_merged.view(b * n, c, e)

        analogy_embedding = self.analogy(listener_merged)
        analogy_embedding = analogy_embedding.view(b, n, -1)

        return analogy_embedding

class ConstrativeDiscriListenerAnalogyModel(nn.Module):
    # contrastive method
    def __init__(self, args):
        super().__init__()
        self.analogy = SharedGroupMLP(**args.constrative_analogy_shared_mlp_configs)

    def forward(self, listener_context, listener_candidates):
        # [B, 2, image_embedding], [B, 8, image_embedding], [B, message_embedding]
        # [B, 2, image_embedding] -> [B, 1, 2, image_embedding] -> [B, 8, 2, image_embedding]
        listener_context = listener_context.unsqueeze(1).repeat(1, 8, 1, 1)

        # [B, 8, image_embedding], [B, 8, 1, image_embedding]
        listener_candidates = listener_candidates.unsqueeze(2)

        # [B, 8, 2 + 1, image_embedding]
        listener_merged = torch.cat([listener_context, listener_candidates], dim=2)

        b, n, c, e = listener_merged.size()
        listener_merged = listener_merged.view(b * n, c, e)

        diff_21 = listener_merged[:, 1, :] - listener_merged[:, 0, :]
        diff_32 = listener_merged[:, 2, :] - listener_merged[:, 1, :]

        listener_merged = torch.stack([diff_21, diff_32], dim=1)

        analogy_embedding = self.analogy(listener_merged)
        analogy_embedding = analogy_embedding.view(b, n, -1)

        return analogy_embedding


class ReconListenerAnalogyModel(nn.Module):
    def __init__(self, args):
        super(ReconListenerAnalogyModel, self).__init__()
        self.args = args
        self.analogy = SharedGroupMLP(**args.recon_analogy_shared_mlp_configs)
        self.attn_heads = nn.ModuleList([nn.Sequential(*[nn.Linear(args.message_embedding_dim, args.rules_dim), nn.Softmax(dim=-1)]) for _ in range(args.symbol_attr_dim)])
    
    def forward(self, listener_context, listener_candidates, message_embedding):
        # [B, 2, image_embedding], [B, 8, image_embedding], [B, message_embedding]
        out = self.analogy(listener_context)
        out = out.view(-1, self.args.symbol_attr_dim, self.args.rules_dim, self.args.image_embedding_dim // self.args.symbol_attr_dim) # -1, 4, 10, 20
        out = [t.squeeze() for t in out.split(1, dim=1)]
        new_out = []
        for o, m in zip(out, self.attn_heads):
            w = m(message_embedding).unsqueeze(1)
            new_out.append(torch.matmul(w, o.squeeze())) 
        new_out = torch.cat(new_out, dim=1)
        # new_out = new_out.view(-1, self.args.image_embedding_dim) # B, 80

        return new_out


class ReconListener(nn.Module):
    # generative method
    # input: image embedding[B, 6, image_embedding]
    # output: rule[B, rule_embedding]
    def __init__(self, args):
        super(ReconListener, self).__init__()
        if args.visual:
            self.visual_model = VisualModel(args)
        elif args.symbol:
            self.visual_model = SymbolModel(args)
        self.args = args
        self.analogy_model = ReconListenerAnalogyModel(args)
        self.predict_head = nn.Linear(args.image_embedding_dim // args.symbol_attr_dim, args.symbol_onehot_dim) # 20, 10

    def forward(self, x, _input, _aux_input):
        listener_context, listener_candidates = _input
        listener_context = self.visual_model(listener_context)
        listener_candidates = self.visual_model(listener_candidates)

        out = self.analogy_model(listener_context, listener_candidates, x) # B, 80

        # generate
        # # out = out.view(-1, self.args.symbol_attr_dim, self.args.symbol_onehot_dim)
        # out = self.predict_head(out) # B, 4, 10
        # out = out.view(-1, self.args.symbol_attr_dim * self.args.symbol_onehot_dim)

        # discriminative
        out = out.view(-1, self.args.image_embedding_dim)
        dots = torch.matmul(listener_candidates, out.unsqueeze(-1)).squeeze(-1)

        return dots # B, 4 * 10 or B, 8


class Speaker(nn.Module):
    # input: image embedding[B, 6, image_embedding]
    # output: message[B, message_embedding]
    def __init__(self, args):
        super().__init__()
        if args.visual:
            self.visual_model = VisualModel(args)
        elif args.symbol:
            self.visual_model = SymbolModel(args)
        if args.speaker_freeze_visual:
            for param in self.visual_model.parameters():
                param.requires_grad = False
        
        if args.use_constrative:
            self.analogy_model = ConstrativeSpeakerAnalogyModel(args)
        else:
            self.analogy_model = SpeakerAnalogyModel(args)
        
        self.message_embedding_noise = args.message_embedding_noise
        self.max_pooling_message_embedding = args.max_pooling_message_embedding
        self.mlp_pooling_message_embedding = args.mlp_pooling_message_embedding

        if self.mlp_pooling_message_embedding:
            self.mlp_pooling = MLPModel(5, 1, [16]) # FIXME: using magic numbers
        
        self.add_LN = args.add_LN
        if self.add_LN:
            self.LN = nn.LayerNorm(args.message_embedding_dim)

    def forward(self, speaker_context, _aux_input=None):
        speaker_context = self.visual_model(speaker_context)
        # if self.training:
        #     speaker_context.register_hook(hook_print_grad)
        message_embedding, rules_embedding = self.analogy_model(speaker_context)

        if self.max_pooling_message_embedding:
            message_embedding = message_embedding.view(-1, 5, 80) # FIXME: using magic numbers
            message_embedding = torch.max(message_embedding, dim=1)

        if self.mlp_pooling_message_embedding:
            message_embedding = message_embedding.view(-1, 5, 80) # FIXME: using magic numbers
            message_embedding = message_embedding.permute(0, 2, 1)
            message_embedding = self.mlp_pooling(message_embedding).squeeze(-1)


        if self.message_embedding_noise == "gaussian":
            noise = torch.normal(0, 1, message_embedding.size()).cuda()
            message_embedding = message_embedding + noise
        
        if self.add_LN:
            message_embedding = self.LN(message_embedding)
        # if self.training:
        #     message_embedding.register_hook(hook_print_grad)

        return message_embedding


class SimpleDiscriListener(nn.Module):
    # input: message[B, 6, image_embedding], image_embedding[B, 2 + 8, image_embedding]
    # output: message[B, message_embedding]
    def __init__(self, args):
        super(SimpleDiscriListener, self).__init__()
        if args.visual:
            self.visual_model = VisualModel(args)
        elif args.symbol:
            self.visual_model = SymbolModel(args)

        self.analogy_model = nn.Sequential(*[nn.Linear(2, 80), nn.Tanh()]) # FIXME
        if args.listener_freeze_visual:
            for param in self.visual_model.parameters():
                param.requires_grad = False

    def forward(self, x, _input, _aux_input):
        # if self.training:
        #     x.register_hook(hook_print_grad)
        listener_context, listener_candidates = _input
        listener_context = self.visual_model(listener_context)
        listener_candidates = self.visual_model(listener_candidates)

        listener_context = listener_context.permute(0, 2, 1)
        listener_context = self.analogy_model(listener_context)
        pred_image = torch.matmul(listener_context, torch.unsqueeze(x, dim=-1)) # B, 80, 80 x B, 80, 1 -> B, 80, 1
        
        
        dots = torch.matmul(listener_candidates, pred_image) # B, 8, 80 x B, 80, 1
        # if self.training:
        #     dots.register_hook(hook_print_grad)
        
        return dots.squeeze()


class DiscriListener(nn.Module):
    # input: message[B, 6, image_embedding], image_embedding[B, 2 + 8, image_embedding]
    # output: message[B, message_embedding]
    def __init__(self, args):
        super(DiscriListener, self).__init__()
        if args.visual:
            self.visual_model = VisualModel(args)
        elif args.symbol:
            self.visual_model = SymbolModel(args)

        if args.use_constrative:
            self.analogy_model = ConstrativeDiscriListenerAnalogyModel(args)
        else:
            self.analogy_model = DiscriListenerAnalogyModel(args)
        if args.listener_freeze_visual:
            for param in self.visual_model.parameters():
                param.requires_grad = False

        self.max_pooling_message_embedding = args.max_pooling_message_embedding
        self.mlp_pooling_message_embedding = args.mlp_pooling_message_embedding

        if self.mlp_pooling_message_embedding:
            self.mlp_pooling = MLPModel(5, 1, [16]) # FIXME: using magic numbers
        
        self.add_LN = args.add_LN
        if self.add_LN:
            self.LN = nn.LayerNorm(args.message_embedding_dim)

    def forward(self, x, _input, _aux_input):
        # if self.training:
        #     x.register_hook(hook_print_grad)
        listener_context, listener_candidates = _input
        listener_context = self.visual_model(listener_context)

        #
        # listener_context = torch.ones_like(listener_context).cuda()

        # if self.training:
        #     listener_context.register_hook(hook_print_grad)
        listener_candidates = self.visual_model(listener_candidates)

        embedded_input = self.analogy_model(listener_context, listener_candidates).tanh() # FIXME: tanh?
        if self.max_pooling_message_embedding:
            embedded_input = embedded_input.view(-1, 5, 80) # FIXME: using magic numbers
            embedded_input = torch.max(embedded_input, dim=1)

        if self.mlp_pooling_message_embedding:
            b, n, c = embedded_input.size()
            embedded_input = embedded_input.view(b, n, 5, 80) # FIXME: using magic numbers
            embedded_input = embedded_input.permute(0, 1, 3, 2)
            embedded_input = self.mlp_pooling(embedded_input).squeeze(-1)
        
        # if self.training:
        #     embedded_input.register_hook(hook_print_grad)
            # x.register_hook(hook_print_grad)

        if self.add_LN:
            embedded_input = F.tanh(embedded_input)
            embedded_input = self.LN(embedded_input)
            
            x = self.LN(x)

        dots = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        
        
        return dots.squeeze(-1)



class EC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.length_cost = args.message_length_cost

        if args.agent_type == "rnn_reinforce":
            self.baselines = defaultdict(baselines.MeanBaseline)
            self.speaker = RnnSenderReinforce(
                Speaker(args),
                **args.rnn_reinforce_speaker_configs
            )
            if args.discri_game:
                self.listener = RnnReceiverDeterministic(
                    DiscriListener(args),
                    # SimpleDiscriListener(args),
                    **args.rnn_reinforce_listener_configs
                )
            else:
                self.listener = RnnReceiverDeterministic(
                    ReconListener(args),
                    **args.rnn_reinforce_listener_configs
                )
        elif args.agent_type == "rnn_gs":
            self.speaker = RnnSenderGS(
                Speaker(args),
                **args.rnn_gs_speaker_configs
            )
            if args.discri_game:
                self.listener = RnnReceiverGS(
                    DiscriListener(args),
                    # SimpleDiscriListener(args),
                    **args.rnn_gs_listener_configs
                )
            else:
                self.listener = RnnReceiverGS(
                    ReconListener(args),
                    **args.rnn_gs_listener_configs
                )
        elif args.agent_type == "transformer_reinforce":
            self.baselines = defaultdict(baselines.MeanBaseline)
            self.speaker = TransformerSenderReinforce(
                Speaker(args),
                **args.transformer_reinforce_speaker_configs
            )
            if args.discri_game:
                self.listener = TransformerReceiverDeterministic(
                    DiscriListener(args),
                    # SimpleDiscriListener(args),
                    **args.transformer_reinforce_listener_configs
                )
            else:
                self.listener = TransformerReceiverDeterministic(
                    ReconListener(args),
                    **args.transformer_reinforce_listener_configs
                )
        
        if args.listener_freeze:
            for param in self.listener.agent.parameters():
                param.requires_grad = False
            for param in self.listener.cell.parameters():
                param.requires_grad = False
            # for param in self.listener.embedding.parameters():
                # param.requires_grad = False


    def loss_discri(self, _sender_input, _message, _receiver_input, receiver_output, labels, _aux_input, input_dict=None):
        # print(receiver_output.size())
        pred = receiver_output.argmax(dim=1)
        acc = (pred == labels).detach().float()
        # similarly, the loss computes cross-entropy between the Receiver-produced target-position probability distribution and the labels
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        return loss, {"acc": acc, "pred": pred.detach()}

    def loss_recon(self, _sender_input, _message, _receiver_input, receiver_output, labels, _aux_input, input_dict=None):
        receiver_output = receiver_output.view(-1, self.args.symbol_attr_dim, self.args.symbol_onehot_dim)
        
        pred = receiver_output.argmax(dim=2)
        label = input_dict['target_image'] if self.args.visual else input_dict['target_symbol'] # B, 4, 10
        if self.args.symbol:
            label = label.argmax(dim=2)
        
        acc = ((pred == label).float().sum(dim=-1) == self.args.symbol_attr_dim).detach().float()

        receiver_output = receiver_output.view(-1, self.args.symbol_onehot_dim)
        
        label = label.view(-1)
        
        loss = F.cross_entropy(receiver_output, label, reduction="none")
        loss = loss.view(-1, self.args.symbol_attr_dim) #(B, 4)
        loss = loss.mean(dim=-1)

        return loss, {"acc": acc, "pred": pred.detach()}
    
    def loss_recon_2(self, _sender_input, _message, _receiver_input, receiver_output, labels, _aux_input, input_dict=None):
         # print(receiver_output.size())
        pred = receiver_output.argmax(dim=1)
        acc = (pred == labels).detach().float()
        # similarly, the loss computes cross-entropy between the Receiver-produced target-position probability distribution and the labels
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        return loss, {"acc": acc, "pred": pred.detach()}

    def gs_game(self, speaker_input, listener_input, label, aux_input=None, input_dict=None):
        loss_f = self.loss_discri if self.args.discri_game else self.loss_recon
        message = self.speaker(speaker_input, aux_input)
        # print(message.size())
        # if self.training:
        #     message.register_hook(hook_print_grad)
        
        if self.args.rule:
            # need adjust message to 4x4
            message = input_dict['rules'].view(-1, 4, 10)
            eos = torch.zeros_like(message[:, 0, :]).unsqueeze(1)
            eos[:, 0, 0] = 1
            message = torch.cat([message, eos], dim=1)
        elif self.args.null_message:
            message = torch.zeros_like(message)
        elif self.args.const_message:
            message = torch.zeros_like(message)
            message[:, 0:self.args.message_max_len, 1] += 1.0 
            message[:, self.args.message_max_len, 0] += 1.0 # bbbba, a is eos
        
        
        receiver_output = self.listener(message, listener_input, aux_input)

        if self.args.use_message_max_len:
            step = -1
            step_loss, step_aux = loss_f(
                speaker_input,
                message[:, step, ...],
                listener_input,
                receiver_output[:, step, ...],
                label,
                aux_input,
                input_dict=input_dict
            )
            return step_loss.mean(), step_aux, message.detach()
        
        else:
            loss = 0
            
            not_eosed_before = torch.ones(receiver_output.size(0)).to(
                receiver_output.device
            )
            expected_length = 0.0

            aux_info = {}
            z = 0.0
            for step in range(receiver_output.size(1)):
                step_loss, step_aux = loss_f(
                    speaker_input,
                    message[:, step, ...],
                    listener_input,
                    receiver_output[:, step, ...],
                    label,
                    aux_input,
                    input_dict=input_dict
                )
                eos_mask = message[:, step, 0]  # always eos == 0, eos_mask==1 -> eos

                add_mask = eos_mask * not_eosed_before
                z += add_mask
                loss += step_loss * add_mask + self.length_cost * (1.0 + step) * add_mask
                expected_length += add_mask.detach() * (1.0 + step)

                for name, value in step_aux.items():
                    aux_info[name] = value * add_mask + aux_info.get(name, 0.0)

                not_eosed_before = not_eosed_before * (1.0 - eos_mask)

            # the remainder of the probability mass
            loss += (
                step_loss * not_eosed_before
                + self.length_cost * (step + 1.0) * not_eosed_before
            )
            expected_length += (step + 1) * not_eosed_before

            z += not_eosed_before
            assert z.allclose(
                torch.ones_like(z)
            ), f"lost probability mass, {z.min()}, {z.max()}"

            for name, value in step_aux.items():
                aux_info[name] = value * not_eosed_before + aux_info.get(name, 0.0)

            aux_info["length"] = expected_length

            return loss.mean(), aux_info, message.detach()

    
    def reinforce_game(self, speaker_input, listener_input, label, aux_input=None, input_dict=None, sender_entropy_coeff=0.1, receiver_entropy_coeff=0.0):
        loss_f = self.loss_discri if self.args.discri_game else self.loss_recon_2
        message, log_prob_s, entropy_s = self.speaker(speaker_input, aux_input)
        message_length = find_lengths(message)



        if self.args.rule:
            # need adjust message to 4x10
            message = input_dict['rules'].view(-1, 4, 10)
            eos = torch.zeros_like(message[:, 0, :]).unsqueeze(1)
            eos[:, 0, 0] = 1
            message = torch.cat([message, eos], dim=1)
            message = torch.argmax(message, dim=-1)

        elif self.args.null_message:
            message = torch.zeros_like(message)
        elif self.args.const_message:
            message = torch.zeros_like(message).long()
            message[:, 0:self.args.message_max_len] += 1 # bbbba, a is eos
            message_length = find_lengths(message)
        
        # print(message)
        receiver_output, log_prob_r, entropy_r = self.listener(
            message, listener_input, aux_input, message_length
        )

        loss, aux_info = loss_f(
            speaker_input, message, listener_input, receiver_output, label, aux_input, input_dict=input_dict
        )

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r)

        for i in range(message.size(1)):
            not_eosed = (i < message_length).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_length.float()

        weighted_entropy = (
            effective_entropy_s.mean() * sender_entropy_coeff
            + entropy_r.mean() * receiver_entropy_coeff
        )

        log_prob = effective_log_prob_s + log_prob_r

        length_loss = message_length.float() * self.length_cost

        policy_length_loss = (
            (length_loss - self.baselines["length"].predict(length_loss)) 
            * effective_log_prob_s
        ).mean()
        policy_loss = (
            (loss.detach() - self.baselines["loss"].predict(loss.detach())) * log_prob
        ).mean()


        # print(policy_loss.item(), policy_length_loss.item(), weighted_entropy.item(), message_length.detach().float().mean().item())
        # if self.training:
        #     print(message_length.detach().float().mean().item())
        
        optimized_loss = policy_length_loss + policy_loss - weighted_entropy
        # print(policy_length_loss, policy_loss, -weighted_entropy)
        # if the receiver is deterministic/differentiable, we apply the actual loss

        optimized_loss += loss.mean()

        # optimized_loss = loss.mean() - optimized_loss

        if self.training:
            self.baselines["loss"].update(loss)
            self.baselines["length"].update(length_loss)

        aux_info["sender_entropy"] = entropy_s.detach()
        aux_info["receiver_entropy"] = entropy_r.detach()
        aux_info["length"] = message_length.float()  # will be averaged
        aux_info['logits'] = receiver_output.detach()

        
        return optimized_loss, aux_info, message.detach()


    def forward(self, input_dict):
        if self.args.visual:
            speaker_input = input_dict['image'][:, 0:6, :].contiguous()
            listener_input = (input_dict['image'][:, 6:8, :].contiguous(), input_dict['image'][:, 8:, :].contiguous())
        elif self.args.symbol:
            speaker_input = input_dict['symbol'][:, 0:6, :].contiguous()
            listener_input = (input_dict['symbol'][:, 6:8, :].contiguous(), input_dict['symbol'][:, 8:, :].contiguous())

        # for test only
        # fake_listener_context = torch.zeros_like(input_dict['image'][:, 6:8, :]) + 127.0
        # fake_listener_context = torch.cat([torch.zeros_like(input_dict['image'][:, 6, :]).unsqueeze(1) + 127.0, input_dict['image'][:, 7, :].unsqueeze(1)], dim=1)
        # listener_input = (fake_listener_context, input_dict['image'][:, 8:, :].contiguous())

        # listener_input = (input_dict['image'][:, 6:8, :].contiguous(), input_dict['image'][:, 8:, :].contiguous())
        if self.args.agent_type in ["transformer_reinforce", "rnn_reinforce"]:
            loss, aux_info, detached_message = self.reinforce_game(speaker_input, listener_input, input_dict["label"], input_dict=input_dict)
        else:
            loss, aux_info, detached_message = self.gs_game(speaker_input, listener_input, input_dict["label"], input_dict=input_dict)
        
        output_dict = dict()

        output_dict['loss'] = loss
        output_dict['message'] = detached_message
        output_dict.update(aux_info)
        return output_dict






if __name__ == "__main__":
    b = 32
    dummy_input_dict = {
        'image': torch.rand(b, 16, 160, 160).cuda(),
        # 'label': F.one_hot(torch.randint(0, 8, (b, )), num_classes=8).float().cuda()
        'label': torch.randint(0, 8, (b, )).cuda()
    }

    dummy_args = Args()
    model = EC(dummy_args).cuda()
    dummy_output_dict = model(dummy_input_dict)
    print(dummy_output_dict['main_loss'])
    print(dummy_output_dict['aux_loss'])
